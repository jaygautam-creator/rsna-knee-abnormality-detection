# %% [markdown]
# # v2 inference — submission from the chunked fold-0 checkpoint
#
# Adapted from `infer_v0`. Three things differ and all three are load-bearing:
#
# * **24 slices, not 16.** The checkpoint was trained on `cache_volumes_24`;
#   feeding it 16 would change the attention pooling's input length and quietly
#   degrade the score with nothing to point at.
# * **convnext_nano with the two-head architecture.** `Net` here is a verbatim
#   copy of the training definition, including the permutation reassembly. It is
#   duplicated rather than imported because the kernel has no access to the repo.
# * **Horizontal-flip TTA.** Train-time augmentation was a width flip, so the
#   model has seen both orientations; averaging them is close to free.
#
# The checkpoint is `train_v3` fold 0: 4,400 steps (~10 epochs) trained across 11
# subprocess chunks, LR fully annealed, loss 0.4818 -> 0.2231. At 513 steps the
# same architecture and labels scored report_auc 0.7608; at 4,400 it scores
# 0.8694. This is the first checkpoint that is not undertrained.
#
# Preprocessing below must match `cache_volumes_24` exactly — same percentile
# window, same slice ordering, same series pick.

# %%
import ctypes
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
ATTN_CONDS = ["Medial Meniscus", "Lateral Meniscus", "MCL", "Fracture", "Baker's"]

BACKBONE = "convnext_nano"
D_SLICES = 24
SIZE = 224
PLANES = ["Axial", "Coronal", "Sagittal"]
BATCH = 8
CKPT_NAME = "v3_fold0.pt"
TTA_FLIP = True

# Training leaked ~43 MB/step of host RAM in the main process with no workers,
# no mmap and no pinned buffers — the signature of glibc retaining freed large
# blocks rather than a Python-visible leak. Inference is only ~1/4 the steps of
# a training epoch, but it runs at the end of nothing and a kill here costs the
# whole submission, so trim the heap periodically as insurance.
try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def trim_heap() -> None:
    if _libc is not None:
        _libc.malloc_trim(0)


def mem() -> str:
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 2**30
    avail = 0.0
    for ln in Path("/proc/meminfo").read_text().splitlines():
        p = ln.split()
        if p[0] == "MemAvailable:":
            avail = int(p[1]) / 2**20
    return f"rss {rss:.1f}G | avail {avail:.1f}G"


dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| gpus:", torch.cuda.device_count())


def find_dir(*markers: str, max_depth: int = 4) -> Path:
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if all((d / m).exists() for m in markers):
            return d
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except PermissionError:
            continue
        for child in children:
            if child.name in ("train_series", "test_series"):
                continue
            frontier.append((child, depth + 1))
    print(f"--- could not find {markers}; /kaggle/input tree (depth 3) ---")
    for p in sorted(Path("/kaggle/input").rglob("*")):
        rel = p.relative_to("/kaggle/input")
        if len(rel.parts) <= 3:
            print(" ", rel, "/" if p.is_dir() else "")
    raise FileNotFoundError(f"no mount containing {markers}")


COMP = find_dir("test.csv", "test_series.csv", "sample_submission.csv")
CKPT = find_dir(CKPT_NAME)
print("comp:", COMP, "| ckpt:", CKPT)

test = pd.read_csv(COMP / "test.csv")
series = pd.read_csv(COMP / "test_series.csv")
sample = pd.read_csv(COMP / "sample_submission.csv")
print(f"test studies {len(test)}  series {len(series)}")

# %% [markdown]
# ## Fail-safe submission first
#
# Written before the model runs so a crash downstream leaves a valid 0.5 file
# rather than no file. Overwritten with real predictions on success.

# %%
fallback = pd.DataFrame({"StudyInstanceUID": test.StudyInstanceUID})
for c in LABELS:
    fallback[c] = 0.5
fallback[sample.columns].to_csv("submission.csv", index=False)
print("fallback submission.csv written")

# %% [markdown]
# ## Series selection + decode — identical to cache_volumes_24

# %%
sel = series.copy()
sel["Fluid_Sensitive"] = sel["Fluid_Sensitive"].fillna(False).astype(bool)
n_slices = {}
for r in sel.itertuples():
    d = COMP / "test_series" / r.StudyInstanceUID / r.SeriesInstanceUID
    n_slices[r.SeriesInstanceUID] = len(list(d.glob("*.dcm"))) if d.exists() else 0
sel["n"] = sel.SeriesInstanceUID.map(n_slices)
sel = sel[sel.n > 0]
sel = (sel.sort_values(["Fluid_Sensitive", "n"], ascending=[False, False])
          .groupby(["StudyInstanceUID", "Anatomical_Plane"], as_index=False)
          .first())
sel = sel[sel.Anatomical_Plane.isin(PLANES)]
plan = {sid: g.set_index("Anatomical_Plane").SeriesInstanceUID.to_dict()
        for sid, g in sel.groupby("StudyInstanceUID")}
have = sel.groupby("StudyInstanceUID").Anatomical_Plane.nunique()
print(f"studies with 3/2/1 planes: {(have==3).sum()}/{(have==2).sum()}/{(have==1).sum()}")
print(f"studies with no usable series: {len(test) - len(have)}")


def _position(path: Path) -> float:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is not None and len(iop) == 6:
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            return float(np.dot(np.cross(r, c), np.array(ipp, float)))
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def _pixels(ds) -> np.ndarray:
    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + \
        float(getattr(ds, "RescaleIntercept", 0) or 0)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def load_volume(sdir: Path) -> np.ndarray:
    files = list(sdir.glob("*.dcm"))
    if not files:
        return np.zeros((D_SLICES, SIZE, SIZE), np.uint8)
    files.sort(key=_position)
    idx = np.linspace(0, len(files) - 1, D_SLICES).round().astype(int)
    out = np.zeros((D_SLICES, SIZE, SIZE), np.float32)
    for j, i in enumerate(idx):
        try:
            a = _pixels(pydicom.dcmread(files[i]))
            out[j] = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    lo, hi = np.percentile(out, [1, 99])
    if hi <= lo:
        return np.zeros((D_SLICES, SIZE, SIZE), np.uint8)
    return (np.clip((out - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


class TestDS(Dataset):
    """Decodes on __getitem__ so DataLoader workers overlap I/O with the GPU.

    Returns uint8 and defers the /255 to the GPU: a float32 batch is 4x the
    host memory of a uint8 one, and host allocation churn is the prime suspect
    in the training leak.
    """

    def __init__(self, sids):
        self.sids = sids

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        sid = self.sids[i]
        vol = np.zeros((3, D_SLICES, SIZE, SIZE), np.uint8)
        for p, plane in enumerate(PLANES):
            suid = plan.get(sid, {}).get(plane)
            if suid:
                vol[p] = load_volume(COMP / "test_series" / sid / suid)
        return torch.from_numpy(vol), i

# %% [markdown]
# ## Model — a verbatim copy of the training definition
#
# Any drift here loads the checkpoint into a different network. `strict=True`
# below is the guard: a silently-unloaded head emits noise that passes every
# shape check and only surfaces as a ~0.5 leaderboard score.

# %%
class Net(nn.Module):
    def __init__(self, name=BACKBONE, n_planes=3):
        super().__init__()
        # pretrained=False: internet is disabled, weights come from the checkpoint
        self.backbone = timm.create_model(name, pretrained=False, in_chans=1, num_classes=0)
        f = self.backbone.num_features
        self.att = nn.Sequential(nn.Linear(f, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attn_idx = [LABELS.index(c) for c in ATTN_CONDS]
        self.mean_idx = [i for i in range(len(LABELS)) if i not in self.attn_idx]
        self.head_attn = nn.Sequential(nn.Dropout(0.3),
                                       nn.Linear(f * n_planes, len(self.attn_idx)))
        self.head_mean = nn.Sequential(nn.Dropout(0.3),
                                       nn.Linear(f * n_planes, len(self.mean_idx)))
        order = self.attn_idx + self.mean_idx
        inv = [0] * len(LABELS)
        for pos, lab in enumerate(order):
            inv[lab] = pos
        self.register_buffer("inv_perm", torch.tensor(inv), persistent=False)

    def forward(self, x):
        B, P, D, H, W = x.shape
        f = self.backbone(x.reshape(B * P * D, 1, H, W)).reshape(B, P, D, -1)
        g_mean = f.mean(2).reshape(B, -1)
        w = torch.softmax(self.att(f), dim=2)
        g_attn = (f * w).sum(2).reshape(B, -1)
        return torch.cat([self.head_attn(g_attn),
                          self.head_mean(g_mean)], dim=1)[:, self.inv_perm]


model = Net()
sd = torch.load(CKPT / CKPT_NAME, map_location="cpu")
model.load_state_dict(sd, strict=True)
model = model.to(dev).eval()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print("checkpoint loaded")

# %%
sids = test.StudyInstanceUID.tolist()
dl = DataLoader(TestDS(sids), batch_size=BATCH, shuffle=False, num_workers=4)

P = np.zeros((len(sids), len(LABELS)), np.float32)
t0 = time.time()
with torch.no_grad():
    for step, (xb, ib) in enumerate(dl):
        xb = xb.to(dev, non_blocking=True).float().div_(255.0)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            out = torch.sigmoid(model(xb))
            if TTA_FLIP:
                out = (out + torch.sigmoid(model(torch.flip(xb, dims=[4])))) / 2
        P[ib.numpy()] = out.float().cpu().numpy()
        if step % 50 == 0:
            trim_heap()
            done = int(ib[-1]) + 1
            el = time.time() - t0
            print(f"  {done}/{len(sids)}  {done/el:.1f} study/s  "
                  f"eta {(len(sids)-done)/(done/el)/60:.1f} min  {mem()}", flush=True)

el = time.time() - t0
print(f"\ninferred {len(sids)} studies in {el/60:.1f} min "
      f"({len(sids)/el:.1f} study/s)  {mem()}")

# %%
sub = pd.DataFrame({"StudyInstanceUID": sids})
for i, c in enumerate(LABELS):
    sub[c] = P[:, i]
sub = sub[sample.columns]
sub.to_csv("submission.csv", index=False)

chk = pd.read_csv("submission.csv")
assert list(chk.columns) == list(sample.columns), "column mismatch"
assert len(chk) == len(test), f"row count {len(chk)} != {len(test)}"
assert chk.StudyInstanceUID.is_unique, "duplicate StudyInstanceUID"
assert set(chk.StudyInstanceUID) == set(test.StudyInstanceUID), "UID set mismatch"
assert chk[LABELS].notna().all().all(), "NaN in predictions"
v = chk[LABELS].to_numpy()
assert v.min() >= 0 and v.max() <= 1, "predictions outside [0,1]"
# A constant column scores exactly 0.5 — catch a dead head before the LB does.
assert (chk[LABELS].nunique() > 1).all(), \
    f"constant column(s): {chk[LABELS].nunique()[chk[LABELS].nunique() <= 1].index.tolist()}"

print(f"submission.csv OK — {len(chk)} rows x {len(chk.columns)} cols")
print(chk[LABELS].describe().loc[["min", "mean", "max"]].round(3).to_string())
json.dump({"backbone": BACKBONE, "d_slices": D_SLICES, "ckpt": CKPT_NAME,
           "tta_flip": TTA_FLIP, "n_test": len(test),
           "minutes": round(el / 60, 1)},
          open("infer_v2_meta.json", "w"), indent=2)
