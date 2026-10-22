# %% [markdown]
# # v0 inference — real leaderboard submission
#
# Internet is disabled here, so `timm` weights come from our own checkpoint, not
# a download. The checkpoint arrives as attached kernel output.
#
# **The preprocessing below must match `cache_volumes` exactly.** Training saw
# volumes built by that notebook; any drift here — a different percentile window,
# a different slice order, a different series pick — changes the input
# distribution and silently degrades the score with nothing to point at. The
# shared logic is duplicated verbatim rather than "improved", and the assertions
# at the end check shape/order rather than trusting it.
#
# Runtime budget is 9 h. The cache run measured ~2.9 study/s on 4 cores decoding
# 3 series x 16 slices, so even 5,000 test studies is well under an hour; the
# DataLoader overlaps decode with GPU so the two costs do not add.

# %%
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

BACKBONE = "resnet18"
POOL = "attn"
D_SLICES = 16
SIZE = 224
PLANES = ["Axial", "Coronal", "Sagittal"]
BATCH = 8

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


CKPT_NAME = f"v0_{POOL}_fold0.pt"
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
# ## Series selection + decode — identical to cache_volumes

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
    """Decodes on __getitem__ so DataLoader workers overlap I/O with the GPU."""

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
        return torch.from_numpy(vol.astype(np.float32) / 255.0), i

# %% [markdown]
# ## Model — architecture must match training exactly

# %%
class Net(nn.Module):
    def __init__(self, name=BACKBONE, n_planes=3, n_out=len(LABELS), pool=POOL):
        super().__init__()
        # pretrained=False: internet is disabled, weights come from the checkpoint
        self.backbone = timm.create_model(name, pretrained=False, in_chans=1, num_classes=0)
        f = self.backbone.num_features
        self.pool = pool
        if pool == "attn":
            self.att = nn.Sequential(nn.Linear(f, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(f * n_planes, n_out))

    def forward(self, x):
        B, P, D, H, W = x.shape
        f = self.backbone(x.reshape(B * P * D, 1, H, W)).reshape(B, P, D, -1)
        if self.pool == "mean":
            g = f.mean(2)
        elif self.pool == "max":
            g = f.amax(2)
        else:
            w = torch.softmax(self.att(f), dim=2)
            g = (f * w).sum(2)
        return self.head(g.reshape(B, -1))


model = Net()
sd = torch.load(CKPT / CKPT_NAME, map_location="cpu")
# strict=True on purpose: a silently-unloaded head would emit noise that still
# passes every shape check and only shows up as a ~0.5 leaderboard score.
model.load_state_dict(sd, strict=True)
model = model.to(dev).eval()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print("checkpoint loaded")

# %%
sids = test.StudyInstanceUID.tolist()
dl = DataLoader(TestDS(sids), batch_size=BATCH, shuffle=False,
                num_workers=4, pin_memory=True)

P = np.zeros((len(sids), len(LABELS)), np.float32)
t0 = time.time()
with torch.no_grad():
    for xb, ib in dl:
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            out = torch.sigmoid(model(xb.to(dev, non_blocking=True)))
        P[ib.numpy()] = out.float().cpu().numpy()
        done = int(ib[-1]) + 1
        if done % 400 < BATCH:
            el = time.time() - t0
            print(f"  {done}/{len(sids)}  {done/el:.1f} study/s  "
                  f"eta {(len(sids)-done)/(done/el)/60:.1f} min", flush=True)

el = time.time() - t0
print(f"\ninferred {len(sids)} studies in {el/60:.1f} min "
      f"({len(sids)/el:.1f} study/s)")
print(f"projected at 5,000 studies: {5000/(len(sids)/el)/3600:.2f} h of the 9 h cap")

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
