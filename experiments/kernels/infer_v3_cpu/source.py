# %% [markdown]
# # v3 inference (CPU) — budget-adaptive fold ensemble
#
# A CPU twin of `infer_v3`, for when the weekly GPU quota is spent. Kaggle blocks
# *saving* a GPU notebook once the quota is gone, but a competition **submission
# rerun** still executes — so a CPU-saved version is a way to score with no quota.
#
# CPU is ~50x slower than the T4, which makes the 9 h cap the binding constraint
# rather than accuracy. The number of forward passes per study (folds x TTA) is
# therefore chosen **at runtime** from the actual hidden test set size; see the
# budget cell. A finished 1-fold run beats a timed-out 2-fold one.
#
# Same network and preprocessing as `infer_v2` (which scored **0.896**); the only
# change is that this averages fold checkpoints instead of running one.
# Everything else is held fixed on purpose, so any LB movement is attributable to
# the ensemble alone.
#
# * **fold 0** — `report_auc` 0.8694, the checkpoint that scored 0.896 solo.
# * **fold 1** — `report_auc` 0.8780, trained on a different 80%, never submitted.
#
# The folds are complementary rather than redundant, which is what makes the
# average worth more than either alone: fold 1 is better on Fracture (0.8961 vs
# 0.8674), Medial OA (0.9230 vs 0.8807) and Lateral OA (0.8920 vs 0.8693), while
# fold 0 is better on Baker's (0.8706 vs 0.8469) and Contusion (0.8916 vs 0.8521).
#
# Cost is one decode pass and two forward passes — decode dominates, so a 2-fold
# ensemble is far cheaper than 2x. Each study is decoded once and scored by both
# folds, each with horizontal-flip TTA (4 forwards/study).
#
# **24 slices, not 16**, and **convnext_nano with the two-head architecture**:
# `Net` is a verbatim copy of the training definition including the permutation
# reassembly, duplicated rather than imported because the kernel has no repo
# access. Preprocessing must match `cache_volumes_24` exactly — same percentile
# window, same slice ordering, same series pick.

# %%
import ctypes
import json
import os
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
# Fold ensemble. Each fold trains on a different 80% and its errors are only
# partly shared with the others, so averaging their sigmoid outputs recovers a
# little of what a single fold overfits. Missing checkpoints are skipped rather
# than fatal — a fold that has not finished training should cost us its share of
# the ensemble, not the whole submission.
#
# fold 2 is listed but not yet trained; it is deliberately absent from
# kernel-metadata.json's kernel_sources (Kaggle rejects a push that names a
# kernel which does not exist). Train it, add the source back, and it joins the
# ensemble with no code change.
CKPT_NAMES = ["v3_fold0.pt", "v3_fold1.pt", "v3_fold2.pt"]
# ...but skipping must not be silent. A mount failure that leaves only fold 0
# would reproduce infer_v2's 0.896 exactly while looking like a successful
# ensemble run, burning one of two daily submissions to learn nothing. Fail at
# minute 0 instead of scoring the wrong thing an hour later.
MIN_CKPTS = 2
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
CKPTS = []
for _name in CKPT_NAMES:
    try:
        CKPTS.append(find_dir(_name) / _name)
    except FileNotFoundError:
        print(f"  {_name} not mounted — excluded from the ensemble")
assert len(CKPTS) >= MIN_CKPTS, (
    f"only {len(CKPTS)} of {CKPT_NAMES} mounted ({[p.name for p in CKPTS]}); "
    f"need >= {MIN_CKPTS}. Check kernel_sources in kernel-metadata.json — a "
    f"1-checkpoint run just reproduces infer_v2 and wastes a submission.")
print("comp:", COMP, "| checkpoints:", [p.name for p in CKPTS])

test = pd.read_csv(COMP / "test.csv")
series = pd.read_csv(COMP / "test_series.csv")
sample = pd.read_csv(COMP / "sample_submission.csv")
print(f"test studies {len(test)}  series {len(series)}")

# %% [markdown]
# ## Fit the ensemble to the time budget
#
# On CPU the cost is linear in the number of forward passes per study
# (folds x TTA), and the hidden test set size is not known until this cell runs —
# the visible stub is 3 studies while the real set may be thousands. A config
# that is correct for 1,000 studies times out at 4,000 and scores nothing.
#
# So rather than hardcode a config, spend the budget: measure `n_test`, then take
# the richest (folds, TTA) combination that still fits. A 2-fold ensemble scores
# best but a finished 1-fold run beats a timed-out 2-fold one.

# %%
SEC_PER_PASS = 6.5      # measured: 26 s/study over 4 passes, CPU, 3 planes x 24 slices
BUDGET_SEC = 8.0 * 3600  # 9 h cap minus an hour for startup, decode jitter and I/O

torch.set_num_threads(os.cpu_count() or 4)

_options = []                                   # (passes, n_folds, tta) richest first
for _nf in range(len(CKPTS), 0, -1):
    for _tta in (True, False):
        _options.append((_nf * (2 if _tta else 1), _nf, _tta))
_options.sort(key=lambda o: -o[0])

N_FOLDS, TTA_FLIP = 1, False
for _passes, _nf, _tta in _options:
    if len(test) * _passes * SEC_PER_PASS <= BUDGET_SEC:
        N_FOLDS, TTA_FLIP = _nf, _tta
        break
else:
    print("WARNING: even 1 fold without TTA may exceed the budget; running leanest.")

CKPTS = CKPTS[:N_FOLDS]
_est = len(test) * N_FOLDS * (2 if TTA_FLIP else 1) * SEC_PER_PASS / 3600
print(f"budget: {len(test)} studies -> {N_FOLDS} fold(s), TTA={TTA_FLIP}, "
      f"est {_est:.1f} h of {BUDGET_SEC/3600:.0f} h")
print("using:", [p.name for p in CKPTS])

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


models = []
for _p in CKPTS:
    _m = Net()
    # strict=True on purpose: a silently ignored key means a head or the
    # attention pool is running with random weights, which looks like a merely
    # mediocre score rather than a bug.
    _m.load_state_dict(torch.load(_p, map_location="cpu"), strict=True)
    _m = _m.to(dev).eval()
    if torch.cuda.device_count() > 1:
        _m = nn.DataParallel(_m)
    models.append(_m)
print(f"{len(models)} checkpoint(s) loaded: {[p.name for p in CKPTS]}")

# %%
sids = test.StudyInstanceUID.tolist()
dl = DataLoader(TestDS(sids), batch_size=BATCH, shuffle=False, num_workers=4)

P = np.zeros((len(sids), len(LABELS)), np.float32)
t0 = time.time()
with torch.no_grad():
    for step, (xb, ib) in enumerate(dl):
        xb = xb.to(dev, non_blocking=True).float().div_(255.0)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            # Average in probability space, not logit space: a fold that is
            # wildly confident and wrong would dominate a logit mean.
            acc, n = 0.0, 0
            for _m in models:
                acc = acc + torch.sigmoid(_m(xb))
                n += 1
                if TTA_FLIP:
                    acc = acc + torch.sigmoid(_m(torch.flip(xb, dims=[4])))
                    n += 1
            out = acc / n
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
json.dump({"backbone": BACKBONE, "d_slices": D_SLICES,
           "ckpts": [p.name for p in CKPTS],
           "tta_flip": TTA_FLIP, "n_test": len(test),
           "device": dev, "n_folds": N_FOLDS,
           # Actual vs. predicted cost — recalibrate SEC_PER_PASS from this if the
           # budget cell picked a config that finished with hours to spare.
           "sec_per_pass_actual": round(
               el / max(len(test) * N_FOLDS * (2 if TTA_FLIP else 1), 1), 2),
           "sec_per_pass_assumed": SEC_PER_PASS,
           "minutes": round(el / 60, 1)},
          open("infer_v3_meta.json", "w"), indent=2)
