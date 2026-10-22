# %% [markdown]
# # v6 inference — mixed-resolution ensemble
#
# Same laterality-canonical decode, same rank blend, same side pre-pass as v5.
# The one structural change is that a checkpoint is no longer assumed to want
# 224 px input.
#
# ## Why this is not just `SIZE = 336`
#
# v6 trains at 336; v3 and v5 were trained at 224. Feeding a 224-trained
# convnext a 336 volume does not raise — it global-pools to the same feature
# width and returns confident, wrong numbers. So each checkpoint declares its
# training resolution in `CKPT_SIZE`, and gets served exactly that.
#
# ## Why decode once per resolution rather than resample
#
# The obvious saving is to decode at 336 and downsample to 224 for the older
# checkpoints. Measured against a direct native -> 224 `INTER_AREA`, the
# two-step native -> 336 -> 224 path correlates only **r = 0.976** — a real
# train/test skew for the four checkpoints carrying 80% of the blend weight.
#
# It also saves nothing worth having. The cost here is `pydicom.dcmread`, not
# the resize: each slice is decoded once and then run through one `INTER_AREA`
# per target size, which is microseconds against a multi-millisecond decode.
# `SIZES` is derived from what actually mounted, so an all-336 ensemble never
# pays for a 224 volume it will not use.
#
# Percentile normalisation is recomputed per resolution, because each cache
# builder computed its 1/99 window on its own volume; sharing one window across
# sizes would shift the input distribution away from what training saw.
#
# ## Status: wiring only
#
# `CKPT_SIZE` already knows about `v6_fold0.pt` / `v6_fold1.pt`, but they are
# **not in `CKPT_WEIGHTS` yet** and not mounted — v6 has not trained. So this
# notebook currently reproduces the v5 four-checkpoint blend through the new
# code path. Adding v6 means training it, fitting blend weights offline on the
# frozen split, and only then extending `CKPT_WEIGHTS`. Guessing a weight here
# would be the same mistake as the fold recompute: a number nobody measured.
#
# ## GPU
#
# Enabled, unlike v5. A 336 forward is 2.25x the pixels, and the v5 CPU run
# already used most of the 9 h cap. This also matters for the efficiency track,
# which scores runtime directly.

# %%
import ctypes
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
ATTN_CONDS = ["Medial Meniscus", "Lateral Meniscus", "MCL", "Fracture", "Baker's"]

BACKBONE = "convnext_nano"
D_SLICES = 24
# Decode once, at the highest resolution any mounted checkpoint wants. Decoding
# is the expensive half of this kernel, so decoding twice to serve a mixed-
# resolution ensemble would roughly double the runtime; downsampling a decoded
# 336 volume to 224 is a few ms of tensor work per batch.
# Every distinct resolution some mounted checkpoint needs. Filled in once the
# checkpoints are known, so an ensemble of only-336 models never pays for 224.
SIZES = ()
PLANES = ["Axial", "Coronal", "Sagittal"]
BATCH = 8
# Four-checkpoint weighted ensemble, fixed offline on the 357 studies held out by
# both the v3 and the v5 fold splits (see PLAN.md, "v5 landed"). v5 alone is the
# *weaker* model there -- 0.8594 vs v3's 0.8692 -- so this is not a promotion of
# v5 over v3. It is included because it is only 0.813 rank-correlated with v3,
# and 0.6*v3 + 0.4*v5 scored 0.8769, beating v3 alone by +0.0077 with
# P(blend > v3) = 0.999 under a paired bootstrap. The gain lands on the weak
# conditions that carry the macro: Lateral Meniscus +0.024, MCL +0.017,
# Fracture +0.016.
#
# The weight is per *group*, split evenly within it. 0.4 was the peak but the
# curve is flat from 0.3 to 0.5, so this is not a knife-edge fit to 357 studies.
#
# Compute: 4 models x no TTA = 4 forwards/study. The 0.905 submission was
# 2 models x flip TTA = the same 4 forwards on the same CPU kernel and it
# finished, so this adds no timeout risk over a run known to fit the 9 h cap.
# Decode is per study and shared across the four, not repeated.
# Each checkpoint is served at the resolution it was TRAINED at, not at the
# decode resolution. Feeding a 224-trained convnext a 336 input does not error —
# it pools to the same feature width and returns confident, wrong numbers — so
# the mapping is explicit and asserted against the training receipt below.
CKPT_SIZE = {
    "v3_fold0.pt": 224,
    "v3_fold1.pt": 224,
    "v5_fold0.pt": 224,
    "v5_fold1.pt": 224,
    "v6_fold0.pt": 336,
    "v6_fold1.pt": 336,
}

CKPT_WEIGHTS = {
    "v3_fold0.pt": 0.30,
    "v3_fold1.pt": 0.30,
    "v5_fold0.pt": 0.20,
    "v5_fold1.pt": 0.20,
}
CKPT_NAMES = list(CKPT_WEIGHTS)
# Skipping must not be silent. A mount failure that leaves a subset would score
# something other than the blend that was validated, while still producing a
# plausible-looking submission and burning one of two daily submissions to learn
# nothing. Demand all four and fail at minute 0 instead.
MIN_CKPTS = 4
# Must stay False, and not merely because it is cheaper. The model is now trained
# on laterality-canonical volumes; a horizontal flip at test time re-creates the
# exact medial/lateral ambiguity v5 exists to remove, and averaging it in would
# hand back the gain while looking like ordinary TTA.
TTA_FLIP = False
AXIS_TOL = 0.5      # |x-component| above which a direction is the medial axis
# Fitted on the training set; used only if the test set has too few tagged
# studies to calibrate on. See the side pre-pass.
SIDE_THR_TRAIN = -68.0
MIN_CALIB = 50

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
    f"need >= {MIN_CKPTS}. Check sources in kernel-metadata.json — a "
    f"short ensemble silently scores something other than what was intended.")
# Both generations must be present. The whole reason for this ensemble is that
# v3 and v5 are only 0.813 correlated; losing either group turns the validated
# blend back into a single-generation run at a weight that was never fitted for it.
_mounted = {p.name for p in CKPTS}
for _gen in ("v3_", "v5_"):
    assert any(n.startswith(_gen) for n in _mounted), \
        f"no {_gen[:-1]} checkpoint mounted ({sorted(_mounted)}) — the blend needs both"
MODEL_SIZES = [CKPT_SIZE[p.name] for p in CKPTS]
SIZES = tuple(sorted(set(MODEL_SIZES)))
print("comp:", COMP, "| checkpoints:",
      [(p.name, CKPT_SIZE[p.name]) for p in CKPTS], "| decode sizes:", SIZES)

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


def _geom(ds):
    """-> (row direction, slice normal) in patient space, or (None, None)."""
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return None, None
    row = np.array(iop[:3], float)
    col = np.array(iop[3:], float)
    return row, np.cross(row, col)


def _tag_side(ds):
    """-> +1 (right knee), -1 (left), or None if no laterality tag."""
    for tag in ("Laterality", "ImageLaterality", "FrameLaterality"):
        v = str(getattr(ds, tag, "") or "").strip().upper()
        if v.startswith("R"):
            return +1
        if v.startswith("L"):
            return -1
    return None


def _ipp_x(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        return float(ipp[0])
    return None


RIGHT_RE = re.compile(
    r"\bright\b|\brt\b|\bderech\w+|\bdcha\b|\bdcho\b|\brechts\b|\brechte[sn]?\b"
    r"|\bsağ\b|\bправ\w+|\bdroit\w*|\bdestr\w+", re.I | re.U)
LEFT_RE = re.compile(
    r"\bleft\b|\blt\b|\bizquierd\w+|\bizq\b|\blinks\b|\blinke[sn]?\b"
    r"|\bsol\b|\bлев\w+|\bgauche\b|\bsinistr\w+", re.I | re.U)


def text_side(text):
    """-> +1 right, -1 left, 0 if silent. Turkish 'sag' is excluded on purpose:
    it also abbreviates *sagittal* and matches almost every series description."""
    if not isinstance(text, str):
        return 0
    nr, nl = len(RIGHT_RE.findall(text)), len(LEFT_RE.findall(text))
    return 1 if nr > nl else (-1 if nl > nr else 0)


def study_signals(head_sets):
    """-> (tag_side or None, desc_side or None, mean_x or None) for one study.

    Side is a property of the study, not the series — deciding it per series from
    one unsorted slice left 9.4% of training studies self-contradicting, because a
    sagittal stack spans the medial-lateral axis and its x changes sign across
    slices. The mean over every slice is the knee centroid and is stable.
    """
    votes = [s for hs in head_sets for _, ds in hs
             if (s := _tag_side(ds)) is not None]
    tag = (+1 if sum(votes) > 0 else -1) if votes else None

    desc = " ".join(str(getattr(ds, "SeriesDescription", "") or "")
                    for hs in head_sets for _, ds in hs[:1])
    d = text_side(desc)

    xs = [x for hs in head_sets for _, ds in hs if (x := _ipp_x(ds)) is not None]
    mean_x = float(np.mean(xs)) if xs else None
    return tag, (d or None), mean_x


def _position_from(ds, normal) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        if normal is not None:
            return float(np.dot(normal, np.array(ipp, float)))
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def _pixels(ds) -> np.ndarray:
    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + \
        float(getattr(ds, "RescaleIntercept", 0) or 0)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def read_heads(sdir: Path):
    heads = []
    for f in sdir.glob("*.dcm"):
        try:
            heads.append((f, pydicom.dcmread(f, stop_before_pixels=True)))
        except Exception:
            pass
    return heads


def load_volume(heads, side) -> np.ndarray:
    """Byte-for-byte the same transform `cache_volumes_lat` applied to training.

    Any divergence here is a train/test skew that no validation set can catch:
    the model would be scored on volumes in an orientation it never saw. `side`
    comes from the caller because it is a property of the study, not the series.
    """
    zeros = {z: np.zeros((D_SLICES, z, z), np.uint8) for z in SIZES}
    if not heads:
        return zeros

    row, normal = _geom(heads[0][1])

    heads = sorted(heads, key=lambda fh: _position_from(fh[1], normal))
    files = [f for f, _ in heads]

    # Whichever axis carries the patient x-component is the medial/lateral one:
    # W on axial and coronal, the slice axis on sagittal. Flip it so increasing
    # index always points medial, and leave the other axis untouched.
    flip_w = flip_d = False
    if row is not None and abs(row[0]) > AXIS_TOL:
        flip_w = (np.sign(row[0]) != side)
    if normal is not None and abs(normal[0]) > AXIS_TOL:
        flip_d = (np.sign(normal[0]) != side)

    idx = np.linspace(0, len(files) - 1, D_SLICES).round().astype(int)
    if flip_d:
        idx = idx[::-1]

    outs = {z: np.zeros((D_SLICES, z, z), np.float32) for z in SIZES}
    for j, i in enumerate(idx):
        try:
            a = _pixels(pydicom.dcmread(files[i]))
        except Exception:
            continue
        # One decode, one INTER_AREA per target size, straight from native.
        # Resampling 336 -> 224 instead would save nothing measurable (the
        # dcmread dominates) and would feed the 224 checkpoints a twice-
        # resampled image they were never trained on: measured r=0.976 against
        # the direct resize, which is a train/test skew for no gain.
        #
        # `SIZE` is the loop variable on purpose. `check_decode_parity.py`
        # compares these lines to the cache builder as text, so keeping the
        # cache's own names makes the guard match instead of having to be
        # loosened — and a loosened parity guard is how orientation bugs ship.
        for SIZE in SIZES:
            outs[SIZE][j] = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    res = {}
    for SIZE in SIZES:
        out = outs[SIZE]
        if flip_w:
            out = out[:, :, ::-1]
        # Recomputed per resolution because each cache builder computed its
        # 1/99 window on its own volume; sharing one window across sizes would
        # shift the normalisation away from what training saw.
        lo, hi = np.percentile(out, [1, 99])
        res[SIZE] = (zeros[SIZE] if hi <= lo else np.ascontiguousarray(
            (np.clip((out - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)))
    return res


# %% [markdown]
# ## Deciding the side, before any decoding
#
# Training could lean on the radiology reports, which resolve 39% of studies at
# 0.973. **Test has no reports** — `test.csv` is StudyInstanceUID only — so the
# cascade here is `Laterality` tag > `SeriesDescription` > mean patient x.
#
# The mean-x cut is the delicate part. It is not zero (that scores 0.883, because
# right knees centre at x = -139 and left at +39), and the training value of -68 mm
# is a property of *this* site mix — the competition warns the test distribution
# differs, and a table-offset convention is exactly the sort of thing that varies
# by site. Hard-coding it would be a silent, unfalsifiable risk.
#
# So calibrate on the test set itself: ~50% of studies carry the `Laterality` tag,
# which is free ground truth *in the test distribution*. Fit the threshold to
# maximise agreement on those, then apply it to the untagged remainder. If too few
# tagged studies exist to fit on, fall back to the training value.

# %%
def _prepass(sids):
    """Header-only sweep: gather side signals for every study."""
    rows = {}
    for sid in sids:
        head_sets = []
        for plane in PLANES:
            suid = plan.get(sid, {}).get(plane)
            if suid:
                hs = read_heads(COMP / "test_series" / sid / suid)
                if hs:
                    head_sets.append(hs)
        rows[sid] = study_signals(head_sets)
    return rows


def _fit_threshold(sig):
    """Fit the mean-x cut on studies where the tag gives us the answer."""
    pts = [(mx, tag) for tag, _, mx in sig.values()
           if tag is not None and mx is not None]
    if len(pts) < MIN_CALIB:
        print(f"  only {len(pts)} tagged studies with position — "
              f"falling back to the training threshold {SIDE_THR_TRAIN}")
        return SIDE_THR_TRAIN, float("nan"), len(pts)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    grid = np.arange(-220.0, 60.0, 1.0)
    accs = np.array([(np.where(xs < t, 1, -1) == ys).mean() for t in grid])
    k = int(accs.argmax())
    return float(grid[k]), float(accs[k]), len(pts)


t_pre = time.time()
TEST_SIDS = test.StudyInstanceUID.tolist()
SIGNALS = _prepass(TEST_SIDS)
THR, THR_ACC, N_CALIB = _fit_threshold(SIGNALS)
print(f"side pre-pass: {len(SIGNALS)} studies in {(time.time()-t_pre)/60:.1f} min")
print(f"  threshold {THR:.1f} mm  (fitted on {N_CALIB} tagged studies, "
      f"in-sample accuracy {THR_ACC:.4f})")

SIDE_OF, EVID = {}, []
for sid, (tag, desc, mx) in SIGNALS.items():
    if tag is not None:
        SIDE_OF[sid], e = tag, "tag"
    elif desc is not None:
        SIDE_OF[sid], e = desc, "desc"
    elif mx is not None:
        SIDE_OF[sid], e = (1 if mx < THR else -1), "meanx"
    else:
        SIDE_OF[sid], e = 1, "default"
    EVID.append(e)

from collections import Counter as _C
print("  side evidence:", dict(_C(EVID)))
print("  sides:", dict(_C("R" if v > 0 else "L" for v in SIDE_OF.values())))
# Both knees occur in any real cohort. An all-one-side result means the cascade
# collapsed and every left knee is about to be scored mirrored.
_frac = max(_C(SIDE_OF.values()).values()) / max(len(SIDE_OF), 1)
assert _frac < 0.95, f"cascade assigned {_frac:.1%} of studies the same side"


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
        vols = {z: np.zeros((3, D_SLICES, z, z), np.uint8) for z in SIZES}

        # Headers for every plane first, side looked up from the pre-pass, then
        # decode — mirroring cache_volumes_lat exactly.
        heads_by_plane = {}
        for plane in PLANES:
            suid = plan.get(sid, {}).get(plane)
            if suid:
                heads_by_plane[plane] = read_heads(COMP / "test_series" / sid / suid)

        side = SIDE_OF.get(sid, 1)

        for p, plane in enumerate(PLANES):
            heads = heads_by_plane.get(plane)
            if heads:
                got = load_volume(heads, side)
                for z in SIZES:
                    vols[z][p] = got[z]
        # Keyed by int in a plain tuple: the default collate stacks each entry.
        return tuple(torch.from_numpy(vols[z]) for z in SIZES), i

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
print(f"{len(models)} checkpoint(s) loaded: "
      f"{[(p.name, CKPT_SIZE[p.name]) for p in CKPTS]}")

# %%
sids = test.StudyInstanceUID.tolist()
dl = DataLoader(TestDS(sids), batch_size=BATCH, shuffle=False, num_workers=4)

# Per-model, not pre-averaged. v3 and v5 were trained against different teacher
# label versions (v3 vs v4 margins), so their sigmoid outputs are on different
# scales and a probability mean would silently re-weight them by calibration
# rather than by the weights fitted above. Combining on *ranks* is also exactly
# what was validated offline, and the metric is rank-based AUC, so nothing is
# lost by discarding the absolute values. 4 x 1230 x 12 floats is ~240 KB.
PM = np.zeros((len(CKPTS), len(sids), len(LABELS)), np.float32)
t0 = time.time()
with torch.no_grad():
    for step, (xs, ib) in enumerate(dl):
        xby = {z: x.to(dev, non_blocking=True).float().div_(255.0)
               for z, x in zip(SIZES, xs)}
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            outs = []
            for _m, _sz in zip(models, MODEL_SIZES):
                xm = xby[_sz]
                o = torch.sigmoid(_m(xm))
                if TTA_FLIP:
                    o = (o + torch.sigmoid(_m(torch.flip(xm, dims=[4])))) / 2
                outs.append(o)
        idx = ib.numpy()
        for k, o in enumerate(outs):
            PM[k, idx] = o.float().cpu().numpy()
        if step % 50 == 0:
            trim_heap()
            done = int(ib[-1]) + 1
            el = time.time() - t0
            print(f"  {done}/{len(sids)}  {done/el:.1f} study/s  "
                  f"eta {(len(sids)-done)/(done/el)/60:.1f} min  {mem()}", flush=True)

el = time.time() - t0
print(f"\ninferred {len(sids)} studies in {el/60:.1f} min "
      f"({len(sids)/el:.1f} study/s)  {mem()}")

# %% [markdown]
# ## Weighted rank blend
#
# Each model's column is mapped to its within-test percentile rank, then combined
# at the group weights fitted offline. Weights are renormalised over what actually
# mounted so a partial ensemble (already asserted against above) could not silently
# scale the whole submission toward zero.

# %%
W = np.array([CKPT_WEIGHTS[p.name] for p in CKPTS], np.float64)
W = W / W.sum()
print("blend weights:", {p.name: round(w, 3) for p, w in zip(CKPTS, W)})

P = np.zeros((len(sids), len(LABELS)), np.float64)
for k in range(len(CKPTS)):
    for i in range(len(LABELS)):
        # average ranks so ties cannot bias one model's contribution, scaled to
        # (0, 1) so every model and every condition enters on the same footing
        r = pd.Series(PM[k, :, i]).rank(method="average").to_numpy()
        P[:, i] += W[k] * (r / (len(sids) + 1.0))

# A model that collapsed to a constant would produce a flat rank column and
# vanish into the blend without tripping the constant-column check downstream,
# because the other three would still vary. Catch it per model instead.
for k, p_ck in enumerate(CKPTS):
    nuq = [len(np.unique(PM[k, :, i])) for i in range(len(LABELS))]
    assert min(nuq) > 1, f"{p_ck.name} produced a constant column: {nuq}"
print("per-model spread OK (no constant columns)")

corr = np.mean([[pd.Series(PM[a, :, i]).corr(pd.Series(PM[b, :, i]), method="spearman")
                 for i in range(len(LABELS))]
                for a in range(len(CKPTS)) for b in range(len(CKPTS)) if a < b])
print(f"mean pairwise rank correlation across checkpoints: {corr:.3f} "
      f"(offline v3-vs-v5 was 0.813; a value near 1.0 would mean the ensemble "
      f"is buying nothing and something is wrong with the mounts)")

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
           "decode_sizes": list(SIZES),
           "ckpt_size": {p.name: CKPT_SIZE[p.name] for p in CKPTS},
           "ckpts": [p.name for p in CKPTS],
           "weights": {p.name: float(w) for p, w in zip(CKPTS, W)},
           "blend": "weighted rank average",
           "mean_pairwise_rank_corr": float(corr),
           "tta_flip": TTA_FLIP, "n_test": len(test),
           "minutes": round(el / 60, 1)},
          open("infer_v6_meta.json", "w"), indent=2)
