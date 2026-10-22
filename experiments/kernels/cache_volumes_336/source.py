# %% [markdown]
# # Volume cache, 336 px, laterality-canonicalised — shard SHARD_PLACEHOLDER of 3
#
# Everything about `cache_volumes_lat` carried forward unchanged — 3 planes, 24
# slices, fluid-sensitive preferred, medial/lateral canonicalised per study — with
# exactly one thing different: **in-plane resolution goes from 224 px to 336 px.**
#
# ## Why now, and why before the retrain
#
# The five folds have to be retrained against the frozen split in `data/folds.csv`
# (`src/labels/folds.py`) before any offline comparison means anything again. That
# retrain is ~40 h of GPU. Doing it at 224 and then again at 336 is ~80 h against
# roughly 240 h of quota for the rest of the competition. So the resolution decision
# has to be made *now*, before the folds are trained, not after.
#
# ## Why this notebook is sharded and the 224 one was not
#
# Bytes scale with the square of the edge:
#
# ```
# 224 px:  3 x 24 x 224 x 224 x 4407 = 15.92 GB   (measured, fit in one kernel)
# 336 px:  3 x 24 x 336 x 336 x 4407 = 35.82 GB   (does not)
# ```
#
# `/kaggle/working` caps at 20 GB, so a single-kernel 336 cache is impossible. This
# notebook writes **studies `[row_lo, row_hi)` only**, and three copies of it —
# `_s0`, `_s1`, `_s2` — cover the corpus at ~11.9 GB each.
#
# Sharded by **study row range, not by plane**. Both fit under the cap, but a row
# shard keeps each training sample's `(3, 24, 336, 336)` slab contiguous inside one
# file, so the dataloader still does one 8.1 MB sequential read per sample. A plane
# shard would turn every sample into three seeks across three memmaps.
#
# `index.csv` carries the **global** row id, so the trainer maps
# `global_row -> (shard, local_row)` and is otherwise unchanged. It already reads
# `d_slices` and `size` out of `cache_meta.json`, so nothing in `train_v5` hardcodes
# 224 — but **`infer_v5/source.py` does** (`SIZE = 224`, line 69). Inference must be
# moved to 336 in the same change or every offline number will be measured against a
# decode the submission does not reproduce. `make check-lat` exists to catch exactly
# that; run it before pushing any 336 training kernel.
#
# ## Is 336 actually buying resolution?
#
# Open question this run answers rather than assumes. If the source DICOMs are mostly
# ~320 px in-plane, then 336 is an *upsample* for much of the corpus: 2.25x the bytes
# and 2.25x the dataloader I/O for interpolated detail that was never acquired. The
# headers are already being read for the geometry, so `Rows`/`Columns` cost nothing
# extra — the native-size distribution is printed and written to `native_size.csv`
# below. Read it before committing 40 h of GPU to this cache.
#
# ## Everything below this line is unchanged from `cache_volumes_lat`
#
# The series selection, the per-study side cascade, and its gates are copied
# verbatim so this run isolates the resolution change. The laterality reasoning —
# why medial is the slice axis on sagittal, why side is decided once per study, why
# the cut is `SIDE_THR` and not zero — is documented in that notebook and holds here
# without change.
#
# Output: `volumes.npy` (n_shard, 3, 24, 336, 336) uint8 + `index.csv` (global row
# ids) + `cache_meta.json` + `sides.csv` + `geometry.csv` + `native_size.csv`.

# %%
import json
import os
import re
import time
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings("ignore")

D_SLICES = 24          # unchanged from cache_volumes_lat — this run isolates resolution
SIZE = 336             # 224 -> 336. The only thing this notebook changes.

# Which slice of the corpus this copy writes. 336 px puts the full cache at 35.8 GB
# and /kaggle/working caps at 20, so the run is split three ways; see the header.
SHARD = 0
N_SHARDS = 3
PLANES = ["Axial", "Coronal", "Sagittal"]
AXIS_TOL = 0.5         # |x-component| above which a direction counts as the medial axis
# Right knees cluster at mean patient x = -139 and left knees at +39, so the
# separating cut is nowhere near zero. Tuned on half the tag-labelled studies and
# scored on the other half: sign(x) gives 0.883, this gives 0.974.
SIDE_THR = -68.0
OUT = Path("/kaggle/working")

print(f"per study: {3 * D_SLICES * SIZE * SIZE / 1e6:.1f} MB")
print(f"shard {SHARD} of {N_SHARDS}")


def find_mount(max_depth: int = 3) -> Path:
    """BFS — the mount sits at /kaggle/input/competitions/<slug>, not depth 1."""
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if (d / "train.csv").exists() and (d / "train_series.csv").exists():
            return d
        if depth >= max_depth:
            continue
        for child in sorted(p for p in d.iterdir() if p.is_dir()):
            if child.name in ("train_series", "test_series"):
                continue
            frontier.append((child, depth + 1))
    raise FileNotFoundError("competition mount not found")


COMP = find_mount()
train = pd.read_csv(COMP / "train.csv")
series = pd.read_csv(COMP / "train_series.csv")
print("mount:", COMP)
print(f"studies {len(train)}  series {len(series)}  cores {os.cpu_count()}")

# %% [markdown]
# ## Series selection — identical to cache_volumes_24
#
# One series per plane per study, fluid-sensitive first, slice count breaking ties.
# Kept the same on purpose: if this cache changes two things at once we cannot read
# the training result. Widening the selection to the discarded non-fluid PD/T1
# series is the *next* experiment, not this one.

# %%
sel = series.copy()
sel["Fluid_Sensitive"] = sel["Fluid_Sensitive"].fillna(False).astype(bool)

n_slices = {}
for r in sel.itertuples():
    d = COMP / "train_series" / r.StudyInstanceUID / r.SeriesInstanceUID
    n_slices[r.SeriesInstanceUID] = len(list(d.glob("*.dcm"))) if d.exists() else 0
sel["n"] = sel.SeriesInstanceUID.map(n_slices)
sel = sel[sel.n > 0]

sel = (sel.sort_values(["Fluid_Sensitive", "n"], ascending=[False, False])
          .groupby(["StudyInstanceUID", "Anatomical_Plane"], as_index=False)
          .first())
sel = sel[sel.Anatomical_Plane.isin(PLANES)]

have = sel.groupby("StudyInstanceUID").Anatomical_Plane.nunique()
print(f"studies with 3/2/1 planes: "
      f"{(have == 3).sum()}/{(have == 2).sum()}/{(have == 1).sum()}")
print(f"studies with no usable series: {len(train) - len(have)}")

plan = {sid: g.set_index("Anatomical_Plane").SeriesInstanceUID.to_dict()
        for sid, g in sel.groupby("StudyInstanceUID")}
reports = dict(zip(train.StudyInstanceUID, train.Report.fillna("")))

# %% [markdown]
# ## Geometry
#
# `_geom` pulls everything orientation-related off one header. `_side` returns +1 for
# a knee whose medial direction is patient +x (a right knee) and -1 for a left, plus
# a tag saying which evidence was used so the fallback rate is auditable rather than
# assumed.

# %%
def _geom(ds):
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


# Laterality words across the languages in this corpus. Two traps, both of which
# cost accuracy when they were in: Turkish "sag" (right) is also the universal
# abbreviation for *sagittal* and matches nearly every series description, so only
# the dotted "sağ" is usable; and bare "r"/"l" match far too much to be worth their
# coverage.
RIGHT_RE = re.compile(
    r"\bright\b|\brt\b|\bderech\w+|\bdcha\b|\bdcho\b|\brechts\b|\brechte[sn]?\b"
    r"|\bsağ\b|\bправ\w+|\bdroit\w*|\bdestr\w+", re.I | re.U)
LEFT_RE = re.compile(
    r"\bleft\b|\blt\b|\bizquierd\w+|\bizq\b|\blinks\b|\blinke[sn]?\b"
    r"|\bsol\b|\bлев\w+|\bgauche\b|\bsinistr\w+", re.I | re.U)


def text_side(text):
    """-> +1 right, -1 left, 0 if silent or self-contradictory."""
    if not isinstance(text, str):
        return 0
    nr, nl = len(RIGHT_RE.findall(text)), len(LEFT_RE.findall(text))
    return 1 if nr > nl else (-1 if nl > nr else 0)


def study_side(head_sets, report=""):
    """Decide the knee side once for the whole study, best evidence first.

    Measured on the 2,196 studies that carry a `Laterality` tag (treated as truth):

    | signal            | coverage | accuracy |
    |-------------------|----------|----------|
    | SeriesDescription |    11.9% |   0.9847 |
    | report            |    39.4% |   0.9734 |
    | mean-x @ SIDE_THR |     100% |   0.9741 |

    head_sets: list of [(path, ds), ...], one per selected series.
    -> (+1 if medial is patient +x (a right knee) else -1, evidence, diagnostics)
    """
    votes = [s for hs in head_sets for _, ds in hs
             if (s := _tag_side(ds)) is not None]
    tag_side = (+1 if sum(votes) > 0 else -1) if votes else None

    desc = " ".join(str(getattr(ds, "SeriesDescription", "") or "")
                    for hs in head_sets for _, ds in hs[:1])
    desc_side = text_side(desc) or None
    rep_side = text_side(report) or None

    xs = [x for hs in head_sets for _, ds in hs if (x := _ipp_x(ds)) is not None]
    mean_x, ipp_side = None, None
    if xs:
        mean_x = float(np.mean(xs))
        # DICOM patient +x is the patient's LEFT, so a right knee sits at more
        # negative x. The cut is SIDE_THR, not 0 — see the constant.
        ipp_side = +1 if mean_x < SIDE_THR else -1

    diag = {"tag_side": tag_side, "desc_side": desc_side, "rep_side": rep_side,
            "ipp_side": ipp_side, "mean_x": mean_x,
            "n_tag_votes": len(votes), "n_slices": len(xs)}

    for side, evidence in ((tag_side, "tag"), (desc_side, "desc"),
                           (rep_side, "report"), (ipp_side, "meanx")):
        if side is not None:
            return side, evidence, diag
    return +1, "default", diag


def _position_from(ds, normal):
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
        a = a.max() - a                       # MONOCHROME1 is inverted
    return a


def read_heads(sdir: Path):
    """Headers only, once — the sort key used to re-read each file."""
    heads = []
    for f in sdir.glob("*.dcm"):
        try:
            heads.append((f, pydicom.dcmread(f, stop_before_pixels=True)))
        except Exception:
            pass
    return heads


def load_volume(heads, side):
    """-> ((D_SLICES, SIZE, SIZE) uint8, stats dict). Zeros if unreadable.

    `side` is supplied by the caller because it is a property of the study, not of
    this series — see `study_side`.
    """
    zero = np.zeros((D_SLICES, SIZE, SIZE), np.uint8)
    if not heads:
        return zero, {}

    row, normal = _geom(heads[0][1])

    heads = sorted(heads, key=lambda fh: _position_from(fh[1], normal))
    files = [f for f, _ in heads]

    # Which axis carries medial/lateral, and does it currently point medial?
    flip_w = flip_d = False
    if row is not None and abs(row[0]) > AXIS_TOL:
        flip_w = (np.sign(row[0]) != side)          # axial / coronal
    if normal is not None and abs(normal[0]) > AXIS_TOL:
        flip_d = (np.sign(normal[0]) != side)       # sagittal

    idx = np.linspace(0, len(files) - 1, D_SLICES).round().astype(int)
    if flip_d:
        idx = idx[::-1]

    out = np.zeros((D_SLICES, SIZE, SIZE), np.float32)
    for j, i in enumerate(idx):               # decode only what we keep
        try:
            a = _pixels(pydicom.dcmread(files[i]))
            out[j] = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        except Exception:
            pass                              # leave slice zeroed
    if flip_w:
        out = out[:, :, ::-1]

    # Native acquired in-plane size, straight off the header we already read. If this
    # is below SIZE the resize is an upsample and 336 is buying interpolation, not
    # detail — see the header. Free to collect, decisive to know.
    h0 = heads[0][1]
    stats = {"flip_w": int(flip_w), "flip_d": int(flip_d),
             "native_rows": int(getattr(h0, "Rows", 0) or 0),
             "native_cols": int(getattr(h0, "Columns", 0) or 0)}

    # Percentile window per volume — MRI intensity has no fixed scale, so a
    # global window would blow out some sequences and flatten others.
    lo, hi = np.percentile(out, [1, 99])
    if hi <= lo:
        return zero, stats
    return (np.clip((out - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), stats


def build_study(sid: str) -> tuple:
    vol = np.zeros((3, D_SLICES, SIZE, SIZE), np.uint8)
    found = plan.get(sid, {})

    # Read every header first, decide the side once, then decode. Deciding per
    # series is what produced 413 self-contradicting studies in version 1.
    heads_by_plane = {}
    for plane in PLANES:
        suid = found.get(plane)
        if suid:
            heads_by_plane[plane] = read_heads(COMP / "train_series" / sid / suid)

    side, evidence, diag = study_side(list(heads_by_plane.values()),
                                      reports.get(sid, ""))

    stats = []
    for p, plane in enumerate(PLANES):
        heads = heads_by_plane.get(plane)
        if heads:
            vol[p], st = load_volume(heads, side)
            if st:
                stats.append(dict(st, plane=plane))

    info = dict(diag, StudyInstanceUID=sid, evidence=evidence,
                side="R" if side > 0 else "L",
                flip_w=sum(s["flip_w"] for s in stats),
                flip_d=sum(s["flip_d"] for s in stats))
    return sid, vol, len(found), stats, info

# %% [markdown]
# ## Decode
#
# Unchanged mechanics: `open_memmap` streams to disk because a 10.6 GB array will not
# fit in a Kaggle CPU worker's RAM, and `imap` (not `imap_unordered`) keeps result k
# paired with `sids[k]` — a misalignment here would attach every volume to the wrong
# labels, so the assert stays load-bearing.

# %%
all_sids = train.StudyInstanceUID.tolist()
N_ALL = len(all_sids)

# Contiguous row range, so index.csv's global ids are a simple offset and the three
# shards concatenate back to the original row order with no gaps or overlap.
bounds = np.linspace(0, N_ALL, N_SHARDS + 1).round().astype(int)
ROW_LO, ROW_HI = int(bounds[SHARD]), int(bounds[SHARD + 1])
sids = all_sids[ROW_LO:ROW_HI]
N = len(sids)
print(f"shard {SHARD}: global rows [{ROW_LO}, {ROW_HI}) — {N} of {N_ALL} studies")
print(f"projected volumes.npy: {3 * D_SLICES * SIZE * SIZE * N / 1e9:.2f} GB")

# The whole reason for sharding. If this trips, the split is miscomputed and the
# kernel would die at 20 GB after hours of decoding with nothing written.
assert 3 * D_SLICES * SIZE * SIZE * N / 1e9 < 19.0, \
    "shard exceeds the /kaggle/working budget — raise N_SHARDS"

arr = np.lib.format.open_memmap(
    OUT / "volumes.npy", mode="w+", dtype=np.uint8, shape=(N, 3, D_SLICES, SIZE, SIZE))

rows, geom_rows, side_rows, t0 = [], [], [], time.time()
with Pool(os.cpu_count()) as pool:
    for k, (sid, vol, n_planes, stats, info) in enumerate(
            pool.imap(build_study, sids, chunksize=4)):
        assert sid == sids[k], f"order broken at {k}"
        arr[k] = vol
        # "row" is the GLOBAL row so the trainer can address the three shards as one
        # table; "local_row" is the offset inside this shard's volumes.npy.
        rows.append({"row": ROW_LO + k, "local_row": k, "shard": SHARD,
                     "StudyInstanceUID": sid, "n_planes": n_planes,
                     "nonzero": int((vol > 0).any())})
        for st in stats:
            geom_rows.append(dict(st, StudyInstanceUID=sid))
        side_rows.append(info)
        if (k + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{N}  {(k+1)/el:.1f} study/s  "
                  f"eta {(N-k-1)/((k+1)/el)/60:.1f} min", flush=True)

arr.flush()
print(f"\ndecoded {N} studies in {(time.time()-t0)/60:.1f} min")

# %% [markdown]
# ## What the canonicalisation actually did
#
# These counts are the point of the run. If `flip_w` and `flip_d` are both ~0 the
# geometry test silently failed and this cache is identical to the old one; if the
# side evidence is mostly `default` then neither the tag nor the position fallback
# resolved, and the flips are noise. Either case must be caught here, not after an
# 8 h training run.

# %%
geom = pd.DataFrame(geom_rows)
sides = pd.DataFrame(side_rows)
index = pd.DataFrame(rows).sort_values("row")
index.to_csv(OUT / "index.csv", index=False)
sides.to_csv(OUT / "sides.csv", index=False)

print("side evidence:", dict(Counter(sides.evidence)))
print("knee side:    ", dict(Counter(sides.side)))
if len(geom):
    geom.to_csv(OUT / "geometry.csv", index=False)
    print("\nflips applied per plane (series level):")
    print(geom.groupby("plane")[["flip_w", "flip_d"]].agg(["sum", "mean"]).round(3))

# %% [markdown]
# ## Native in-plane resolution — is 336 detail or interpolation?
#
# The one question this run exists to answer. `native_min` is the shorter acquired
# edge, which is what the `INTER_AREA` resize to a square 336 is actually working
# from. The fraction at or above 336 is the fraction of the corpus where this cache
# carries real acquired detail that the 224 cache threw away; the rest is upsampled
# and costs 2.25x the bytes for nothing.

# %%
native = pd.DataFrame()
if len(geom) and geom.native_rows.gt(0).any():
    native = geom[geom.native_rows > 0].copy()
    native["native_min"] = native[["native_rows", "native_cols"]].min(axis=1)
    native.to_csv(OUT / "native_size.csv", index=False)

    print("native shorter edge, by plane:")
    print(native.groupby("plane").native_min
          .describe(percentiles=[.1, .25, .5, .75, .9]).round(1))
    print("\nmost common native sizes:")
    print(native.native_min.value_counts().head(8))
    frac_336 = float((native.native_min >= SIZE).mean())
    frac_224 = float((native.native_min >= 224).mean())
    print(f"\nseries with native >= 224: {frac_224:.1%}")
    print(f"series with native >= {SIZE}: {frac_336:.1%}  <-- 336 gains detail only here")
    print(f"median native shorter edge: {native.native_min.median():.0f} px")
else:
    frac_336 = frac_224 = float("nan")
    print("no Rows/Columns tags recovered — cannot judge native resolution")

# %% [markdown]
# ## Does the fallback agree with the tag?
#
# Half the studies carry no laterality tag, so the mean-x rule decides them alone.
# The tagged half is the only place we can check it — scoring mean-x against the tag
# there is the closest thing to ground truth available, and if it disagrees, half
# this cache is canonicalised backwards with nothing downstream to say so.

# %%
lab = sides[sides.tag_side.notna()].copy()
print(f"tag-labelled studies (treated as truth): {len(lab)}")

print("\nper-signal, scored against the tag:")
for col, name in (("desc_side", "SeriesDescription"), ("rep_side", "report"),
                  ("ipp_side", "mean-x @ SIDE_THR")):
    spoke = lab[lab[col].notna()]
    a = float((spoke[col] == spoke.tag_side).mean()) if len(spoke) else float("nan")
    print(f"  {name:22s} coverage {len(spoke)/len(lab):6.1%}   accuracy {a:.4f}")

# What the cascade actually produces on the labelled half, with the tag removed so
# it is scored the way it will be used on the untagged half.
fallback = lab.desc_side.fillna(lab.rep_side).fillna(lab.ipp_side)
agree = float((fallback == lab.tag_side).mean())
print(f"\ncascade WITHOUT the tag, vs the tag: {agree:.4f}")

# Doing nothing is not "no error" — it means left and right knees are mirror
# images of each other, so the majority side is the best consistency achievable.
sc = Counter(sides.side)
status_quo = max(sc.values()) / sum(sc.values())
print(f"status quo (no canonicalisation):   {status_quo:.4f}")
imp = (1 - status_quo) / max(1 - agree, 1e-9)
print(f"improvement: {imp:.1f}x fewer mirrored studies")

both = lab

empty = int((index.nonzero == 0).sum())
meta = {
    "n_studies": N, "shape": [3, D_SLICES, SIZE, SIZE], "dtype": "uint8",
    "shard": SHARD, "n_shards": N_SHARDS, "row_lo": ROW_LO, "row_hi": ROW_HI,
    "n_studies_all": N_ALL,
    "native_min_median": (float(native.native_min.median()) if len(native)
                          else None),
    "frac_native_ge_size": (None if native.empty else round(frac_336, 4)),
    "frac_native_ge_224": (None if native.empty else round(frac_224, 4)),
    "planes": PLANES, "d_slices": D_SLICES, "size": SIZE,
    "laterality_canonical": True,
    "empty_studies": empty,
    "plane_counts": index.n_planes.value_counts().sort_index().to_dict(),
    "side_evidence": dict(Counter(sides.evidence)),
    "side_counts": dict(Counter(sides.side)),
    "cascade_agreement": round(agree, 4),
    "status_quo_consistency": round(status_quo, 4),
    "side_thr": SIDE_THR,
    "n_agreement_testable": int(len(both)),
    "flip_w": int(geom.flip_w.sum()) if len(geom) else 0,
    "flip_d": int(geom.flip_d.sum()) if len(geom) else 0,
    "gb": round((OUT / "volumes.npy").stat().st_size / 1e9, 2),
}
(OUT / "cache_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))

# A cache that is silently mostly zeros would train to garbage and look fine.
assert empty < N * 0.02, f"{empty}/{N} studies decoded empty — investigate"

# The whole reason this notebook exists. Left and right knees both occur in any real
# MRI cohort, so a canonicalisation that fired on almost nothing did not work.
assert len(geom), "no geometry recorded — every series failed to read a header"
assert meta["side_evidence"].get("default", 0) < 0.02 * len(sides), \
    "side unresolved for >2% of studies — tag and mean-x both failed"
assert meta["flip_w"] + meta["flip_d"] > 0.1 * len(geom), \
    "canonicalisation barely fired — the axis test is not matching the geometry"

# The check that version 1 of this notebook failed. The mean-x fallback decides
# roughly half the studies on its own, so if it cannot reproduce the tag on the
# half where both exist, it is not fit to decide the other half either.
assert len(both) > 0.2 * len(sides), \
    "too few tagged studies to validate the cascade against"
# 0.95, not 0.98. The bar is not perfection, it is being decisively better than
# doing nothing: no canonicalisation leaves ~44% of studies mirrored relative to
# the rest, so a residual of a few percent is ordinary label noise by comparison.
# v2 scored 0.901 here and was correctly rejected.
assert agree > 0.95, (
    f"the tag-free cascade reproduces the Laterality tag only {agree:.1%} of the "
    "time, and it decides every untagged study — roughly half the data would be "
    "mirrored. Do not train on it.")
# Shard bookkeeping. The three shards are concatenated by global row downstream, so
# an off-by-one here silently pairs volumes with the wrong labels — the same class of
# bug as the imap ordering assert above, and just as invisible afterwards.
assert len(index) == N == ROW_HI - ROW_LO, "shard row count does not match its range"
assert index.row.tolist() == list(range(ROW_LO, ROW_HI)), \
    "global row ids are not the contiguous range this shard claims"
assert index.local_row.tolist() == list(range(N)), "local rows are not 0..N-1"
assert (index.StudyInstanceUID.tolist()
        == train.StudyInstanceUID.tolist()[ROW_LO:ROW_HI]), \
    "shard study order does not match train.csv order at these global rows"
assert meta["gb"] < 19.0, f"{meta['gb']} GB written — over the working-dir budget"

print("\ncache OK")
