# %% [markdown]
# # Decode the train set once — compact volume cache
#
# Every training run currently re-decodes ~820k DICOMs. At the 16.5 ms/slice
# measured by the baseline notebook that is ~1 h of pure `pydicom` per run, spent
# on the resource we have least of (30 GPU-h/week, currently exhausted).
#
# This notebook is **CPU-only**, so it costs no GPU quota. It decodes once and
# writes a fixed-shape `uint8` array that later runs mmap in milliseconds.
#
# Two decisions do most of the work:
#
# * **Header-first slice selection.** We read every file with
#   `stop_before_pixels=True` to sort the series correctly, then decode *only*
#   the `D_SLICES` we keep. That is ~48 pixel decodes per study instead of ~186 —
#   the reason this finishes in well under the 12 h CPU limit.
# * **One series per anatomical plane**, preferring fluid-sensitive. Effusion,
#   Synovitis and Contusion are the conditions our report labeller is worst at,
#   and they are exactly the ones that live on fluid-sensitive sequences, so we
#   should not let a T1 win the pick by slice count alone.
#
# Output: `volumes.npy` (N, 3, D, H, W) uint8 + `index.csv` + `cache_meta.json`.

# %%
import json
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings("ignore")

D_SLICES = 16          # slices kept per plane
SIZE = 224             # in-plane resolution
PLANES = ["Axial", "Coronal", "Sagittal"]
OUT = Path("/kaggle/working")
SEED = 0

# (3, 16, 224, 224) uint8 = 2.4 MB/study -> ~10.6 GB for 4,407, under the 20 GB cap.
print(f"per study: {3 * D_SLICES * SIZE * SIZE / 1e6:.1f} MB")


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
# ## Series selection
#
# One series per plane per study. Fluid-sensitive wins; slice count breaks ties.
# Studies missing a plane are kept with that plane left as zeros rather than
# dropped — losing a study entirely costs a label, and the labels are the
# expensive thing here.

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

print(sel.groupby("Anatomical_Plane").agg(n_studies=("StudyInstanceUID", "nunique"),
                                          mean_slices=("n", "mean")).round(1))
have = sel.groupby("StudyInstanceUID").Anatomical_Plane.nunique()
print(f"\nstudies with 3/2/1 planes: "
      f"{(have == 3).sum()}/{(have == 2).sum()}/{(have == 1).sum()}")
print(f"studies with no usable series: {len(train) - len(have)}")

plan = {sid: g.set_index("Anatomical_Plane").SeriesInstanceUID.to_dict()
        for sid, g in sel.groupby("StudyInstanceUID")}

# %% [markdown]
# ## Decode
#
# `_position` sorts by patient coordinate rather than filename — DICOM filenames
# here are UIDs and carry no order, so sorting them lexically would silently
# scramble every volume. `InstanceNumber` is the fallback.

# %%
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
        a = a.max() - a                       # MONOCHROME1 is inverted
    return a


def load_volume(sdir: Path) -> np.ndarray:
    """-> (D_SLICES, SIZE, SIZE) uint8, or zeros if unreadable."""
    files = list(sdir.glob("*.dcm"))
    if not files:
        return np.zeros((D_SLICES, SIZE, SIZE), np.uint8)

    files.sort(key=_position)                 # headers only: cheap
    idx = np.linspace(0, len(files) - 1, D_SLICES).round().astype(int)

    out = np.zeros((D_SLICES, SIZE, SIZE), np.float32)
    for j, i in enumerate(idx):               # decode only what we keep
        try:
            a = _pixels(pydicom.dcmread(files[i]))
            out[j] = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        except Exception:
            pass                              # leave slice zeroed

    # Percentile window per volume — MRI intensity has no fixed scale, so a
    # global window would blow out some sequences and flatten others.
    lo, hi = np.percentile(out, [1, 99])
    if hi <= lo:
        return np.zeros((D_SLICES, SIZE, SIZE), np.uint8)
    return (np.clip((out - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def build_study(sid: str) -> tuple:
    vol = np.zeros((3, D_SLICES, SIZE, SIZE), np.uint8)
    found = plan.get(sid, {})
    for p, plane in enumerate(PLANES):
        suid = found.get(plane)
        if suid:
            vol[p] = load_volume(COMP / "train_series" / sid / suid)
    return sid, vol, len(found)

# %%
sids = train.StudyInstanceUID.tolist()
N = len(sids)

# open_memmap streams to disk — a 10.6 GB array will not fit in the 16 GB of RAM
# a Kaggle CPU worker has once pydicom and the pool are resident.
arr = np.lib.format.open_memmap(
    OUT / "volumes.npy", mode="w+", dtype=np.uint8, shape=(N, 3, D_SLICES, SIZE, SIZE))

rows, t0 = [], time.time()
with Pool(os.cpu_count()) as pool:
    # imap (not imap_unordered) preserves input order, so result k is sids[k];
    # the assert makes that guarantee load-bearing instead of assumed, because a
    # misalignment here would pair every volume with the wrong labels.
    for k, (sid, vol, n_planes) in enumerate(pool.imap(build_study, sids, chunksize=4)):
        assert sid == sids[k], f"order broken at {k}"
        arr[k] = vol
        rows.append({"row": k, "StudyInstanceUID": sid, "n_planes": n_planes,
                     "nonzero": int((vol > 0).any())})
        if (k + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{N}  {(k+1)/el:.1f} study/s  "
                  f"eta {(N-k-1)/((k+1)/el)/60:.1f} min", flush=True)

arr.flush()
print(f"\ndecoded {N} studies in {(time.time()-t0)/60:.1f} min")

# %%
index = pd.DataFrame(rows).sort_values("row")
index.to_csv(OUT / "index.csv", index=False)

empty = int((index.nonzero == 0).sum())
meta = {
    "n_studies": N, "shape": [3, D_SLICES, SIZE, SIZE], "dtype": "uint8",
    "planes": PLANES, "d_slices": D_SLICES, "size": SIZE,
    "empty_studies": empty,
    "plane_counts": index.n_planes.value_counts().sort_index().to_dict(),
    "gb": round((OUT / "volumes.npy").stat().st_size / 1e9, 2),
}
(OUT / "cache_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))

# A cache that is silently mostly zeros would train to garbage and look fine.
assert empty < N * 0.02, f"{empty}/{N} studies decoded empty — investigate"
print("\ncache OK")
