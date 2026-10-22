# %% [markdown]
# # RSNA Knee — DICOM probe
#
# Phase 2's main engineering risk is not the model, it is throughput. There are
# ~820k DICOMs across 24,371 series in a mix of transfer syntaxes (uncompressed
# Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR LE). JPEG 2000 in
# particular decodes slowly in pure Python, and decoding on the fly every epoch
# would consume the whole GPU budget.
#
# So before writing any training code, measure:
#
# 1. which transfer syntaxes actually appear, and in what proportion
# 2. decode throughput **per syntax** — the slow one sets the schedule
# 3. image geometry (rows/cols/spacing) and slices per series
# 4. which of the 86 allowlisted tags are usable for ordering and cropping
# 5. the projected wall-clock to cache the whole training set once
#
# CPU-only and deliberately cheap: it samples series rather than reading all
# 820k files.

# %%
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

N_SERIES = 60          # sampled series
SEED = 0
OUT = "/kaggle/working"


def find_comp(max_depth: int = 3) -> Path:
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
    raise FileNotFoundError("competition data not found")


COMP = find_comp()
series = pd.read_csv(f"{COMP}/train_series.csv")
print("mount:", COMP, "| series:", len(series))

# %%
random.seed(SEED)
sample = series.sample(N_SERIES, random_state=SEED)

rows = []
for r in sample.itertuples():
    sdir = COMP / "train_series" / r.StudyInstanceUID / r.SeriesInstanceUID
    if not sdir.is_dir():
        continue
    files = sorted(sdir.glob("*.dcm"))
    if not files:
        continue

    # Header-only read is what an ordering/QA pass would do.
    t0 = time.time()
    hdr = pydicom.dcmread(files[0], stop_before_pixels=True)
    t_hdr = time.time() - t0

    # Full decode of a single slice: the operation that dominates caching.
    t0 = time.time()
    ds = pydicom.dcmread(files[0])
    arr = ds.pixel_array
    t_pix = time.time() - t0

    rows.append({
        "series": r.SeriesInstanceUID,
        "plane": r.Anatomical_Plane,
        "fluid": r.Fluid_Sensitive,
        "n_slices": len(files),
        "syntax": str(hdr.file_meta.TransferSyntaxUID),
        "syntax_name": str(hdr.file_meta.TransferSyntaxUID.name),
        "rows": int(getattr(hdr, "Rows", 0)),
        "cols": int(getattr(hdr, "Columns", 0)),
        "dtype": str(arr.dtype),
        "photometric": str(getattr(hdr, "PhotometricInterpretation", "")),
        "px_spacing": str(getattr(hdr, "PixelSpacing", "")),
        "thickness": float(getattr(hdr, "SliceThickness", 0) or 0),
        "t_hdr_ms": round(t_hdr * 1000, 2),
        "t_pix_ms": round(t_pix * 1000, 2),
        "mb": round(files[0].stat().st_size / 1e6, 2),
    })

df = pd.DataFrame(rows)
print(f"probed {len(df)} series\n")

# %% [markdown]
# ## Transfer syntax is the schedule driver

# %%
syn = (df.groupby("syntax_name")
         .agg(n_series=("series", "size"),
              median_decode_ms=("t_pix_ms", "median"),
              median_hdr_ms=("t_hdr_ms", "median"),
              median_slices=("n_slices", "median"))
         .sort_values("n_series", ascending=False))
print(syn.to_string())

print("\ngeometry:")
print(df[["rows", "cols", "n_slices", "thickness"]].describe().T[
    ["mean", "50%", "min", "max"]].round(2).to_string())
print("\nphotometric:", Counter(df.photometric))
print("dtypes:", Counter(df.dtype))
print("planes:", Counter(df.plane))

# %% [markdown]
# ## Projected cost of caching the training set once

# %%
total_slices = df.n_slices.sum()
total_decode_s = (df.t_pix_ms * df.n_slices).sum() / 1000
per_series_s = total_decode_s / len(df)

all_series = len(series)
est_hours = per_series_s * all_series / 3600
print(f"sampled: {len(df)} series, {total_slices} slices, "
      f"{total_decode_s:.1f}s decode")
print(f"mean {per_series_s:.2f} s/series (single core)")
print(f"\nALL {all_series} series, 1 core : {est_hours:.1f} h")
for w in (2, 4, 8):
    print(f"ALL {all_series} series, {w} cores: {est_hours / w:.1f} h "
          f"({'FITS' if est_hours / w < 9 else 'TOO SLOW'} in one 9 h session)")

# Worst-case syntax sets the tail.
slow = syn.median_decode_ms.idxmax()
print(f"\nslowest syntax: {slow} at {syn.median_decode_ms.max():.1f} ms/slice")

df.to_csv(f"{OUT}/dicom_probe.csv", index=False)
with open(f"{OUT}/dicom_probe_summary.json", "w") as f:
    json.dump({
        "n_probed": len(df),
        "per_series_s": float(per_series_s),
        "est_hours_1core": float(est_hours),
        "syntaxes": syn.to_dict("index"),
    }, f, indent=2, default=str)

# %% [markdown]
# ## Wider transfer-syntax survey
#
# The 60-series sample found *only* uncompressed Explicit VR LE, but the data
# description promises a mix including JPEG 2000 — which decodes an order of
# magnitude slower and would change the schedule. 60 series is too small to rule
# out a syntax at a few percent prevalence, so survey headers only (no pixel
# decode) over a much larger sample. This is cheap: one header read per series.

# %%
SURVEY = 2000
t0 = time.time()
syn_counter, plane_syn = Counter(), Counter()
missing = 0
for r in series.sample(min(SURVEY, len(series)), random_state=SEED).itertuples():
    sdir = COMP / "train_series" / r.StudyInstanceUID / r.SeriesInstanceUID
    f = next(sdir.glob("*.dcm"), None) if sdir.is_dir() else None
    if f is None:
        missing += 1
        continue
    name = str(pydicom.dcmread(f, stop_before_pixels=True)
               .file_meta.TransferSyntaxUID.name)
    syn_counter[name] += 1
    plane_syn[(r.Anatomical_Plane, name)] += 1

n = sum(syn_counter.values())
print(f"surveyed {n} series in {time.time() - t0:.0f}s (missing dirs: {missing})\n")
for name, c in syn_counter.most_common():
    print(f"  {c / n:6.2%}  {c:5d}  {name}")

compressed = {k: v for k, v in syn_counter.items()
              if "JPEG" in k or "RLE" in k or "Deflated" in k}
print(f"\ncompressed share: {sum(compressed.values()) / n:.2%} {compressed}")

# %% [markdown]
# ## Tags available for slice ordering and physical cropping
#
# Ordering slices by `InstanceNumber` alone is unreliable across vendors. The
# correct order is the projection of `ImagePositionPatient` onto the slice
# normal derived from `ImageOrientationPatient`. Confirm those survived the
# 86-tag allowlist before designing the preprocessing around them.

# %%
WANT = ["ImagePositionPatient", "ImageOrientationPatient", "SliceLocation",
        "InstanceNumber", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
        "Rows", "Columns", "PhotometricInterpretation", "RescaleSlope",
        "RescaleIntercept", "WindowCenter", "WindowWidth", "SeriesDescription",
        "ScanningSequence", "SequenceVariant", "ScanOptions", "MRAcquisitionType",
        "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
        "MagneticFieldStrength", "Manufacturer", "BodyPartExamined", "Laterality"]

probe = sample.head(12)
present = Counter()
for r in probe.itertuples():
    sdir = COMP / "train_series" / r.StudyInstanceUID / r.SeriesInstanceUID
    files = sorted(sdir.glob("*.dcm"))
    if not files:
        continue
    hdr = pydicom.dcmread(files[0], stop_before_pixels=True)
    for tag in WANT:
        if tag in hdr:
            present[tag] += 1

n = len(probe)
for tag in WANT:
    mark = "OK " if present[tag] == n else ("SOME" if present[tag] else "MISS")
    print(f"  [{mark}] {tag:28s} {present[tag]}/{n}")
