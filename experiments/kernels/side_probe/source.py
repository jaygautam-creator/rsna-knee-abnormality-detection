# %% [markdown]
# # Which knee is it? — header probe
#
# `cache_volumes_lat` v2 failed its own gate: the mean-x fallback reproduces the
# `Laterality` tag only **90.1%** of the time, and it decides ~50% of studies alone.
# The distributions overlap (right knees span x = -252..-34, left knees -200..+124),
# so no threshold on x can separate them — mean-x is finished as a primary signal.
#
# The reports get **97.3%** where they state a side, but they only state one for 37%
# of studies. Combined tag > report > mean-x reaches 92.7%. Still not enough.
#
# So before spending GPU on an image-based side classifier, check whether the answer
# is simply sitting in a header we never read. Knee protocols very often name the
# side (`R KNEE`, `Knie links`, `RODILLA DCHA`), and `PatientPosition` would explain
# a sign inversion if some studies are acquired feet-first.
#
# One header per selected series — ~13k reads, a few minutes, CPU only.
# Output: `side_probe.csv`, scored offline against the tag.

# %%
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pydicom

warnings.filterwarnings("ignore")

PLANES = ["Axial", "Coronal", "Sagittal"]
OUT = Path("/kaggle/working")

FIELDS = [
    "PatientPosition", "SeriesDescription", "ProtocolName", "StudyDescription",
    "BodyPartExamined", "Laterality", "ImageLaterality", "SeriesNumber",
    "Manufacturer", "InstitutionName", "AcquisitionDeviceProcessingDescription",
    "RequestedProcedureDescription", "PerformedProcedureStepDescription",
]


def find_mount(max_depth: int = 3) -> Path:
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
series = series[series.Anatomical_Plane.isin(PLANES)]
print(f"studies {len(train)}  series {len(series)}  cores {os.cpu_count()}")


def probe(row):
    sid, suid, plane = row
    d = COMP / "train_series" / sid / suid
    rec = {"StudyInstanceUID": sid, "SeriesInstanceUID": suid, "plane": plane}
    try:
        f = next(iter(d.glob("*.dcm")))
        ds = pydicom.dcmread(f, stop_before_pixels=True)
    except Exception:
        return rec
    for k in FIELDS:
        v = getattr(ds, k, None)
        if v is not None:
            rec[k] = str(v).strip()
    return rec


rows = list(series[["StudyInstanceUID", "SeriesInstanceUID",
                    "Anatomical_Plane"]].itertuples(index=False, name=None))
with Pool(os.cpu_count()) as pool:
    out = pool.map(probe, rows, chunksize=16)

df = pd.DataFrame(out)
df.to_csv(OUT / "side_probe.csv", index=False)
print("wrote", df.shape)

# %% [markdown]
# ## What is actually populated
#
# A field present on 2% of series cannot rescue 50% of studies, so print the fill
# rate before reading anything into the values.

# %%
print("fill rate by field:")
for k in FIELDS:
    if k in df.columns:
        n = df[k].notna().sum()
        print(f"  {k:45s} {n:6d}  {n/len(df):6.1%}")
    else:
        print(f"  {k:45s} {0:6d}   0.0%")

print("\nPatientPosition values:")
if "PatientPosition" in df.columns:
    print(df.PatientPosition.value_counts(dropna=False).head(10).to_string())

for k in ("SeriesDescription", "ProtocolName", "StudyDescription"):
    if k in df.columns:
        print(f"\n{k} — 15 most common:")
        print(df[k].value_counts().head(15).to_string())
