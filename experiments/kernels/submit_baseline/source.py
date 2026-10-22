# %% [markdown]
# # Baseline submission — format check + test-set I/O budget
#
# This scores ~0.50 by construction and is not trying to do better. It exists to
# retire two risks while failure is still cheap:
#
# 1. **Mechanics.** That a notebook with internet disabled reads the *hidden*
#    test set (the local `test.csv` has 3 rows; the real one is swapped in at
#    submission), writes `submission.csv` in the exact expected shape, and is
#    accepted by the grader.
# 2. **The 9 h budget.** The image model's binding constraint is almost certainly
#    DICOM decode throughput, not GPU time. We measure it *on the real test set*
#    and extrapolate, so the model is designed against a measured number instead
#    of a guess.
#
# Ordering matters: the submission is written **before** the probe runs, and the
# probe is wrapped so that nothing it does can invalidate an already-valid file.

# %%
import os
import time
from pathlib import Path

import pandas as pd

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

PROBE_STUDIES = 24      # sampled studies for the I/O measurement
PROBE_BUDGET_S = 420    # hard wall-clock cap on the probe
SEED = 0


def find_mount(max_depth: int = 3) -> Path:
    """Locate the competition mount without hardcoding the slug.

    Breadth-first to depth 3 — the mount is not always a direct child of
    /kaggle/input. This mirrors find_comp() in notebooks/dicom_probe, which
    established the layout; anchoring on test.csv + sample_submission.csv since
    those are what this notebook actually consumes.
    """
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if (d / "test.csv").exists() and (d / "sample_submission.csv").exists():
            return d
        if depth >= max_depth:
            continue
        for child in sorted(p for p in d.iterdir() if p.is_dir()):
            if child.name in ("train_series", "test_series"):
                continue          # huge DICOM trees, never the anchor
            frontier.append((child, depth + 1))

    # Failing blind here costs a whole run, so show the layout we did find.
    print("--- /kaggle/input tree (depth 2) ---")
    for p in sorted(Path("/kaggle/input").rglob("*")):
        rel = p.relative_to("/kaggle/input")
        if len(rel.parts) <= 2:
            print(" ", rel, "/" if p.is_dir() else "")
    raise FileNotFoundError("competition mount not found under /kaggle/input")


COMP = find_mount()
test = pd.read_csv(COMP / "test.csv")
series = pd.read_csv(COMP / "test_series.csv")
print("mount:", COMP)
print(f"test studies: {len(test)}  series: {len(series)}")

# %% [markdown]
# ## 1. Write the submission first
#
# A constant column has no ordering, so every constant scores exactly 0.50 —
# 0.5 is as good as train prevalence here and makes the placeholder obvious in
# the LB history. We take the column list from `sample_submission.csv` rather
# than from our own constant so that a silent upstream rename shows up as a
# crash here instead of as a malformed file.

# %%
sample = pd.read_csv(COMP / "sample_submission.csv")
assert list(sample.columns) == ["StudyInstanceUID"] + LABELS, list(sample.columns)

sub = pd.DataFrame({"StudyInstanceUID": test.StudyInstanceUID})
for c in LABELS:
    sub[c] = 0.5
sub = sub[sample.columns]

sub.to_csv("submission.csv", index=False)

# Re-read from disk: verifies what the grader will actually see, not what we
# think we wrote.
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == list(sample.columns), "column mismatch"
assert len(chk) == len(test), f"row count {len(chk)} != {len(test)}"
assert chk.StudyInstanceUID.is_unique, "duplicate StudyInstanceUID"
assert set(chk.StudyInstanceUID) == set(test.StudyInstanceUID), "UID set mismatch"
assert chk[LABELS].notna().all().all(), "NaN in predictions"
assert chk[LABELS].to_numpy().min() >= 0 and chk[LABELS].to_numpy().max() <= 1

print(f"submission.csv OK — {len(chk)} rows x {len(chk.columns)} cols")
print(chk.head(3).to_string(index=False)[:200])

# %% [markdown]
# ## 2. Measure DICOM throughput on the hidden test set
#
# Everything below is instrumentation. It is wrapped in `try/except` because a
# valid `submission.csv` already exists on disk — a probe failure should cost us
# the measurement, never the submission.

# %%
try:
    import pydicom

    rng = series.sample(frac=1.0, random_state=SEED)
    study_ids = test.StudyInstanceUID.drop_duplicates().head(PROBE_STUDIES)
    probe = rng[rng.StudyInstanceUID.isin(study_ids)]

    t0 = time.time()
    rows, decoded_slices = [], 0
    for r in probe.itertuples():
        if time.time() - t0 > PROBE_BUDGET_S:
            print(f"probe hit its {PROBE_BUDGET_S}s cap — reporting partial")
            break
        sdir = COMP / "test_series" / r.StudyInstanceUID / r.SeriesInstanceUID
        files = sorted(sdir.glob("*.dcm"))
        if not files:
            continue
        s = time.time()
        for f in files:
            ds = pydicom.dcmread(f)
            _ = ds.pixel_array          # force decode, this is the real cost
            decoded_slices += 1
        rows.append({
            "study": r.StudyInstanceUID,
            "slices": len(files),
            "secs": time.time() - s,
            "plane": getattr(r, "Anatomical_Plane", None),
            "fluid": getattr(r, "Fluid_Sensitive", None),
        })

    df = pd.DataFrame(rows)
    elapsed = time.time() - t0

    if df.empty:
        print("no series decoded — cannot estimate budget")
    else:
        n_studies_probed = df.study.nunique()
        s_per_study = df.secs.sum() / n_studies_probed
        s_per_slice = df.secs.sum() / max(decoded_slices, 1)
        cores = os.cpu_count() or 4

        print(f"\ndecoded {len(df)} series / {decoded_slices} slices "
              f"across {n_studies_probed} studies in {elapsed:.0f}s")
        print(f"  {s_per_slice*1000:.1f} ms/slice, {s_per_study:.2f} s/study "
              f"(single core, all series)")
        print(f"  cores available: {cores}")

        # The hidden test set is larger than the 3-row local stub. Project the
        # full-decode cost across plausible sizes so the number is usable now.
        print(f"\n{'test studies':>14}{'1 core':>10}{f'{cores} cores':>12}")
        for n in (1000, 2500, 5000):
            h1 = s_per_study * n / 3600
            print(f"{n:>14}{h1:>9.1f}h{h1/cores:>11.1f}h")
        print(f"\nbudget is 9.0h total (decode + GPU + overhead).")
        print("if decoding ALL series blows the budget, the image model must "
              "select series (test_series.csv has Fluid_Sensitive / "
              "Fat_Suppression / Anatomical_Plane) rather than read everything.")

        if not df.plane.isna().all():
            print("\nper-plane cost:")
            print(df.groupby("plane", dropna=False)
                    .agg(n_series=("secs", "size"),
                         mean_s=("secs", "mean"),
                         mean_slices=("slices", "mean"))
                    .round(2).to_string())

except Exception as e:                      # noqa: BLE001 - probe must never fail the run
    print(f"probe failed ({type(e).__name__}: {e}) — submission.csv is unaffected")

# %%
print("\nfinal check:", pd.read_csv("submission.csv").shape)
