"""Sanity-check a produced label file before anything trains on it.

`train_labels_v1.csv` is the foundation of the whole pipeline — every downstream
result inherits its errors, and a silent failure here (all-0.5 rows, a collapsed
condition, studies dropped) would look like a modelling problem for weeks.
Check it explicitly.

Run: python -m knee.labels.validate_labels artifacts/train_labels_v1.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.paths import REPO_ROOT as ROOT

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def main(path: str) -> int:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    pred = pd.read_csv(path)
    fail = []

    print(f"=== {Path(path).name} ===")
    print(f"rows {len(pred)}  (train.csv has {len(train)})")
    if len(pred) != len(train):
        fail.append(f"row count {len(pred)} != {len(train)}")

    missing = set(train.StudyInstanceUID) - set(pred.StudyInstanceUID)
    if missing:
        fail.append(f"{len(missing)} studies missing")
    if pred.StudyInstanceUID.duplicated().any():
        fail.append("duplicate StudyInstanceUID rows")
    absent = [c for c in LABELS if c not in pred.columns]
    if absent:
        fail.append(f"missing columns: {absent}")
        print("FAIL:", *fail, sep="\n  ")
        return 1

    p = pred[LABELS]
    if p.isna().any().any():
        fail.append(f"NaNs: {p.isna().sum().sum()}")
    if not ((p >= 0) & (p <= 1)).all().all():
        fail.append("values outside [0, 1]")

    # A condition stuck at one value carries no signal and, being constant,
    # would silently contribute a meaningless 0.5 AUC to the macro average.
    print("\nper-condition distribution:")
    for c in LABELS:
        s = p[c]
        exact_half = float((s == 0.5).mean())
        line = (f"  {c:18s} mean {s.mean():.3f}  sd {s.std():.3f}  "
                f"p10 {s.quantile(.1):.3f}  p90 {s.quantile(.9):.3f}  "
                f">0.5 {(s > 0.5).mean():.3f}  ==0.5 {exact_half:.3f}")
        if s.std() < 0.02:
            line += "   <-- COLLAPSED"
            fail.append(f"{c} has near-zero variance")
        if exact_half > 0.05:
            # 0.5 is the fallback when neither Yes nor No token was scoreable.
            line += "   <-- MANY FALLBACKS"
            fail.append(f"{c} fell back to 0.5 on {exact_half:.1%} of studies")
        print(line)

    # Agreement with the 58 expert labels, recomputed here so the check does not
    # depend on the notebook having printed it.
    gold = train[train[LABELS[0]].notna()]
    merged = gold[["StudyInstanceUID"] + LABELS].merge(
        pred, on="StudyInstanceUID", suffixes=("_gold", "_pred"))
    if len(merged):
        aucs = []
        for c in LABELS:
            g = merged[f"{c}_gold"].astype(int)
            if g.nunique() > 1:
                aucs.append(roc_auc_score(g, merged[f"{c}_pred"]))
        print(f"\ngold check: n={len(merged)}  mean AUC {np.mean(aucs):.4f}"
              f"   (regex on the same studies: 0.6287)")
        if np.mean(aucs) < 0.80:
            fail.append(f"mean AUC {np.mean(aucs):.4f} below 0.80")

    # Report-derived labels are identical for studies sharing a report; that is
    # expected, but it is also what forces grouped CV, so surface it.
    dup = train.assign(h=train.Report.fillna("").str.strip()).h.duplicated(keep=False)
    print(f"studies sharing a report with another: {int(dup.sum())} "
          f"-> use src/labels/folds.py, not random CV")

    print("\n" + ("FAILED:\n  " + "\n  ".join(fail) if fail else "all checks passed"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "artifacts/train_labels_v1.csv"))
