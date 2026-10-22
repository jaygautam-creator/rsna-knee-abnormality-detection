"""Merge the v4 relabel into the v3 label set.

The relabel kernel emits one raw margin per (condition, new variant). This
script combines them with v3's existing `v0` margins — same Qwen3-8B teacher, so
they are directly comparable (verified at corr 1.0000 on the 58) — and leaves
the other eight conditions, plus PF OA, untouched.

    Synovitis  = mean(z(v0), z(v2))
    Contusion  = mean(z(v0), z(v2), z(v4))
    Effusion   = mean(z(v0), z(v4))
    PF OA      = v0 unchanged — no rewording beat it above coin-flip confidence

Combining lives here rather than in the kernel for two reasons: v0 is local, and
z-scoring has to span the whole corpus. Two shards standardised against their
own means would splice into one inconsistent column.

Two more things have to be right for the trainer to accept the result:

1. **Scale.** The trainer does BCE against soft targets. Averaged z-scores sit
   on an arbitrary scale, so if the relabelled columns arrived that way their
   effective loss weight would change for reasons unrelated to label quality.
   The new ranking is therefore quantile-mapped onto the v3 margin distribution
   of the same condition: ordering is entirely the new one, target values are
   distributed exactly as before.

2. **Completeness.** A partial label set that silently trains is worse than a
   crash, so a missing study is fatal rather than filled.

Usage:
    python -m knee.labels.merge_v4            # writes artifacts/train_labels_v4.csv
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.paths import REPO_ROOT as ROOT

# Chosen in the sweep by bootstrap confidence, not by argmax — see
# notebooks/label_relabel_v4/source.py for the table and the reasoning.
COMBINE = {
    "Synovitis": ["v0", "v2"],
    "Contusion": ["v0", "v2", "v4"],
    "Effusion": ["v0", "v4"],
    "PF OA": ["v0"],
}
WEAK = list(COMBINE)
V3_GOLD_AUC = {"Synovitis": 0.7467, "Contusion": 0.8097, "Effusion": 0.8106,
               "PF OA": 0.8533}
V3_ALL12_MEAN = 0.8740


def load_shards(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        raise FileNotFoundError(f"no relabel shards matched {pattern}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    dup = int(df.StudyInstanceUID.duplicated().sum())
    if dup:
        raise ValueError(f"{dup} studies appear in more than one shard")
    print(f"loaded {len(paths)} shard(s), {len(df)} studies")
    return df


def zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


def quantile_map(new_margin: np.ndarray, ref_margin: np.ndarray) -> np.ndarray:
    """Reorder `ref_margin`'s values to follow `new_margin`'s ranking.

    Rank-preserving by construction, so the AUC of the output equals the AUC of
    `new_margin`; the value distribution is exactly `ref_margin`'s.
    """
    order = np.argsort(np.argsort(new_margin))
    return np.sort(ref_margin)[order]


def main() -> None:
    v3_prob = pd.read_csv(ROOT / "artifacts/train_labels_v3.csv")
    v3_mar = pd.read_csv(ROOT / "artifacts/train_margins_v3.csv")
    new = load_shards("artifacts/relabel_v4/relabel_v4_raw_shard*.csv")

    missing = set(v3_mar.StudyInstanceUID) - set(new.StudyInstanceUID)
    if missing:
        raise ValueError(f"{len(missing)} studies missing from the relabel — "
                         "shards are incomplete, refusing to write a partial set")
    new = new.set_index("StudyInstanceUID").loc[v3_mar.StudyInstanceUID]

    out_mar = v3_mar.copy()
    for cond, variants in COMBINE.items():
        if variants == ["v0"]:
            print(f"{cond:18s} unchanged (v0)")
            continue
        parts = []
        for v in variants:
            col = v3_mar[cond].to_numpy(np.float64) if v == "v0" \
                else new[f"{cond}|{v}"].to_numpy(np.float64)
            parts.append(zscore(col))
        combined = np.mean(parts, axis=0)
        out_mar[cond] = quantile_map(combined, v3_mar[cond].to_numpy(np.float64))
        print(f"{cond:18s} {'+'.join(variants)}")

    labels = [c for c in v3_prob.columns if c != "StudyInstanceUID"]
    out_prob = out_mar.copy()
    m = out_prob[labels].to_numpy(np.float64)
    out_prob[labels] = 1.0 / (1.0 + np.exp(-m))

    # Gate on the 58: the only reason to write this file is that it beats v3.
    gold = pd.read_csv(ROOT / "data/train.csv").dropna(subset=["ACL"])
    ev = gold[["StudyInstanceUID"] + labels].merge(
        out_prob[["StudyInstanceUID"] + labels], on="StudyInstanceUID",
        suffixes=("_g", "_p"))
    rows = []
    for cond in WEAK:
        y = ev[f"{cond}_g"].astype(int).to_numpy()
        auc = float(roc_auc_score(y, ev[f"{cond}_p"].to_numpy()))
        rows.append({"condition": cond, "v3": V3_GOLD_AUC[cond],
                     "v4": round(auc, 4),
                     "delta": round(auc - V3_GOLD_AUC[cond], 4)})
    rep = pd.DataFrame(rows)
    print()
    print(rep.to_string(index=False))
    mean_v3 = float(np.mean(list(V3_GOLD_AUC.values())))
    mean_v4 = float(rep.v4.mean())
    print(f"\nweak-four mean vs gold: {mean_v3:.4f} -> {mean_v4:.4f} "
          f"({mean_v4 - mean_v3:+.4f})")

    full = float(np.mean([roc_auc_score(ev[f"{c}_g"].astype(int), ev[f"{c}_p"])
                          for c in labels]))
    print(f"all-twelve mean vs gold: {V3_ALL12_MEAN:.4f} -> {full:.4f} "
          f"({full - V3_ALL12_MEAN:+.4f})")

    out_prob.to_csv(ROOT / "artifacts/train_labels_v4.csv", index=False)
    out_mar.to_csv(ROOT / "artifacts/train_margins_v4.csv", index=False)
    json.dump({"combine": COMBINE, "weak_mean_v3": mean_v3,
               "weak_mean_v4": mean_v4, "all12_mean_v3": V3_ALL12_MEAN,
               "all12_mean_v4": full, "per_condition": rows},
              open(ROOT / "artifacts/labels_v4_meta.json", "w"), indent=2)
    print("\nwrote artifacts/train_labels_v4.csv")
    if mean_v4 <= mean_v3:
        print("WARNING: v4 does not beat v3 on the 58 — do not spend 8 h "
              "retraining on this without a reason to trust it over the gate.")


if __name__ == "__main__":
    main()
