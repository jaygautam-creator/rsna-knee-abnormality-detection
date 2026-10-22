"""Score every submission variant on the 58 expert-labelled studies, from one evaluation run's component dumps.

    python scripts/evaluate_gold58.py runs/gold58 data/train.csv

``runs/gold58`` is the output of pipeline/eval/gold58-eval.ipynb (58 gold + 46 other training studies scored end to
end, so ranks are taken over 104 rows as in a real run). It must contain raptor_raw.csv, raptor224_raw.csv,
ours_sub.csv, _own.csv and s6_raw.csv. The variants are rebuilt from pipeline/variants/*.json with knee.blend.

Prints per-finding AUCs, an S6-weight sweep, and a paired bootstrap against v90. These numbers are in-sample for
any weight that was chosen on the same 58 studies; see scripts/split_half_cv.py for held-out estimates.
"""

import sys

import numpy as np
import pandas as pd

from knee.blend import full_chain
from knee.constants import LABELS
from knee.io import load_components, load_gold
from knee.metrics import macro_auc, paired_bootstrap, per_finding_auc
from knee.paths import PIPELINE_DIR
from knee.variants import Variant


def main(run_dir: str, train_csv: str) -> None:
    comp = load_components(run_dir)
    gold = load_gold(train_csv)
    ids = [u for u in gold.index if u in comp["raptor384"].index]
    y = gold.loc[ids].to_numpy()
    print(f"gold studies scored: {len(ids)}/58, rows ranked: {len(comp['raptor384'])}")

    def predict(raptor_weight, w_s6):
        return full_chain(comp["raptor384"], comp["raptor224"], comp["ours"], comp["reader"], comp["s6"],
                          raptor_weight, w_s6).loc[ids, LABELS].to_numpy()

    print("\nsingle components (gold-58 macro AUC):")
    for k in ("raptor384", "raptor224", "ours", "reader", "s6"):
        print(f"  {k:10s} {macro_auc(y, comp[k].loc[ids].to_numpy()):.4f}")

    preds = {}
    for p in sorted((PIPELINE_DIR / "variants").glob("*.json")):
        v = Variant.from_json(p)
        preds[v.name] = predict(v.raptor_weight, v.w_s6)
    pd.set_option("display.width", 220)
    tab = pd.DataFrame({k: per_finding_auc(y, p) for k, p in preds.items()}, index=LABELS)
    tab.loc["MACRO"] = tab.mean()
    print("\nvariants (gold-58 AUC per finding):")
    print(tab.round(4).to_string())

    v90 = Variant.from_json(PIPELINE_DIR / "variants" / "v90.json")
    print("\nS6 weight sweep on the v90 Stage 3:")
    for w in (0.0001, 0.10, 0.20, 0.30, 0.35, 0.45, 0.60):
        print(f"  W_S6 {w:.2f}: {macro_auc(y, predict(v90.raptor_weight, w)):.4f}")

    print("\npaired bootstrap vs v90 (in-sample):")
    for k, r in paired_bootstrap(y, preds, "v90", n_boot=2000).items():
        print(f"  {k:10s} mean {r['mean']:+.4f}  90% CI [{r['lo5']:+.4f}, {r['hi95']:+.4f}]  "
              f"P(better) {r['p_better']:.2f}  P(worse) {r['p_worse']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    np.seterr(all="ignore")
    main(sys.argv[1], sys.argv[2])
