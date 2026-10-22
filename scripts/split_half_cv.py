"""Held-out (split-half) test of blend-weight selection rules on the 58 expert-labelled studies.

    python scripts/split_half_cv.py runs/gold58 data/train.csv

Each rule picks weights per finding on a random half (29 studies) and is scored on the other half, 600 times.
A rule is trusted only if it helps the held-out half in >= 80 % of splits and hurts in <= 10 %.
Baseline: v90's Stage 3 weights with W_S6 = 0.35.
"""

import sys

import numpy as np

from knee.blend import V90_RAPTOR_WEIGHT, full_chain, pct
from knee.constants import LABELS
from knee.io import load_components, load_gold
from knee.metrics import split_half

GRID = [0.15, 0.30, 0.50, 0.65, 0.85]


def main(run_dir: str, train_csv: str) -> None:
    comp = load_components(run_dir)
    gold = load_gold(train_csv)
    ids = [u for u in gold.index if u in comp["raptor384"].index]
    rows = [comp["raptor384"].index.get_loc(u) for u in ids]
    y = gold.loc[ids].to_numpy()
    base_w = {c: V90_RAPTOR_WEIGHT.get(c, 0.5) for c in LABELS}

    def column(c, raptor_w, w_s6=0.35):
        w = dict(base_w, **{c: raptor_w})
        return full_chain(comp["raptor384"], comp["raptor224"], comp["ours"], comp["reader"], comp["s6"],
                          w, w_s6)[c].to_numpy()[rows]

    def run(name, cands):
        r = split_half(y, cands)
        print(f"  {name:42s} held-out mean {r['mean']:+.4f}  helps {r['p_helps']:.2f}  hurts {r['p_hurts']:.2f}")
        return r

    print("Stage 3 Raptor-weight rules (baseline: v90 weights, W_S6 0.35):")
    base_cols = {j: column(c, base_w[c]) for j, c in enumerate(LABELS)}
    run("keep v90 weight or revert to 0.5 (v94 rule)", {j: [base_cols[j], column(c, 0.5)] for j, c in enumerate(LABELS)})
    run("free choice from the grid", {j: [base_cols[j]] + [column(c, g) for g in GRID] for j, c in enumerate(LABELS)})

    print("\nco-occurrence nudge (final_c <- rank((1-a) final_c + a final_d), a in {0.1, 0.2, 0.3}):")
    final = {c: column(c, base_w[c]) for c in LABELS}
    for a_, b_ in [("Synovitis", "Effusion"), ("Lateral Meniscus", "Lateral OA"), ("Fracture", "ACL")]:
        j = LABELS.index(a_)
        cands = [final[a_]] + [pct((1 - al) * final[a_] + al * final[b_]) for al in (0.1, 0.2, 0.3)]
        run(f"{a_} <- {b_}", {j: cands})


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    np.seterr(all="ignore")
    main(sys.argv[1], sys.argv[2])
