"""Score the v0 regex labeler against the 58 expert-labelled studies.

This is the baseline the LLM labeler has to beat. Run: python -m knee.labels.eval_rules
"""


import pandas as pd

from knee.labels.rules_en import LABELS, label_report
from knee.paths import REPO_ROOT as ROOT

EN_CUES = (" the ", " with ", " is ", " no ", " and ")


def is_english(text: str) -> bool:
    return sum(cue in text.lower() for cue in EN_CUES) >= 2


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    labelled = train[train[LABELS[0]].notna()]
    # The rules are English-only, so scoring them on other languages would
    # understate them. Restrict to the English subset.
    sub = labelled[labelled.Report.map(is_english)]

    pred = pd.DataFrame([label_report(t) for t in sub.Report], index=sub.index)
    gt = sub[LABELS].astype(int)

    rows = []
    for c in LABELS:
        tp = int(((pred[c] == 1) & (gt[c] == 1)).sum())
        fp = int(((pred[c] == 1) & (gt[c] == 0)).sum())
        fn = int(((pred[c] == 0) & (gt[c] == 1)).sum())
        tn = int(((pred[c] == 0) & (gt[c] == 0)).sum())
        rows.append({
            "condition": c, "n_pos": int(gt[c].sum()),
            "TP": tp, "FP": fp, "FN": fn,
            "acc": round((tp + tn) / len(sub), 3),
            "prec": round(tp / (tp + fp), 2) if tp + fp else None,
            "rec": round(tp / (tp + fn), 2) if tp + fn else None,
        })

    out = pd.DataFrame(rows)
    print(f"English labelled studies: {len(sub)} / {len(labelled)}\n")
    print(out.to_string(index=False))
    print(f"\nMEAN ACCURACY: {out.acc.mean():.3f}")
    exact = (pred[LABELS].values == gt[LABELS].values).all(axis=1).sum()
    print(f"studies with all 12 correct: {exact}/{len(sub)}")


if __name__ == "__main__":
    main()
