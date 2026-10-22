"""Evaluation on a small labelled set: macro ROC-AUC, paired bootstrap, and the split-half selection test.

The split-half test is the guard against tuning on 58 studies: a weight-picking *rule* is applied to one random half
and scored on the other half, many times. Only rules that help the held-out half most of the time are trusted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def per_finding_auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ROC-AUC per column; NaN where the column has a single class."""
    y, p = np.asarray(y), np.asarray(p)
    return np.array([_auc(y[:, j], p[:, j]) for j in range(y.shape[1])])


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Unweighted mean ROC-AUC over the columns that have both classes (the competition metric)."""
    return float(np.nanmean(per_finding_auc(y, p)))


def paired_bootstrap(y: np.ndarray, preds: Mapping[str, np.ndarray], baseline: str, n_boot: int = 2000,
                     seed: int = 0) -> dict[str, dict[str, float]]:
    """Resample studies with replacement; report each candidate's macro-AUC change vs ``baseline``.

    Returns {name: {mean, lo5, hi95, p_better, p_worse}}. In-sample: if the candidates were chosen on the same
    studies, these probabilities are optimistic. Use ``split_half`` for an honest estimate of a selection rule.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = {k: [] for k in preds}
    arrays = {k: np.asarray(p) for k, p in preds.items()}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        g = y[idx]
        ok = [j for j in range(y.shape[1]) if 0 < g[:, j].sum() < n]
        scores = {k: np.mean([roc_auc_score(g[:, j], p[idx, j]) for j in ok]) for k, p in arrays.items()}
        for k in arrays:
            deltas[k].append(scores[k] - scores[baseline])
    out = {}
    for k, d in deltas.items():
        d = np.asarray(d)
        out[k] = {"mean": float(d.mean()), "lo5": float(np.percentile(d, 5)), "hi95": float(np.percentile(d, 95)),
                  "p_better": float((d > 0).mean()), "p_worse": float((d < 0).mean())}
    return out


def split_half(y: np.ndarray, candidates: Mapping[int, Sequence[np.ndarray]], n_splits: int = 600, seed: int = 7,
               n_findings: int | None = None) -> dict:
    """Held-out test of a per-finding selection rule.

    ``candidates[j]`` lists alternative prediction columns for finding ``j``; index 0 is the baseline. On each split
    the rule picks, per finding, the candidate with the best AUC on half A (ties keep the baseline) and is scored on
    half B. The change in macro AUC counts findings without candidates as unchanged.

    Returns {mean, median, p_helps, p_hurts, per_finding: {j: mean held-out change}}.
    """
    y = np.asarray(y)
    n = len(y)
    n_findings = n_findings or y.shape[1]
    rng = np.random.default_rng(seed)
    totals, per = [], {j: [] for j in candidates}
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        tot = 0.0
        for j, cands in candidates.items():
            score_a = [_auc(y[a, j], np.asarray(c)[a]) for c in cands]
            if np.all(np.isnan(score_a)):
                continue
            k = int(np.nanargmax(score_a))
            b0, b1 = _auc(y[b, j], np.asarray(cands[0])[b]), _auc(y[b, j], np.asarray(cands[k])[b])
            if np.isnan(b0) or np.isnan(b1):
                continue
            tot += (b1 - b0) / n_findings
            per[j].append(b1 - b0)
        totals.append(tot)
    t = np.asarray(totals)
    return {"mean": float(t.mean()), "median": float(np.median(t)), "p_helps": float((t > 0).mean()),
            "p_hurts": float((t < 0).mean()), "per_finding": {j: float(np.mean(v)) if v else 0.0 for j, v in per.items()}}
