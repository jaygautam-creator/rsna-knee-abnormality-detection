"""The final-stage blend maths of the inference notebook, as plain functions.

Every function mirrors a cell of ``pipeline/base/infer-v90.ipynb`` exactly, so offline experiments on dumped component
predictions reproduce what the notebook would have written:

* ``stage3``  (cell 8)  per-finding rank blend of the Raptor half (Raptor-384 + Raptor-224) and R6.
* ``part_b``  (cell 9)  rank blend of the 2.5D MIL reader into the Stage 3 output ("end" mode), then re-ranked.
* ``part_c``  (cell 10) S6 all-data blend: (1 - w) R74 + w R_S6 on the studies S6 scored, a centred nudge otherwise.

All inputs are DataFrames indexed by StudyInstanceUID with the 12 finding columns. The metric is rank-based (macro
ROC-AUC), so every stage works on percentile ranks.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from knee.constants import LABELS

# Stage 3 weights of the original public recipe (v74 .. v90): share of the Raptor half, R6 gets the rest.
V90_RAPTOR_WEIGHT = {"Medial OA": 0.5, "Lateral OA": 0.15, "PF OA": 0.65, "Synovitis": 0.85, "Baker's": 0.30}
DEFAULT_RAPTOR_WEIGHT = 0.5
A224 = 0.35   # share of the Raptor half given to the 224-pixel model
# Part B reader weights per finding (fixed in the public recipe).
PART_B_WEIGHT = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02,
                 "MCL": 0.02, "Lateral Meniscus": 0.008, "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008,
                 "Effusion": 0.004, "Synovitis": 0.003}


def pct(values) -> np.ndarray:
    """Percentile rank in (0, 1], ties averaged (pandas ``rank(method='average', pct=True)``)."""
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average", pct=True).to_numpy()


def pct_on(values, ok) -> np.ndarray:
    """Percentile rank among the rows in ``ok`` only; other rows are NaN."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool)
    out = np.full(len(values), np.nan)
    if ok.any():
        out[ok] = pct(values[ok])
    return out


def _weight(w: float | Mapping[str, float], finding: str, default: float) -> float:
    return float(w.get(finding, default)) if isinstance(w, Mapping) else float(w)


def stage3(raptor384: pd.DataFrame, raptor224: pd.DataFrame, ours: pd.DataFrame,
           raptor_weight: float | Mapping[str, float] = V90_RAPTOR_WEIGHT, a224: float = A224,
           labels: list[str] = LABELS) -> pd.DataFrame:
    """Stage 3: ``w * Raptor half + (1 - w) * rank(R6)`` per finding (not re-ranked, as in the notebook).

    A Raptor model that failed on a study leaves 0.5 in every column; that study takes the other Raptor model's rank,
    and R6 alone if both failed.
    """
    idx = raptor384.index
    r224_, ours_ = raptor224.loc[idx], ours.loc[idx]
    f384 = (raptor384[labels].to_numpy() == 0.5).all(axis=1)
    f224 = (r224_[labels].to_numpy() == 0.5).all(axis=1)
    both = f384 & f224
    out = {}
    for c in labels:
        r384, r224 = pct_on(raptor384[c], ~f384), pct_on(r224_[c], ~f224)
        rr = np.where(f384, r224, np.where(f224, r384, (1.0 - a224) * r384 + a224 * r224))
        ro = pct(ours_[c])
        wr = _weight(raptor_weight, c, DEFAULT_RAPTOR_WEIGHT)
        b = wr * np.nan_to_num(rr) + (1.0 - wr) * ro
        b[both] = ro[both]
        out[c] = b
    return pd.DataFrame(out, index=idx)


def part_b(stack: pd.DataFrame, reader: pd.DataFrame, reader_weight: Mapping[str, float] = PART_B_WEIGHT,
           labels: list[str] = LABELS) -> pd.DataFrame:
    """Part B ("end" mode): ``rank((1 - w) rank(stack) + w rank(reader))`` per finding."""
    reader = reader.loc[stack.index]
    out = {}
    for c in labels:
        w = float(reader_weight.get(c, 0.0))
        out[c] = pct((1.0 - w) * pct(stack[c]) + w * pct(reader[c]))
    return pd.DataFrame(out, index=stack.index)


def part_c(v74: pd.DataFrame, s6: pd.DataFrame, w_s6: float | Mapping[str, float],
           labels: list[str] = LABELS) -> pd.DataFrame:
    """Part C: blend S6 into the v74 ranks on the studies S6 scored (rows of ``s6`` that are not all-NaN).

    Full coverage gives exactly ``rank((1 - w) R74 + w R_S6)``; partial coverage moves only the covered rows, by a
    nudge centred so that uncovered studies keep their v74 rank. Exact ties of the mix follow v74's order.
    """
    s6 = s6.reindex(v74.index)
    ok = ~s6[labels].isna().all(axis=1).to_numpy()
    out = {}
    for c in labels:
        r74 = pct(v74[c])
        b = r74.copy()
        if ok.any():
            w = _weight(w_s6, c, 0.0)
            rs6 = pct(s6[c].to_numpy()[ok])
            b[ok] = (1.0 - w) * r74[ok] + w * rs6 + w * (r74[ok] - pct(r74[ok]))
        out[c] = pct(b + 1e-9 * r74)
    return pd.DataFrame(out, index=v74.index)


def full_chain(raptor384: pd.DataFrame, raptor224: pd.DataFrame, ours: pd.DataFrame, reader: pd.DataFrame,
               s6: pd.DataFrame, raptor_weight: float | Mapping[str, float] = V90_RAPTOR_WEIGHT,
               w_s6: float | Mapping[str, float] = 0.20) -> pd.DataFrame:
    """Stage 3 -> Part B -> Part C, i.e. the notebook's final submission from its dumped components."""
    st = stage3(raptor384, raptor224, ours, raptor_weight)
    v74 = part_b(st, reader)
    return part_c(v74, s6, w_s6)
