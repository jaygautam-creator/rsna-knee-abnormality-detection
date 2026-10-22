"""Grouped CV folds for the knee competition.

Labels are derived from the radiology report, so studies sharing a report get
identical targets. 4,407 studies hold only 4,257 unique reports, and one report
is shared by 37 studies — splitting such a group across folds lets the model
score on validation rows it memorised in training. Group on the report, not the
study.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from knee.paths import REPO_ROOT

_WS = re.compile(r"\s+")


def report_group(reports: pd.Series) -> pd.Series:
    norm = reports.fillna("").str.lower().str.replace(_WS, " ", regex=True).str.strip()
    return norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())


def make_folds(train: pd.DataFrame, labels: pd.DataFrame, n_splits: int = 5,
               seed: int = 42) -> pd.Series:
    """Return a fold index per row of `train`, grouped by report and stratified
    on overall abnormality burden so folds carry comparable positive rates."""
    groups = report_group(train.Report)
    cols = [c for c in labels.columns if c != "StudyInstanceUID"]
    burden = (labels[cols].to_numpy() > 0.5).sum(1)
    strata = np.digitize(burden, [1, 3, 5, 7])

    fold = pd.Series(-1, index=train.index, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(splitter.split(train, strata, groups)):
        fold.iloc[va] = k
    assert (fold >= 0).all()
    return fold


# ---------------------------------------------------------------------------
# Frozen folds
#
# `make_folds` stratifies on abnormality burden derived from `labels`, and
# `StratifiedGroupKFold` assigns groups greedily — so a small change in the
# label version cascades. Measured 2026-08-20: moving v3 -> v4 labels changed
# burden on 6.6% of studies and reassigned **74.6%** of folds. That silently
# broke comparability between every training run, leaving only 357 shared
# held-out rows to compare v3 against v5 when there should have been ~880.
#
# So the split is computed once and committed. Every run reads it. Improving
# the labels must never move a study between folds again.
# ---------------------------------------------------------------------------

FROZEN = REPO_ROOT / "data" / "folds.csv"


def freeze_folds(train: pd.DataFrame, labels: pd.DataFrame, path: Path = FROZEN,
                 n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Compute the split once and write it. Refuses to clobber an existing file —
    regenerating is exactly the mistake this exists to prevent."""
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. The frozen split is deliberately immutable; "
            "delete it by hand only if you intend to invalidate every run that used it.")
    fold = make_folds(train, labels, n_splits=n_splits, seed=seed)
    out = pd.DataFrame({"StudyInstanceUID": train.StudyInstanceUID.to_numpy(),
                        "fold": fold.to_numpy()})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_folds(train: pd.DataFrame, path: Path = FROZEN) -> np.ndarray:
    """Fold index per row of `train`, in `train` row order.

    Use this everywhere instead of `make_folds`. Asserts the frozen file covers
    exactly the studies in `train`, so a changed corpus fails loudly rather than
    silently realigning."""
    fr = pd.read_csv(path)
    missing = set(train.StudyInstanceUID) - set(fr.StudyInstanceUID)
    assert not missing, f"{len(missing)} studies absent from {path.name}"
    m = fr.set_index("StudyInstanceUID").fold
    return train.StudyInstanceUID.map(m).to_numpy()
