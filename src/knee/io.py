"""Loaders for evaluation-run outputs and the expert labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from knee.constants import LABELS

COMPONENT_FILES = {"raptor384": "raptor_raw.csv", "raptor224": "raptor224_raw.csv", "ours": "ours_sub.csv",
                   "reader": "_own.csv", "s6": "s6_raw.csv"}


def load_components(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Per-model prediction dumps of one evaluation run, aligned on Raptor-384's row order."""
    run_dir = Path(run_dir)
    comp = {k: pd.read_csv(run_dir / f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS]
            for k, f in COMPONENT_FILES.items()}
    idx = comp["raptor384"].index
    return {k: v.loc[idx] for k, v in comp.items()}


def load_gold(train_csv: str | Path) -> pd.DataFrame:
    """The 58 studies of train.csv that carry expert labels (all 12 columns non-null)."""
    tr = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    gold = tr[tr["ACL"].notna()].set_index("StudyInstanceUID")[LABELS].astype(int)
    if len(gold) != 58:
        raise ValueError(f"expected 58 expert-labelled studies, found {len(gold)}")
    return gold
