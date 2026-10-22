"""Check a finished Kaggle commit run before submitting it.

    kaggle kernels output <owner>/<kernel> -p runs/v94 --file-pattern '.*\\.(csv|json)$'
    python scripts/validate_run.py runs/v94 pipeline/variants/v94.json

Exit code 0 only if every check passes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knee.constants import LABELS
from knee.variants import Variant


def main(run_dir: str, variant_json: str) -> int:
    d, v = Path(run_dir), Variant.from_json(variant_json)
    s6 = json.loads((d / "s6_receipt.json").read_text())
    blend = json.loads((d / "blend_receipt.json").read_text())
    sub = pd.read_csv(d / "submission.csv")
    vals = sub[LABELS].to_numpy()
    w = s6.get("w_s6")
    want_w = v.w_s6 if isinstance(v.w_s6, dict) else float(v.w_s6)
    checks = {
        "S6 blended": s6.get("status") == "blended",
        "S6 parity exact (0.0)": s6.get("parity_max_abs_dp") == 0.0,
        "S6 weight as configured": (w == want_w) if isinstance(want_w, dict) else abs(float(w) - want_w) < 1e-9,
        "Stage 3 weights as configured": {k: float(x) for k, x in blend["weights"]["raptor_per_class"].items()}
        == {k: float(x) for k, x in v.raptor_weight.items()},
        "no model failures": blend["raptor384_failed"] == 0 and blend["raptor224_failed"] == 0 and blend["ours_failed"] == 0,
        "columns match sample_submission": list(sub.columns) == ["StudyInstanceUID"] + LABELS,
        "values finite, in [0, 1]": bool(np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all()),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    ok = all(checks.values())
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
