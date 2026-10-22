"""Build a Kaggle kernel folder for a submission variant.

    python scripts/build_variant.py pipeline/variants/v94.json --owner <kaggle-username> --out build/v94
    kaggle kernels push -p build/v94

The S6 parity reference rows are re-inserted from pipeline/local/s6_parity.json when that file exists (it is
gitignored: the rows are public-test study IDs). Without it the notebook's commit run stops at the parity check.
"""

import argparse
import json
import sys
from pathlib import Path

from knee.paths import LOCAL_DIR, PIPELINE_DIR
from knee.variants import Variant, build_notebook, changed_cells, read_settings, write_kernel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", type=Path, help="variant JSON in pipeline/variants/")
    ap.add_argument("--owner", required=True, help="Kaggle username that will own the kernel")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--parity", type=Path, default=LOCAL_DIR / "s6_parity.json")
    ap.add_argument("--base", type=Path, default=PIPELINE_DIR / "base" / "infer-v90.ipynb")
    a = ap.parse_args()

    v = Variant.from_json(a.variant)
    base = json.loads(a.base.read_text())
    parity = json.loads(a.parity.read_text()) if a.parity.is_file() else None
    nb = build_notebook(base, v, parity)
    meta = json.loads((a.base.parent / "kernel-metadata.json").read_text())
    slug = f"knee-infer-{v.name}"
    out = write_kernel(a.out, nb, meta, a.owner, slug)
    print(f"built {out} | kernel {a.owner}/{slug} | cells changed vs base: {changed_cells(nb, base)}")
    print("settings:", json.dumps(read_settings(nb)))
    if parity is None:
        print("WARNING: no parity file; the commit run will stop at the S6 parity check", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
