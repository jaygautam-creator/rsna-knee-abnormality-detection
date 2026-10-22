"""Build submission-notebook variants from the v90 base notebook and a small JSON config.

Every final-day submission differed from v90 only in a few constants, so each one is a config in
``pipeline/variants/`` instead of a 1 MB notebook copy. The builder edits exactly those constants, by anchored string
replacement that must match once, and checks the result:

* Stage 3 (cell with ``W_RAPTOR_C``): the per-finding Raptor weight dict.
* Part C (cell with ``W_S6``): the S6 weight, scalar or per finding, and optionally the S6 deadline constants.
* the title cell.

The public base notebook ships with an empty ``S6_PARITY`` (the reference rows are 3 public test-study IDs, i.e.
competition data). Pass ``parity`` (loaded from the gitignored ``pipeline/local/s6_parity.json``) to re-insert them;
without it the notebook's commit run stops at the parity check on purpose.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from knee.constants import LABELS

TITLE_ANCHOR = "# RSNA Knee v90: v74 (Raptor-384"
RAPTOR_ANCHOR = ("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, "
                 "\"Baker's\": 0.30}   # [V2]")
S6_ANCHOR = "\nW_S6 = 0.20\n"
S6_ASSERT = "assert 0.0 < W_S6 < 0.5 and S6_STOP_H < S6_KILL_H < 8.75"
S6_BLEND = "            b[ok] = (1.0 - w) * r74[ok] + w * _c_pct(s6c) + w * (r74[ok] - _c_pct(r74[ok]))\n"
TIMING_ANCHORS = {"S6_STOP_H": "S6_STOP_H = 8.40 ", "S6_KILL_H": "S6_KILL_H = 8.55 ", "S6_MIN_LEFT_H": "S6_MIN_LEFT_H = 0.50 "}
PARITY_PLACEHOLDER = "S6_PARITY = {}            # [public copy]"


@dataclass(frozen=True)
class Variant:
    name: str
    raptor_weight: dict[str, float]
    w_s6: float | dict[str, float]
    s6_timing: dict[str, float] = field(default_factory=dict)
    description: str = ""
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> Variant:
        d = json.loads(Path(path).read_text())
        return cls(name=d["name"], raptor_weight=d["stage3_raptor_weight"], w_s6=d["w_s6"],
                   s6_timing=d.get("s6_timing", {}), description=d.get("description", ""),
                   provenance=d.get("provenance", {}))

    def validate(self) -> None:
        """Mirror the notebook's own asserts, so a bad config fails here rather than after a GPU run."""
        def need(ok: bool, msg: str) -> None:
            if not ok:
                raise ValueError(f"variant {self.name}: {msg}")

        need(set(self.raptor_weight) <= set(LABELS), f"unknown findings {set(self.raptor_weight) - set(LABELS)}")
        need(all(0.0 <= float(v) <= 1.0 for v in self.raptor_weight.values()), "Stage 3 weights must be in [0, 1]")
        if isinstance(self.w_s6, dict):
            need(sorted(self.w_s6) == sorted(LABELS), "a per-finding W_S6 must name all 12 findings")
            need(all(0.0 <= float(v) < 0.75 for v in self.w_s6.values()), "per-finding W_S6 must be in [0, 0.75)")
        else:
            need(0.0 < float(self.w_s6) < 0.5, "scalar W_S6 must be in (0, 0.5)")
        need(set(self.s6_timing) <= set(TIMING_ANCHORS), f"unknown timing keys {set(self.s6_timing) - set(TIMING_ANCHORS)}")
        t = {"S6_STOP_H": 8.40, "S6_KILL_H": 8.55, **self.s6_timing}
        need(t["S6_STOP_H"] < t["S6_KILL_H"] < 8.75, "need S6_STOP_H < S6_KILL_H < 8.75 (Kaggle's 9 h limit)")


def _replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise ValueError(f"anchor found {n} times (expected 1): {old[:80]!r}")
    return text.replace(old, new, 1)


def _find_cell(nb: dict, needle: str) -> int:
    hits = [i for i, c in enumerate(nb["cells"]) if needle in "".join(c["source"])]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} cells contain {needle!r}")
    return hits[0]


def _set_source(nb: dict, i: int, text: str) -> None:
    cell = nb["cells"][i]
    cell["source"] = text
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        compile(text if not text.startswith("%%") else text.split("\n", 1)[1], f"cell{i}", "exec")
    else:
        cell.pop("outputs", None)   # a markdown cell with "outputs": null crashes Kaggle's final render


def format_parity(parity: dict[str, list[float]]) -> str:
    rows = "".join(f"    {uid!r}: {json.dumps(v)},\n" for uid, v in parity.items())
    return "S6_PARITY = {             # reference S6 probabilities on the 3 public test studies\n" + rows + "}\n"


def build_notebook(base: dict, v: Variant, parity: dict | None = None) -> dict:
    """Return a new notebook dict for variant ``v``; ``base`` is not modified."""
    v.validate()
    nb = copy.deepcopy(base)

    i8 = _find_cell(nb, RAPTOR_ANCHOR)   # (cell 6 has a stray "W_RAPTOR_C = {" inside a comment)
    s = "".join(nb["cells"][i8]["source"])
    wr = {k: float(x) for k, x in v.raptor_weight.items()}
    s = _replace_once(s, RAPTOR_ANCHOR, f"W_RAPTOR_C = {wr!r}   # [variant {v.name}] [V2]")
    _set_source(nb, i8, s)

    i10 = _find_cell(nb, S6_ANCHOR.strip("\n"))
    c = "".join(nb["cells"][i10]["source"])
    if isinstance(v.w_s6, dict):
        w = {k: round(float(v.w_s6[k]), 3) for k in LABELS}
        c = _replace_once(c, S6_ANCHOR, f"\nW_S6 = {w!r}   # per-finding S6 weight\n")
        c = _replace_once(c, S6_ASSERT, "assert all(0.0 <= _v < 0.75 for _v in W_S6.values()) and "
                                        "sorted(W_S6) == sorted(S6_LABELS) and S6_STOP_H < S6_KILL_H < 8.75")
        c = _replace_once(c, S6_BLEND,
                          "            wc = float(w[c]) if isinstance(w, dict) else float(w)   # per-finding weight\n"
                          "            b[ok] = (1.0 - wc) * r74[ok] + wc * _c_pct(s6c) + wc * (r74[ok] - _c_pct(r74[ok]))\n")
    else:
        c = _replace_once(c, S6_ANCHOR, f"\nW_S6 = {float(v.w_s6):.2f}\n")
    for key, value in v.s6_timing.items():
        c = _replace_once(c, TIMING_ANCHORS[key], f"{key} = {float(value):.2f} ")
    if parity is not None:
        start = c.index(PARITY_PLACEHOLDER)
        end = c.index("\n", c.index("\n", start) + 1) + 1        # the placeholder spans two lines
        c = c[:start] + format_parity(parity) + c[end:]
    _set_source(nb, i10, c)

    m0 = "".join(nb["cells"][0]["source"])
    _set_source(nb, 0, _replace_once(m0, TITLE_ANCHOR, f"# RSNA Knee {v.name}: v74 (Raptor-384"))

    check_notebook(nb, base)
    return nb


def check_notebook(nb: dict, base: dict) -> None:
    """Structural checks that caught real failures: same cell count and keys, no ``outputs: null`` anywhere."""
    assert len(nb["cells"]) == len(base["cells"])
    for i, (a, b) in enumerate(zip(nb["cells"], base["cells"], strict=False)):
        assert set(a) == set(b), f"cell {i} keys changed: {set(a) ^ set(b)}"
        assert not ("outputs" in a and a["outputs"] is None), f"cell {i} has outputs=None"
        assert not (a["cell_type"] == "markdown" and "outputs" in a), f"markdown cell {i} carries outputs"


def read_settings(nb: dict) -> dict:
    """Parse the constants a variant controls back out of a notebook (for checks and provenance)."""
    want = {"W_RAPTOR_C", "W_S6", "S6_STOP_H", "S6_KILL_H", "S6_MIN_LEFT_H"}
    out = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.startswith("%%"):
            continue
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in want:
                    try:
                        out[name] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    return out


def changed_cells(nb: dict, base: dict) -> list[int]:
    return [i for i, (a, b) in enumerate(zip(nb["cells"], base["cells"], strict=False)) if "".join(a["source"]) != "".join(b["source"])]


def write_kernel(out_dir: str | Path, nb: dict, metadata: dict, owner: str, slug: str) -> Path:
    """Write ``<slug>.ipynb`` and ``kernel-metadata.json`` ready for ``kaggle kernels push -p out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fn = f"{slug}.ipynb"
    (out / fn).write_text(json.dumps(nb, indent=1) + "\n")
    meta = dict(metadata, id=f"{owner}/{slug}", title=slug.replace("-", " ").title(), code_file=fn)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out
