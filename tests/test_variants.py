import json

import pytest

from knee.constants import LABELS
from knee.paths import PIPELINE_DIR
from knee.variants import PARITY_PLACEHOLDER, Variant, build_notebook, changed_cells, read_settings

BASE = json.loads((PIPELINE_DIR / "base" / "infer-v90.ipynb").read_text())
VARIANTS = sorted((PIPELINE_DIR / "variants").glob("*.json"))
FAKE_PARITY = {"study-a": [0.1] * 12, "study-b": [0.2] * 12, "study-c": [0.3] * 12}


def test_base_notebook_settings_are_v90():
    s = read_settings(BASE)
    assert s["W_S6"] == 0.20
    assert s["W_RAPTOR_C"] == {"Medial OA": 0.5, "Lateral OA": 0.15, "PF OA": 0.65, "Synovitis": 0.85, "Baker's": 0.30}
    assert (s["S6_STOP_H"], s["S6_KILL_H"], s["S6_MIN_LEFT_H"]) == (8.40, 8.55, 0.50)


def test_public_base_carries_no_parity_rows():
    text = json.dumps(BASE)
    assert PARITY_PLACEHOLDER in text
    assert "1.2.826.0.1.3680043" not in text


@pytest.mark.parametrize("path", VARIANTS, ids=lambda p: p.stem)
def test_variant_builds_and_round_trips(path):
    v = Variant.from_json(path)
    nb = build_notebook(BASE, v)
    s = read_settings(nb)
    assert s["W_RAPTOR_C"] == pytest.approx(v.raptor_weight)
    if isinstance(v.w_s6, dict):
        assert s["W_S6"] == pytest.approx({k: round(v.w_s6[k], 3) for k in LABELS})
    else:
        assert s["W_S6"] == pytest.approx(v.w_s6)
    for k, val in v.s6_timing.items():
        assert s[k] == pytest.approx(val)
    assert set(changed_cells(nb, BASE)) <= {0, 8, 10}


@pytest.mark.parametrize("path", VARIANTS, ids=lambda p: p.stem)
def test_variant_provenance_is_recorded(path):
    d = json.loads(path.read_text())
    assert d["provenance"]["kernel"].startswith("knee-infer-")
    assert len(d["provenance"]["submitted_notebook_sha256"]) == 64


def test_parity_rows_are_inserted_when_given():
    v = Variant.from_json(PIPELINE_DIR / "variants" / "v94.json")
    nb = build_notebook(BASE, v, FAKE_PARITY)
    src = "".join(nb["cells"][10]["source"])
    assert PARITY_PLACEHOLDER not in src and "'study-b': [0.2" in src
    compile(src, "cell10", "exec")


def test_markdown_cells_never_carry_outputs():
    v = Variant.from_json(PIPELINE_DIR / "variants" / "v96-c.json")
    nb = build_notebook(BASE, v)
    assert all("outputs" not in c for c in nb["cells"] if c["cell_type"] == "markdown")


@pytest.mark.parametrize("bad", [
    dict(raptor_weight={"Not a finding": 0.5}, w_s6=0.2),
    dict(raptor_weight={}, w_s6=0.6),
    dict(raptor_weight={}, w_s6={"ACL": 0.2}),
    dict(raptor_weight={}, w_s6=0.2, s6_timing={"S6_STOP_H": 8.7, "S6_KILL_H": 8.6}),
])
def test_invalid_variants_are_rejected(bad):
    with pytest.raises(ValueError):
        build_notebook(BASE, Variant(name="bad", **bad))
