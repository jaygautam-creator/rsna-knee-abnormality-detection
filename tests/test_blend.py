import numpy as np
import pandas as pd
import pytest

from knee.blend import V90_RAPTOR_WEIGHT, part_b, part_c, pct, pct_on, stage3
from knee.constants import LABELS


def frame(rng, n=40, fill=None):
    idx = [f"s{i:03d}" for i in range(n)]
    data = rng.random((n, len(LABELS))) if fill is None else np.full((n, len(LABELS)), fill)
    return pd.DataFrame(data, index=idx, columns=LABELS)


def test_pct_averages_ties_and_is_in_unit_interval():
    r = pct([3.0, 1.0, 1.0, 2.0])
    assert r.tolist() == [1.0, 0.375, 0.375, 0.75]


def test_pct_on_ranks_only_selected_rows():
    r = pct_on([5.0, 1.0, 3.0, 9.0], [True, False, True, True])
    assert np.isnan(r[1])
    assert r[[0, 2, 3]].tolist() == pytest.approx([2 / 3, 1 / 3, 1.0])


def test_stage3_weight_one_follows_raptor_half_and_zero_follows_r6():
    rng = np.random.default_rng(0)
    r384, r224, ours = frame(rng), frame(rng), frame(rng)
    only_raptor = stage3(r384, r224, ours, raptor_weight=1.0)
    only_r6 = stage3(r384, r224, ours, raptor_weight=0.0)
    for c in LABELS:
        expect = 0.65 * pct(r384[c]) + 0.35 * pct(r224[c])
        assert np.allclose(only_raptor[c], expect)
        assert np.allclose(only_r6[c], pct(ours[c]))


def test_stage3_failed_raptor_study_takes_the_other_model():
    rng = np.random.default_rng(1)
    r384, r224, ours = frame(rng), frame(rng), frame(rng)
    r384.iloc[0] = 0.5                       # Raptor-384 failure marker: 0.5 in every column
    out = stage3(r384, r224, ours, raptor_weight=1.0)
    assert np.isfinite(out.to_numpy()).all()
    assert out["ACL"].iloc[0] == pytest.approx(pct(r224["ACL"])[0])


def test_part_b_zero_weight_is_rank_of_stack():
    rng = np.random.default_rng(2)
    st, reader = frame(rng), frame(rng)
    out = part_b(st, reader, reader_weight={})
    for c in LABELS:
        assert np.allclose(out[c], pct(st[c]))


def test_part_c_full_coverage_equals_rank_of_linear_mix():
    rng = np.random.default_rng(3)
    v74, s6 = frame(rng), frame(rng)
    out = part_c(v74, s6, 0.35)
    for c in LABELS:
        r74 = pct(v74[c])
        mix = 0.65 * r74 + 0.35 * pct(s6[c])
        assert np.allclose(out[c], pct(mix + 1e-9 * r74))


def test_part_c_uniform_dict_equals_scalar():
    rng = np.random.default_rng(4)
    v74, s6 = frame(rng), frame(rng)
    a = part_c(v74, s6, 0.2)
    b = part_c(v74, s6, {c: 0.2 for c in LABELS})
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_part_c_per_finding_weight_only_moves_that_finding():
    rng = np.random.default_rng(5)
    v74, s6 = frame(rng), frame(rng)
    a = part_c(v74, s6, 0.2)
    b = part_c(v74, s6, dict({c: 0.2 for c in LABELS}, Fracture=0.4))
    changed = [c for c in LABELS if not np.array_equal(a[c], b[c])]
    assert changed == ["Fracture"]


def test_part_c_partial_coverage_keeps_uncovered_order():
    rng = np.random.default_rng(6)
    v74, s6 = frame(rng, n=60), frame(rng, n=60)
    s6.iloc[30:] = np.nan                    # S6 scored only the first half (deadline)
    out = part_c(v74, s6, 0.45)
    for c in LABELS:
        unc = out[c].to_numpy()[30:]
        ref = v74[c].to_numpy()[30:]
        assert (np.argsort(unc) == np.argsort(ref)).all()


def test_v90_weights_cover_known_findings_only():
    assert set(V90_RAPTOR_WEIGHT) <= set(LABELS)
