import numpy as np
import pytest

from knee.metrics import macro_auc, paired_bootstrap, per_finding_auc, split_half


def labels(rng, n=58, k=12):
    y = (rng.random((n, k)) < 0.4).astype(int)
    y[0], y[1] = 1, 0                     # every column has both classes
    return y


def test_perfect_and_inverted_predictions():
    rng = np.random.default_rng(0)
    y = labels(rng)
    assert macro_auc(y, y + 0.0) == pytest.approx(1.0)
    assert macro_auc(y, 1.0 - y) == pytest.approx(0.0)


def test_single_class_column_is_skipped():
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    p = np.array([[0.9, 0.1], [0.1, 0.2], [0.8, 0.3], [0.2, 0.4]])
    a = per_finding_auc(y, p)
    assert a[0] == 1.0 and np.isnan(a[1])
    assert macro_auc(y, p) == 1.0


def test_bootstrap_of_identical_predictions_is_zero():
    rng = np.random.default_rng(1)
    y = labels(rng)
    p = rng.random(y.shape)
    r = paired_bootstrap(y, {"a": p, "b": p.copy()}, "a", n_boot=200)
    assert r["b"]["mean"] == 0.0 and r["b"]["p_better"] == 0.0


def test_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(2)
    y = labels(rng)
    noise = rng.random(y.shape)
    r = paired_bootstrap(y, {"weak": noise, "strong": y + 0.3 * noise}, "weak", n_boot=200)
    assert r["strong"]["p_better"] > 0.99


def test_split_half_identical_candidates_changes_nothing():
    rng = np.random.default_rng(3)
    y = labels(rng)
    p = rng.random(y.shape)
    r = split_half(y, {j: [p[:, j], p[:, j].copy()] for j in range(12)}, n_splits=50)
    assert r["mean"] == 0.0 and r["p_helps"] == 0.0 and r["p_hurts"] == 0.0


def test_split_half_pure_noise_candidates_do_not_generalise():
    rng = np.random.default_rng(4)
    y = labels(rng)
    base = y + rng.random(y.shape)          # informative baseline
    cands = {j: [base[:, j]] + [rng.random(len(y)) for _ in range(4)] for j in range(12)}
    r = split_half(y, cands, n_splits=100)
    assert r["mean"] <= 0.0


def test_split_half_real_signal_generalises():
    rng = np.random.default_rng(5)
    y = labels(rng)
    weak = rng.random(y.shape)
    cands = {j: [weak[:, j], y[:, j] + 0.5 * weak[:, j]] for j in range(12)}
    r = split_half(y, cands, n_splits=100)
    assert r["mean"] > 0.1 and r["p_helps"] > 0.95
