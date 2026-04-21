"""Tests for services/portfolio_analytics math — no DB, synthetic series."""

import math

from services.portfolio_analytics import (
    _log_returns, _corr, _covariance, _stdev, _aligned_returns,
    _fx_for_month, _percentile, _max_drawdown,
)


def test_log_returns_simple():
    series = {(2024, 1): 100.0, (2024, 2): 110.0, (2024, 3): 99.0}
    rets = _log_returns(series)
    assert set(rets.keys()) == {(2024, 2), (2024, 3)}
    assert math.isclose(rets[(2024, 2)], math.log(1.10), rel_tol=1e-9)
    assert math.isclose(rets[(2024, 3)], math.log(99 / 110), rel_tol=1e-9)


def test_log_returns_skips_non_positive():
    series = {(2024, 1): 100.0, (2024, 2): 0.0, (2024, 3): 50.0}
    rets = _log_returns(series)
    # The 2024-02 return is invalid (log of 0) so both (2) and (3) drop.
    assert (2024, 2) not in rets
    assert (2024, 3) not in rets


def test_log_returns_empty_if_one_point():
    assert _log_returns({(2024, 1): 100.0}) == {}
    assert _log_returns({}) == {}


def test_stdev_and_cov_hand_computed():
    # Returns for two perfectly-correlated series; correlation should be 1.
    a = [0.01, 0.02, -0.01, 0.03]
    b = [2 * x for x in a]                # same shape, scaled
    assert math.isclose(_corr(a, b), 1.0, rel_tol=1e-9)
    # Variance of a: mean=0.0125, squared deviations / (n-1)
    mean_a = sum(a) / len(a)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    assert math.isclose(_covariance(a, a), var_a, rel_tol=1e-9)
    assert math.isclose(_stdev(a), math.sqrt(var_a), rel_tol=1e-9)


def test_corr_anti_correlated():
    a = [0.01, 0.02, -0.01, 0.03]
    b = [-x for x in a]
    assert math.isclose(_corr(a, b), -1.0, rel_tol=1e-9)


def test_corr_independent_small():
    # Uncorrelated-ish series
    a = [0.01, -0.02, 0.00, 0.03, -0.01]
    b = [0.02, 0.01, -0.01, -0.02, 0.02]
    c = _corr(a, b)
    # Just verify it's in [-1, 1] and finite.
    assert -1.0 <= c <= 1.0


def test_aligned_returns_drops_low_coverage():
    # ticker A has 10 obs, ticker B has 3 → B dropped when min_months=5
    rets = {
        "A": {(2024, m): 0.01 for m in range(1, 11)},
        "B": {(2024, m): 0.02 for m in range(1, 4)},
    }
    kept, months, matrix = _aligned_returns(rets, min_months=5)
    assert kept == ["A"] or ("A" in kept and "B" not in kept)
    # Single ticker → no alignment possible
    assert len(matrix) == 0 or len(matrix) == 1


def test_aligned_returns_picks_overlap():
    rets = {
        "A": {(2024, m): 0.01 for m in range(1, 13)},      # Jan-Dec
        "B": {(2024, m): 0.02 for m in range(4, 13)},      # Apr-Dec
    }
    kept, months, matrix = _aligned_returns(rets, min_months=6)
    assert set(kept) == {"A", "B"}
    assert len(months) == 9        # Apr..Dec
    assert len(matrix) == 2
    assert len(matrix[0]) == 9


def test_fx_carry_forward():
    fx = {"GBP": {(2024, 1): 1.27, (2024, 3): 1.26}}
    # Exact hit
    assert _fx_for_month(fx, "GBP", (2024, 1)) == 1.27
    # Carry-forward from Jan through Feb (no Feb row)
    assert _fx_for_month(fx, "GBP", (2024, 2)) == 1.27
    # Exact hit again
    assert _fx_for_month(fx, "GBP", (2024, 3)) == 1.26
    # Forward extrapolation — uses latest <= target
    assert _fx_for_month(fx, "GBP", (2024, 6)) == 1.26
    # Before any data → None
    assert _fx_for_month(fx, "GBP", (2023, 12)) is None
    # USD is identity
    assert _fx_for_month(fx, "USD", (2024, 1)) == 1.0
    # Unknown currency → None
    assert _fx_for_month(fx, "XYZ", (2024, 1)) is None


def test_percentile_matches_numpy_linear():
    # Hand-computed: for [1, 2, 3, 4, 5] the 25th percentile with
    # linear interpolation is 2.0, 50th is 3.0, 90th is 4.6.
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert math.isclose(_percentile(xs, 25), 2.0, abs_tol=1e-9)
    assert math.isclose(_percentile(xs, 50), 3.0, abs_tol=1e-9)
    assert math.isclose(_percentile(xs, 90), 4.6, abs_tol=1e-9)
    assert math.isclose(_percentile(xs, 0),   1.0, abs_tol=1e-9)
    assert math.isclose(_percentile(xs, 100), 5.0, abs_tol=1e-9)


def test_percentile_single_value_and_empty():
    assert _percentile([], 50) == 0.0
    assert _percentile([42.0], 10) == 42.0
    assert _percentile([42.0], 99) == 42.0


def test_max_drawdown_flat_series_is_zero():
    # All zero returns → no drawdown.
    assert _max_drawdown([0, 0, 0, 0]) == 0.0
    # Pure rally → no drawdown.
    assert _max_drawdown([0.01, 0.02, 0.03]) == 0.0


def test_max_drawdown_v_shape():
    # Log returns: +20%, −30%, +15% → cumulative path:
    #   0 → 0.20 → −0.10 → 0.05, peak was 0.20, trough 0.20−(−0.10) = 0.30
    # Expected drawdown = 1 - e^(-0.30) ≈ 0.2592
    dd = _max_drawdown([0.20, -0.30, 0.15])
    assert 0.25 < dd < 0.27, f"unexpected max_dd={dd}"


def test_max_drawdown_late_trough():
    # Rally then crash: +0.1, +0.1, −0.5 → peak 0.2 → trough at end gives
    # drawdown of 0.5 log → 1 - e^(-0.5) ≈ 0.393
    dd = _max_drawdown([0.1, 0.1, -0.5])
    assert 0.38 < dd < 0.41
