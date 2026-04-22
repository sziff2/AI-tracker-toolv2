"""Tests for services/portfolio_optimiser — synthetic inputs, known answers."""

import math

import pytest

# scipy is a new dep — let the suite pass if it's not installed (e.g. local
# dev box without numerical stack). The test will be skipped instead of erroring.
scipy = pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from services.portfolio_optimiser import (
    Constraints, solve_mv, solve_kelly, solve_cvar, efficient_frontier,
)


# ─────────────────────────────────────────────────────────────────
# Toy 3-asset universe
# ─────────────────────────────────────────────────────────────────
TICKERS = ["A", "B", "C"]
SECTORS = ["Tech", "Tech", "Energy"]
COUNTRIES = ["US", "EU", "US"]

# A & B: high return, high vol; C: low return, low vol, low corr.
MUS = [0.12, 0.10, 0.06]
SDS = [0.20, 0.18, 0.10]
CORR = [
    [1.00, 0.60, 0.10],
    [0.60, 1.00, 0.05],
    [0.10, 0.05, 1.00],
]
COV = [[CORR[i][j] * SDS[i] * SDS[j] for j in range(3)] for i in range(3)]


def _wsum(weights_decimal):
    return sum(weights_decimal)


def test_mv_respects_max_position_cap():
    """With max_position=0.10, no single weight should exceed 0.10."""
    out = solve_mv(TICKERS, MUS, COV, SECTORS, COUNTRIES,
                   Constraints(max_position=0.10), lam=2.0)
    assert out["success"], out["message"]
    for w in out["weights_decimal"]:
        assert w <= 0.10 + 1e-6, f"weight {w} exceeds cap 0.10"
    assert _wsum(out["weights_decimal"]) <= 1.0 + 1e-6


def test_mv_unconstrained_picks_high_return_with_low_lambda():
    """Low λ → more weight on the highest-return asset (A)."""
    out = solve_mv(TICKERS, MUS, COV, SECTORS, COUNTRIES,
                   Constraints(max_position=1.0), lam=0.5)
    w = out["weights_decimal"]
    assert w[0] >= w[2], f"A should outweigh C at low λ; got {w}"


def test_mv_high_lambda_diversifies_to_low_vol():
    """High λ → minimum variance corner, weight on low-vol C dominates."""
    out = solve_mv(TICKERS, MUS, COV, SECTORS, COUNTRIES,
                   Constraints(max_position=1.0), lam=50.0)
    w = out["weights_decimal"]
    assert w[2] >= w[0], f"C should outweigh A at high λ; got {w}"


def test_mv_sector_cap_enforced():
    """Sector cap of 0.15 on Tech (A+B) limits A+B to 0.15 total."""
    out = solve_mv(TICKERS, MUS, COV, SECTORS, COUNTRIES,
                   Constraints(max_position=1.0, sector_caps={"Tech": 0.15}), lam=0.5)
    w = out["weights_decimal"]
    tech = w[0] + w[1]
    assert tech <= 0.15 + 1e-6, f"Tech sum {tech} exceeds 0.15"


def test_mv_country_cap_enforced():
    """US cap of 0.20 limits A + C to 0.20."""
    out = solve_mv(TICKERS, MUS, COV, SECTORS, COUNTRIES,
                   Constraints(max_position=1.0, country_caps={"US": 0.20}), lam=2.0)
    w = out["weights_decimal"]
    us = w[0] + w[2]
    assert us <= 0.20 + 1e-6, f"US sum {us} exceeds 0.20"


def test_kelly_higher_fraction_takes_more_risk():
    """fraction=1.0 should put more in the high-return assets than fraction=0.25."""
    high = solve_kelly(TICKERS, MUS, COV, SECTORS, COUNTRIES,
                       Constraints(max_position=1.0), kelly_fraction=1.0)
    low  = solve_kelly(TICKERS, MUS, COV, SECTORS, COUNTRIES,
                       Constraints(max_position=1.0), kelly_fraction=0.25)
    # High-return tilt = w_A + w_B
    high_tilt = high["weights_decimal"][0] + high["weights_decimal"][1]
    low_tilt  = low["weights_decimal"][0]  + low["weights_decimal"][1]
    assert high_tilt >= low_tilt, f"full-Kelly should over-weight risky assets vs quarter-Kelly: {high_tilt} vs {low_tilt}"


def test_cvar_runs_on_synthetic_returns():
    """CVaR LP returns valid weights given a small synthetic returns matrix."""
    rng = np.random.default_rng(42)
    T = 60
    R = rng.multivariate_normal(mean=MUS, cov=np.array(COV) / 12.0, size=T)
    out = solve_cvar(TICKERS, R.tolist(), SECTORS, COUNTRIES,
                     Constraints(max_position=0.40), alpha=0.95)
    assert out["success"], out["message"]
    for w in out["weights_decimal"]:
        assert 0 <= w <= 0.40 + 1e-6
    assert _wsum(out["weights_decimal"]) <= 1.0 + 1e-6
    # CVaR objective is finite (sign depends on whether tail is loss or gain).
    assert math.isfinite(out["cvar_at_optimum"])


def test_cvar_picks_cash_when_all_mean_returns_negative():
    """Sanity-check rationality: with all-negative expected returns and
    sum(w) ≤ 1 (cash allowed), the optimal CVaR portfolio should be all
    cash — minimum loss = zero loss."""
    rng = np.random.default_rng(7)
    T = 80
    bad_mus = [-0.20, -0.15, -0.10]
    R = rng.multivariate_normal(mean=bad_mus, cov=np.array(COV) / 12.0, size=T)
    out = solve_cvar(TICKERS, R.tolist(), SECTORS, COUNTRIES,
                     Constraints(max_position=0.50), alpha=0.95)
    assert out["success"]
    invested = sum(out["weights_decimal"])
    assert invested < 0.05, f"expected near-zero investment with all-negative μ; got {invested}"


def test_efficient_frontier_returns_pareto_upper_hull():
    pts = efficient_frontier(
        TICKERS, MUS, COV, SECTORS, COUNTRIES,
        Constraints(max_position=1.0), n_points=15,
    )
    assert len(pts) >= 3
    # Sorted by vol ascending, expected return must be non-decreasing
    vols = [p["vol"] for p in pts]
    rets = [p["exp_ret"] for p in pts]
    assert vols == sorted(vols)
    assert rets == sorted(rets), f"frontier not monotonic in return: {rets}"
    # Sharpe ratios should be finite
    for p in pts:
        assert math.isfinite(p["sharpe"])


def test_mv_long_only_no_negative_weights():
    out = solve_mv(TICKERS, MUS, COV, SECTORS, COUNTRIES,
                   Constraints(max_position=0.5, long_only=True), lam=2.0)
    assert all(w >= -1e-9 for w in out["weights_decimal"])
