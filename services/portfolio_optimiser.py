"""
Server-side portfolio optimiser — Phase D upgrade from the client-side
projected-gradient implementation.

Solves long-only mean-variance, fractional Kelly, and historical CVaR
using scipy.optimize.minimize (SLSQP) for QPs and scipy.optimize.linprog
for CVaR (Rockafellar–Uryasev linear programme).

All weights are decimals (0..1). Inputs μ are expected returns as
decimals (0.10 == +10%). Σ is the annualised covariance matrix in the
SAME unit² as μ.

Constraints are declared via the Constraints dataclass and apply
uniformly across methods:
  - 0 ≤ wᵢ ≤ max_position
  - sum(w) ≤ 1                              (cash held when slack)
  - per-sector cap:   Σ_{i∈sector_s} wᵢ ≤ cap_s
  - per-country cap:  Σ_{i∈country_c} wᵢ ≤ cap_c

The efficient frontier helper sweeps λ on a log scale and keeps the
Pareto upper-hull (each point dominates lower-vol/lower-return ones).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import minimize, linprog


# ─────────────────────────────────────────────────────────────────
# Constraints
# ─────────────────────────────────────────────────────────────────
@dataclass
class Constraints:
    max_position: float = 0.10                # 0..1 per-name cap
    sum_to_one: bool = True                   # sum(w) <= 1 (cash slack)
    long_only: bool = True                    # wᵢ ≥ 0
    sector_caps: dict = field(default_factory=dict)   # {sector_str: cap}
    country_caps: dict = field(default_factory=dict)  # {country_str: cap}

    def per_position_bounds(self, n: int) -> list[tuple[float, float]]:
        lo = 0.0 if self.long_only else -self.max_position
        return [(lo, self.max_position)] * n


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def _build_scipy_constraints(
    n: int,
    sectors: list[str],
    countries: list[str],
    constraints: Constraints,
) -> list[dict]:
    """Convert our Constraints object into scipy's list-of-dicts format
    (each dict has keys 'type' ∈ {'eq','ineq'}, 'fun', and optional 'jac').
    All inequalities are written as fun(w) ≥ 0 per scipy convention."""
    cons: list[dict] = []

    if constraints.sum_to_one:
        # 1 - sum(w) ≥ 0  →  sum(w) ≤ 1
        cons.append({
            "type": "ineq",
            "fun": lambda w: 1.0 - np.sum(w),
            "jac": lambda w: -np.ones_like(w),
        })

    # Sector caps
    for sec, cap in (constraints.sector_caps or {}).items():
        idxs = np.array([i for i, s in enumerate(sectors) if s == sec], dtype=int)
        if idxs.size == 0:
            continue
        # cap - sum(w[idxs]) ≥ 0
        def _make(idxs, cap):
            mask = np.zeros(n)
            mask[idxs] = 1.0
            return (
                lambda w, mask=mask, cap=cap: cap - float(mask @ w),
                lambda w, mask=mask: -mask,
            )
        f, j = _make(idxs, cap)
        cons.append({"type": "ineq", "fun": f, "jac": j})

    # Country caps
    for cty, cap in (constraints.country_caps or {}).items():
        idxs = np.array([i for i, c in enumerate(countries) if c == cty], dtype=int)
        if idxs.size == 0:
            continue
        def _make(idxs, cap):
            mask = np.zeros(n)
            mask[idxs] = 1.0
            return (
                lambda w, mask=mask, cap=cap: cap - float(mask @ w),
                lambda w, mask=mask: -mask,
            )
        f, j = _make(idxs, cap)
        cons.append({"type": "ineq", "fun": f, "jac": j})

    return cons


def _initial_guess(n: int, max_position: float) -> np.ndarray:
    """Equal-weight starting point, capped to feasibility."""
    if n <= 0:
        return np.array([])
    w0 = np.ones(n) / n
    cap = max_position
    if w0[0] > cap:
        w0 = np.ones(n) * cap
    # If sum > 1 after capping at cap (shouldn't happen if cap*n>=1), scale.
    s = w0.sum()
    if s > 1.0:
        w0 = w0 / s
    return w0


def _result_to_dict(res, tickers: list[str]) -> dict:
    """Standardise the SLSQP/linprog result envelope."""
    success = bool(getattr(res, "success", True))
    weights_arr = np.asarray(res.x, dtype=float)
    weights_arr = np.clip(weights_arr, 0.0, None)         # numerical sliver
    w_dict = {t: float(weights_arr[i] * 100.0) for i, t in enumerate(tickers)}
    invested = float(weights_arr.sum() * 100.0)
    return {
        "weights": w_dict,
        "weights_decimal": [float(x) for x in weights_arr],
        "invested_pct": invested,
        "cash_pct": max(0.0, 100.0 - invested),
        "success": success,
        "status": int(getattr(res, "status", 0)),
        "message": str(getattr(res, "message", "")),
        "iterations": int(getattr(res, "nit", getattr(res, "nit", 0)) or 0),
    }


# ─────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────
def solve_mv(
    tickers: list[str],
    mus: list[float],
    cov: list[list[float]],
    sectors: list[str],
    countries: list[str],
    constraints: Constraints,
    lam: float = 5.0,
    max_iter: int = 200,
) -> dict:
    """Markowitz mean-variance:  max μᵀw - (λ/2) wᵀΣw  subject to constraints.
    Returns the same envelope as the other solvers (see _result_to_dict)."""
    n = len(tickers)
    if n == 0:
        return {"weights": {}, "weights_decimal": [], "invested_pct": 0,
                "cash_pct": 100, "success": False, "status": -1,
                "message": "no tickers", "iterations": 0}

    mus_v = np.asarray(mus, dtype=float)
    cov_m = np.asarray(cov, dtype=float)

    def obj(w: np.ndarray) -> float:
        return float(-(mus_v @ w) + 0.5 * lam * (w @ cov_m @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        return -mus_v + lam * (cov_m @ w)

    bounds = constraints.per_position_bounds(n)
    cons = _build_scipy_constraints(n, sectors, countries, constraints)
    w0 = _initial_guess(n, constraints.max_position)

    res = minimize(
        obj, w0, jac=grad, bounds=bounds, constraints=cons,
        method="SLSQP",
        options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
    )
    out = _result_to_dict(res, tickers)
    out["method"] = "mv"
    out["lambda"] = lam
    out["objective"] = float(-res.fun)            # back to "utility" form (positive)
    return out


def solve_kelly(
    tickers: list[str],
    mus: list[float],
    cov: list[list[float]],
    sectors: list[str],
    countries: list[str],
    constraints: Constraints,
    kelly_fraction: float = 0.5,
    max_iter: int = 200,
) -> dict:
    """Fractional Kelly via constrained MV. The full-Kelly portfolio is
    Σ⁻¹μ; under quadratic utility, fractional Kelly with fraction f is
    equivalent to MV with risk-aversion λ = 1/f. Solve as a MV with
    that λ, then apply caps via the same SLSQP machinery."""
    if kelly_fraction <= 0:
        return solve_mv(tickers, mus, cov, sectors, countries, constraints, lam=1e6)
    lam = 1.0 / max(kelly_fraction, 1e-3)
    out = solve_mv(tickers, mus, cov, sectors, countries, constraints, lam=lam, max_iter=max_iter)
    out["method"] = "kelly"
    out["kelly_fraction"] = kelly_fraction
    return out


def solve_cvar(
    tickers: list[str],
    returns_matrix: list[list[float]],     # T x N (T=time, N=tickers)
    sectors: list[str],
    countries: list[str],
    constraints: Constraints,
    alpha: float = 0.95,
    target_return: float | None = None,    # min portfolio expected return
    force_full_investment: bool = False,   # add sum(w)=1 to avoid all-cash optimum
) -> dict:
    """Minimise historical CVaR using the Rockafellar–Uryasev LP.

    Variables: w (N), ν (1, the VaR level), u (T, tail slacks)
    Objective: ν + (1 / (T · (1 - α))) · Σ uₜ
    s.t.       uₜ ≥ -(Rₜ · w) - ν   ∀t
               uₜ ≥ 0
               wᵢ ≥ 0, wᵢ ≤ max_position
               sum(w) ≤ 1
               sector / country caps
               (optional) μᵀw ≥ target_return
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2D (T rows, N cols)")
    T, n = R.shape
    if n != len(tickers):
        raise ValueError(f"shape mismatch: returns_matrix has {n} cols but {len(tickers)} tickers")
    if T < 12:
        return {"weights": {}, "weights_decimal": [], "invested_pct": 0,
                "cash_pct": 100, "success": False, "status": -1,
                "message": f"need ≥12 monthly observations, got {T}", "iterations": 0,
                "method": "cvar"}

    # Variable layout: [w_0..w_{n-1}, nu, u_0..u_{T-1}]   length = n + 1 + T
    # linprog needs A_ub @ x <= b_ub, bounds, c.
    # Objective: minimise nu + (1/(T(1-α))) Σ uₜ
    c = np.zeros(n + 1 + T)
    c[n] = 1.0                                  # nu
    c[n + 1:] = 1.0 / (T * (1.0 - alpha))        # u coeffs

    # Tail constraints: -R_t · w - nu - u_t ≤ 0   (for each t)
    # Rewrite: -R[t,:] · w + (-1)·nu + (-1)·u_t ≤ 0
    A_tail = np.zeros((T, n + 1 + T))
    A_tail[:, :n] = -R                           # -R · w
    A_tail[:, n] = -1.0                           # -nu
    for t in range(T):
        A_tail[t, n + 1 + t] = -1.0               # -u_t
    b_tail = np.zeros(T)

    A_ub_rows = [A_tail]
    b_ub_rows = [b_tail]

    # Sum(w) <= 1
    if constraints.sum_to_one:
        row = np.zeros(n + 1 + T)
        row[:n] = 1.0
        A_ub_rows.append(row.reshape(1, -1))
        b_ub_rows.append(np.array([1.0]))

    # Sector caps
    for sec, cap in (constraints.sector_caps or {}).items():
        idxs = [i for i, s in enumerate(sectors) if s == sec]
        if not idxs:
            continue
        row = np.zeros(n + 1 + T)
        for i in idxs:
            row[i] = 1.0
        A_ub_rows.append(row.reshape(1, -1))
        b_ub_rows.append(np.array([cap]))

    # Country caps
    for cty, cap in (constraints.country_caps or {}).items():
        idxs = [i for i, c_ in enumerate(countries) if c_ == cty]
        if not idxs:
            continue
        row = np.zeros(n + 1 + T)
        for i in idxs:
            row[i] = 1.0
        A_ub_rows.append(row.reshape(1, -1))
        b_ub_rows.append(np.array([cap]))

    A_ub = np.vstack(A_ub_rows)
    b_ub = np.concatenate(b_ub_rows)

    A_eq = None
    b_eq = None
    # Optional: force sum(w) = 1 so CVaR can't trivially pick cash. This is
    # the standard CVaR-portfolio formulation in Rockafellar-Uryasev.
    if force_full_investment:
        eq_row = np.zeros(n + 1 + T)
        eq_row[:n] = 1.0
        A_eq = eq_row.reshape(1, -1)
        b_eq = np.array([1.0])
    # Optional minimum expected return (μᵀw ≥ target_return  →  -μᵀw ≤ -target)
    if target_return is not None:
        # we'd append to A_ub but we don't compute μ here — leave as future hook
        pass

    # Bounds
    bounds = []
    bounds.extend([(0.0, constraints.max_position) for _ in range(n)])   # w
    bounds.append((None, None))                                          # nu free
    bounds.extend([(0.0, None) for _ in range(T)])                       # u >= 0

    res = linprog(
        c=c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    weights = res.x[:n] if res.x is not None else np.zeros(n)
    weights = np.clip(weights, 0.0, None)
    nu = float(res.x[n]) if res.x is not None else 0.0
    obj = float(res.fun) if res.fun is not None else 0.0

    return {
        "weights": {t: float(weights[i] * 100.0) for i, t in enumerate(tickers)},
        "weights_decimal": [float(x) for x in weights],
        "invested_pct": float(weights.sum() * 100.0),
        "cash_pct": max(0.0, 100.0 - float(weights.sum() * 100.0)),
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message or ""),
        "iterations": int(getattr(res, "nit", 0) or 0),
        "method": "cvar",
        "alpha": alpha,
        "var_at_optimum": -nu,        # VaR (positive loss) at optimum
        "cvar_at_optimum": obj,        # CVaR loss (positive)
    }


def efficient_frontier(
    tickers: list[str],
    mus: list[float],
    cov: list[list[float]],
    sectors: list[str],
    countries: list[str],
    constraints: Constraints,
    n_points: int = 25,
    lam_min: float = 0.25,
    lam_max: float = 40.0,
) -> list[dict]:
    """Sweep λ on a log scale, run solve_mv at each, keep the upper Pareto
    frontier. Returns list of {lambda, vol, exp_ret, sharpe, weights, cash_pct}."""
    if n_points < 2 or len(tickers) < 2:
        return []
    mus_v = np.asarray(mus, dtype=float)
    cov_m = np.asarray(cov, dtype=float)
    points = []
    log_min, log_max = np.log(lam_min), np.log(lam_max)
    for k in range(n_points):
        lam = float(np.exp(log_min + (log_max - log_min) * k / (n_points - 1)))
        out = solve_mv(tickers, mus, cov, sectors, countries, constraints, lam=lam)
        if not out["success"]:
            continue
        w = np.asarray(out["weights_decimal"], dtype=float)
        if w.sum() <= 1e-6:
            continue
        port_ret = float(mus_v @ w)
        port_var = float(w @ cov_m @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        sharpe = (port_ret / port_vol) if port_vol > 0 else 0.0
        points.append({
            "lambda": lam,
            "vol": port_vol,
            "exp_ret": port_ret,
            "sharpe": sharpe,
            "weights": out["weights"],
            "cash_pct": out["cash_pct"],
            "iterations": out["iterations"],
        })
    # Pareto upper hull: sort by vol asc, keep monotonically rising ret.
    points.sort(key=lambda p: p["vol"])
    frontier = []
    best_ret = -float("inf")
    for p in points:
        if p["exp_ret"] > best_ret:
            frontier.append(p)
            best_ret = p["exp_ret"]
    return frontier
