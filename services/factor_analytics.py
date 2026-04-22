"""
Factor analytics — Phase 2b stress tests via per-holding factor exposures.

Eight factor proxies, all USD ETFs with 10+ years of Yahoo history:

    equity        SPY     S&P 500
    rates         TLT     20+ Year Treasury (long duration)
    oil           USO     United States Oil Fund (Brent / WTI exposure)
    usd           UUP     Invesco Bullish USD ETF
    credit        HYG     iShares iBoxx HY Corp Bond
    growth        IWF     Russell 1000 Growth
    value         IWD     Russell 1000 Value
    quality       QUAL    iShares MSCI USA Quality Factor
    momentum      MTUM    iShares MSCI USA Momentum Factor

For each holding we estimate β to each factor via OLS on monthly USD log
returns over a 36-month rolling window. Betas live in the
`holding_factor_exposures` table and are refreshed monthly by a Celery
Beat job after the price feed runs.

Shock interpretation: a shock vector is {factor_key: pct_move} where
each pct_move is the monthly return of the factor's proxy ETF (NOT a
basis-point move on yields). Portfolio impact for shock vector s:

    portfolio_return = Σᵢ wᵢ × Σⱼ βᵢⱼ × sⱼ
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import UUID

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# ─────────────────────────────────────────────────────────────────
# Factor universe
# ─────────────────────────────────────────────────────────────────
FACTOR_PROXIES: dict[str, dict] = {
    "equity":   {"ticker": "SPY US",  "yahoo": "SPY",  "label": "Equity (SPY)",
                 "description": "S&P 500. Captures broad market direction. β≈1 typical."},
    "rates":    {"ticker": "TLT US",  "yahoo": "TLT",  "label": "Rates innov. (TLT)",
                 "description": "Long-duration Treasury move ORTHOGONAL to equity. Positive shock = yields fell with equity flat."},
    "oil":      {"ticker": "USO US",  "yahoo": "USO",  "label": "Oil innov. (USO)",
                 "description": "WTI move orthogonal to equity & rates. Positive = oil rally beyond what market would predict."},
    "usd":      {"ticker": "UUP US",  "yahoo": "UUP",  "label": "USD innov. (UUP)",
                 "description": "Dollar strength orthogonal to equity / rates / oil."},
    "credit":   {"ticker": "HYG US",  "yahoo": "HYG",  "label": "Credit innov. (HYG)",
                 "description": "HY credit spread move orthogonal to earlier factors. Positive = spreads tightened."},
    "value":    {"ticker": "IWD US",  "yahoo": "IWD",  "label": "Value innov. (IWD)",
                 "description": "Russell 1000 Value innovation. Positive = value outperformed beyond what equity explains."},
    "growth":   {"ticker": "IWF US",  "yahoo": "IWF",  "label": "Growth innov. (IWF)",
                 "description": "Russell 1000 Growth innovation, orthogonal to equity + value."},
    "quality":  {"ticker": "QUAL US", "yahoo": "QUAL", "label": "Quality innov. (QUAL)",
                 "description": "Quality factor innovation, orthogonal to all earlier factors."},
    "momentum": {"ticker": "MTUM US", "yahoo": "MTUM", "label": "Momentum innov. (MTUM)",
                 "description": "Momentum factor innovation, orthogonal to all earlier factors."},
}

FACTOR_KEYS = list(FACTOR_PROXIES.keys())
FACTOR_TICKERS = [v["ticker"] for v in FACTOR_PROXIES.values()]


def list_factor_proxies() -> list[dict]:
    """Public catalogue for the UI — keys + labels + descriptions."""
    return [{"key": k, **v} for k, v in FACTOR_PROXIES.items()]


# ─────────────────────────────────────────────────────────────────
# Beta estimation (rolling OLS)
# ─────────────────────────────────────────────────────────────────
FACTOR_PRIORITY = ["equity", "rates", "oil", "usd", "credit", "value", "growth", "quality", "momentum"]


def _orthogonalise(raw: dict[str, list[float]]) -> dict[str, list[float]]:
    """Sequentially residualise factors in priority order. Each factor
    after equity is replaced by the OLS residual of regressing it on
    all earlier factors. Net result: the equity beta represents true
    market sensitivity (~1 for diversified stocks) and each other beta
    represents incremental sensitivity to that factor independent of
    everything earlier — the shocks compose without double-counting."""
    keys_in_order = [k for k in FACTOR_PRIORITY if k in raw]
    if not keys_in_order:
        return {}
    n = len(next(iter(raw.values())))
    Z = np.zeros((n, len(keys_in_order)))
    out: dict[str, list[float]] = {}
    for col, key in enumerate(keys_in_order):
        y = np.asarray(raw[key], dtype=float)
        if col == 0:
            Z[:, col] = y                              # equity stays raw
        else:
            X = np.column_stack([np.ones(n), Z[:, :col]])
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                Z[:, col] = y - X @ beta                # residual
            except np.linalg.LinAlgError:
                Z[:, col] = y - y.mean()
        out[key] = Z[:, col].tolist()
    return out


async def _factor_returns_aligned(
    db: AsyncSession,
    window_months: int,
) -> tuple[list[tuple[int, int]], dict[str, list[float]]]:
    """Return (months_used, {factor_key: [returns]}) where every factor
    has a return at every listed month. Returned series are ORTHOGONAL —
    each factor after equity is the residual after regressing on all
    prior factors. This makes betas additive and shocks composable."""
    from services.portfolio_analytics import (
        _load_fx_by_month, _load_monthly_series_usd, _log_returns,
    )
    fx = await _load_fx_by_month(db)
    factor_returns: dict[str, dict[tuple[int, int], float]] = {}
    for key, meta in FACTOR_PROXIES.items():
        rs = await db.execute(
            text("SELECT id FROM companies WHERE ticker = :t"),
            {"t": meta["ticker"]},
        )
        cid = rs.scalar_one_or_none()
        if cid is None:
            continue
        series = await _load_monthly_series_usd(db, cid, window_months, fx)
        if len(series) < 13:
            continue
        rets = _log_returns(series)
        if len(rets) < 12:
            continue
        factor_returns[key] = rets
    if len(factor_returns) < 2:
        return [], {}
    common = sorted(set.intersection(*(set(r.keys()) for r in factor_returns.values())))
    raw_aligned = {k: [factor_returns[k][m] for m in common] for k in factor_returns}
    return common, _orthogonalise(raw_aligned)


async def _holding_returns(
    db: AsyncSession,
    company_id,
    months: list[tuple[int, int]],
) -> list[float] | None:
    """Return the holding's USD monthly log returns for exactly the given
    month list, or None if it doesn't have a return for every month."""
    from services.portfolio_analytics import (
        _load_fx_by_month, _load_monthly_series_usd, _log_returns,
    )
    if not months:
        return None
    fx = await _load_fx_by_month(db)
    series = await _load_monthly_series_usd(db, company_id, len(months) + 24, fx)
    rets = _log_returns(series)
    out = []
    for m in months:
        if m not in rets:
            return None
        out.append(rets[m])
    return out


async def estimate_holding_betas(
    db: AsyncSession,
    company_id,
    window_months: int = 36,
) -> dict | None:
    """Run a single OLS regression of holding monthly USD log returns on
    the 9 factor return series over the trailing `window_months`. Returns
    {factor_key: beta}, R², n_months, and the regression intercept (alpha).
    Returns None if data is insufficient."""
    months_all, factor_rets = await _factor_returns_aligned(db, window_months + 6)
    if len(months_all) < window_months:
        return None
    months = months_all[-window_months:]
    factor_rets = {k: v[-window_months:] for k, v in factor_rets.items()}

    h_rets = await _holding_returns(db, company_id, months)
    if h_rets is None or len(h_rets) < window_months // 2:
        return None

    # Build design matrix (T × (k+1)) — k factors plus intercept
    factor_keys_present = [k for k in FACTOR_KEYS if k in factor_rets]
    X_factors = np.column_stack([factor_rets[k] for k in factor_keys_present])
    if X_factors.shape[0] < 24 or X_factors.shape[1] < 4:
        return None
    # Prepend an intercept column of 1s.
    X = np.column_stack([np.ones(X_factors.shape[0]), X_factors])
    y = np.array(h_rets, dtype=float)

    # Plain OLS — factors are orthogonalised upstream so multicollinearity
    # is not an issue. β = (XᵀX)⁻¹ Xᵀy.
    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        params = XtX_inv @ X.T @ y
        y_pred = X @ params
        residuals = y - y_pred
        ss_res = float((residuals ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        n, k_total = X.shape
        if n > k_total:
            sigma2 = ss_res / (n - k_total)
            se = np.sqrt(np.maximum(np.diag(sigma2 * XtX_inv), 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                tstats_arr = np.where(se > 0, params / se, 0.0)
        else:
            tstats_arr = np.zeros_like(params)
    except np.linalg.LinAlgError as exc:
        return {"error": f"singular design matrix: {exc}"}
    except Exception as exc:
        return {"error": str(exc)}

    betas = {k: float(b) for k, b in zip(factor_keys_present, params[1:])}
    tstats = {k: float(t) for k, t in zip(factor_keys_present, tstats_arr[1:])}
    return {
        "betas": betas,
        "alpha": float(params[0]),
        "tstats": tstats,
        "r_squared": float(r_squared),
        "n_months": len(h_rets),
        "as_of_month": f"{months[-1][0]:04d}-{months[-1][1]:02d}",
    }


async def refresh_all_holding_betas(
    db: AsyncSession,
    window_months: int = 36,
) -> dict:
    """Recompute betas for every active holding and write to the
    holding_factor_exposures table. Run by Celery Beat monthly."""
    rs = await db.execute(text(
        "SELECT id, ticker FROM companies WHERE coverage_status = 'active'"
    ))
    companies = [(r.id, r.ticker) for r in rs]
    written = 0
    skipped = []
    for cid, ticker in companies:
        result = await estimate_holding_betas(db, cid, window_months=window_months)
        if not result or "error" in result:
            skipped.append({"ticker": ticker, "reason": (result or {}).get("error", "insufficient data")})
            continue
        await db.execute(text("""
            INSERT INTO holding_factor_exposures
                (company_id, window_months, betas, alpha, tstats, r_squared, n_months, as_of_month, computed_at)
            VALUES
                (:cid, :win, :betas, :alpha, :tstats, :r2, :n, :asof, :now)
            ON CONFLICT (company_id, window_months)
            DO UPDATE SET betas = EXCLUDED.betas, alpha = EXCLUDED.alpha,
                          tstats = EXCLUDED.tstats, r_squared = EXCLUDED.r_squared,
                          n_months = EXCLUDED.n_months, as_of_month = EXCLUDED.as_of_month,
                          computed_at = EXCLUDED.computed_at
        """), {
            "cid": str(cid), "win": window_months,
            "betas": _json(result["betas"]), "alpha": result["alpha"],
            "tstats": _json(result["tstats"]), "r2": result["r_squared"],
            "n": result["n_months"], "asof": result["as_of_month"],
            "now": datetime.now(timezone.utc),
        })
        written += 1
    await db.commit()
    return {"written": written, "skipped": skipped, "window_months": window_months}


def _json(obj):
    """Helper that lets ON CONFLICT DO UPDATE handle JSONB cleanly."""
    import json as _js
    return _js.dumps(obj)


# ─────────────────────────────────────────────────────────────────
# Apply a shock
# ─────────────────────────────────────────────────────────────────
async def apply_factor_shock(
    db: AsyncSession,
    portfolio_id: str,
    shocks: dict[str, float],         # {factor_key: monthly_pct_move (decimal)}
    window_months: int = 36,
) -> dict:
    """Compute the portfolio's expected one-month return under the given
    factor shock vector, using cached betas (or computing them on demand
    for any holding that's missing a row).

        portfolio_return = Σᵢ wᵢ × (αᵢ + Σⱼ βᵢⱼ × sⱼ + uncovered_residual)

    Holdings without enough history are excluded and reported.
    """
    rs = await db.execute(text("""
        SELECT c.id AS cid, c.ticker, c.sector, c.country, h.weight
          FROM portfolio_holdings h
          JOIN companies c ON c.id = h.company_id
         WHERE h.portfolio_id = :pid AND COALESCE(h.weight,0) > 0
         ORDER BY h.weight DESC NULLS LAST
    """), {"pid": portfolio_id})
    holdings = [(r.cid, r.ticker, r.sector, r.country, float(r.weight or 0)) for r in rs]
    if not holdings:
        return {"error": "no holdings"}

    portfolio_return = 0.0
    per_holding = []
    excluded = []
    by_sector: dict[str, float] = {}
    by_country: dict[str, float] = {}

    for cid, ticker, sector, country, weight in holdings:
        rs2 = await db.execute(text("""
            SELECT betas, alpha FROM holding_factor_exposures
             WHERE company_id = :cid AND window_months = :win
        """), {"cid": str(cid), "win": window_months})
        row = rs2.fetchone()
        betas: dict | None = None
        alpha = 0.0
        if row:
            betas = row.betas if isinstance(row.betas, dict) else __import__("json").loads(row.betas or "{}")
            alpha = float(row.alpha or 0)
        else:
            # On-demand fallback so portfolios work before the Celery beta-refresh job runs.
            est = await estimate_holding_betas(db, cid, window_months=window_months)
            if est and "betas" in est:
                betas = est["betas"]
                alpha = est["alpha"]
        if betas is None:
            excluded.append({"ticker": ticker, "reason": "no factor exposures"})
            continue

        impact = sum(betas.get(k, 0.0) * shocks.get(k, 0.0) for k in FACTOR_KEYS)
        # Don't include alpha in shock impact — alpha is the unexplained
        # constant, which would bias every shock by the historical alpha.
        contribution = (weight / 100.0) * impact
        per_holding.append({
            "ticker": ticker,
            "sector": sector,
            "country": country,
            "weight_pct": round(weight, 3),
            "factor_impact_pct": round(impact * 100, 3),
            "contribution_pct": round(contribution * 100, 3),
            "betas": {k: round(betas.get(k, 0.0), 3) for k in FACTOR_KEYS},
        })
        portfolio_return += contribution
        if sector:
            by_sector[sector] = by_sector.get(sector, 0.0) + contribution
        if country:
            by_country[country] = by_country.get(country, 0.0) + contribution

    per_holding.sort(key=lambda x: x["contribution_pct"])

    return {
        "portfolio_id": portfolio_id,
        "window_months": window_months,
        "shocks": shocks,
        "portfolio_return": round(portfolio_return, 6),
        "n_holdings_in_shock": len(per_holding),
        "holdings_excluded": excluded,
        "per_holding": per_holding,
        "by_sector":  {k: round(v, 6) for k, v in sorted(by_sector.items(), key=lambda x: x[1])},
        "by_country": {k: round(v, 6) for k, v in sorted(by_country.items(), key=lambda x: x[1])},
    }
