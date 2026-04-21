"""
Historical stress-test scenarios — Phase 2a of the risk roadmap.

Replays named market episodes against the portfolio's CURRENT weights
using each holding's actual monthly USD log returns during that window.
Holdings without coverage in the window are excluded and reported.

Design:
  - Each scenario is a calendar window (start, end) with a short name
    and description suitable for a PM note.
  - Portfolio return for month m = Σ wᵢ × rᵢ,m over holdings that had
    a trading month m. Holdings missing a month have their weight
    temporarily redistributed across the survivors for that month
    (equivalent to excluding them from the period-specific calc).
  - Cumulative return is compounded via simple-return chaining:
    (1 + R_cum) = Π (1 + R_month).
"""

from __future__ import annotations

import math
from datetime import date
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# ─────────────────────────────────────────────────────────────────
# Preset historical windows
# ─────────────────────────────────────────────────────────────────
# Key → (label, description, start, end)
HISTORICAL_WINDOWS: dict[str, dict] = {
    "gfc_crash": {
        "label": "Global Financial Crisis (crash)",
        "description": "Lehman, AIG, credit-spread blowout, -45% S&P peak-to-trough.",
        "start": "2008-08-01", "end": "2009-03-31",
    },
    "gfc_recovery": {
        "label": "GFC recovery rally",
        "description": "Reflation + QE1. S&P +60% off the March '09 lows.",
        "start": "2009-04-01", "end": "2009-12-31",
    },
    "euro_crisis": {
        "label": "Eurozone debt crisis (Aug–Oct 2011)",
        "description": "Greek default risk, bank contagion, -20% risk-asset drawdown.",
        "start": "2011-08-01", "end": "2011-10-31",
    },
    "taper_tantrum": {
        "label": "Taper tantrum (2013)",
        "description": "Bernanke QE-taper shock — bonds sold, EMs crushed.",
        "start": "2013-05-01", "end": "2013-09-30",
    },
    "energy_crash": {
        "label": "Oil / energy crash (2014–16)",
        "description": "Oil -70%. Energy & industrial cyclicals derated.",
        "start": "2014-06-01", "end": "2016-01-31",
    },
    "q4_2018": {
        "label": "Q4 2018 selloff",
        "description": "Fed hiking into a slowdown. -20% S&P in three months.",
        "start": "2018-10-01", "end": "2018-12-31",
    },
    "covid_crash": {
        "label": "Covid crash",
        "description": "Feb-Mar 2020. Fastest bear in history, -34% S&P in 23 days.",
        "start": "2020-02-01", "end": "2020-03-31",
    },
    "covid_rally": {
        "label": "Post-Covid rally",
        "description": "Massive monetary + fiscal response, +50% S&P off March lows.",
        "start": "2020-04-01", "end": "2020-12-31",
    },
    "rate_shock_2022": {
        "label": "2022 rate shock",
        "description": "Fed hiking from 0→4.5%. Growth derated, -25% S&P peak-to-trough.",
        "start": "2022-01-01", "end": "2022-10-31",
    },
    "svb_2023": {
        "label": "SVB / regional bank crisis",
        "description": "March 2023. Mid-sized US bank failures, KRE -30%.",
        "start": "2023-03-01", "end": "2023-05-31",
    },
    "deepseek_2025": {
        "label": "DeepSeek / AI capex shock",
        "description": "Jan 2025. Cheap open-source AI model triggered hyperscaler capex panic.",
        "start": "2025-01-01", "end": "2025-02-28",
    },
}


def list_presets() -> list[dict]:
    """Public preset list, sorted chronologically, for the UI picker."""
    items = []
    for key, meta in HISTORICAL_WINDOWS.items():
        items.append({"key": key, **meta})
    items.sort(key=lambda x: x["start"])
    return items


# ─────────────────────────────────────────────────────────────────
# Stress replay against current holdings
# ─────────────────────────────────────────────────────────────────
async def _load_holding_monthly_returns_in_window(
    db: AsyncSession,
    company_id,
    start: date,
    end: date,
) -> dict[tuple[int, int], float]:
    """Return {(year, month): simple_monthly_return_usd} across the window.
    Uses price_records + fx_rates (identical path to portfolio_analytics)."""
    from services.portfolio_analytics import (
        _load_fx_by_month, _load_monthly_series_usd, _log_returns,
    )
    # We re-use the 120-month USD-series helper; window clipping happens below.
    fx = await _load_fx_by_month(db)
    # Request a wide window so the series covers the stress period regardless
    # of when it is relative to "today". 300 months back from today = 25yr.
    series = await _load_monthly_series_usd(db, company_id, 300, fx)
    if not series:
        return {}
    # Window returns cover months where start ≤ month ≤ end. To compute
    # the FIRST month's return we also need the previous month's EOM
    # price as a baseline — so include one extra month before start_key.
    start_key = (start.year, start.month)
    # Compute "one month before start_key" for baseline.
    if start_key[1] == 1:
        baseline_key = (start_key[0] - 1, 12)
    else:
        baseline_key = (start_key[0], start_key[1] - 1)
    end_key = (end.year, end.month)
    clipped = {
        ym: v for ym, v in series.items()
        if baseline_key <= ym <= end_key
    }
    if len(clipped) < 2:
        return {}
    # Simple monthly returns: (p_t - p_{t-1}) / p_{t-1}. Only emit returns
    # where `curr` falls inside the window [start_key, end_key].
    ordered = sorted(clipped.keys())
    out: dict[tuple[int, int], float] = {}
    for prev, curr in zip(ordered[:-1], ordered[1:]):
        if curr < start_key:
            continue
        p0, p1 = clipped[prev], clipped[curr]
        if p0 and p0 > 0 and p1 > 0:
            out[curr] = p1 / p0 - 1
    return out


async def compute_historical_stress(
    db: AsyncSession,
    portfolio_id: str | UUID,
    scenario_key: str,
) -> dict:
    """Replay scenario_key against portfolio_id's current weights.

    Response shape (values are decimal simple-return fractions unless noted):
        {
            "scenario_key": "covid_crash",
            "label": "...",
            "description": "...",
            "start": "2020-02-01", "end": "2020-03-31",
            "n_months": 2,
            "months_covered": ["2020-02", "2020-03"],
            "portfolio_return": -0.27,                    # cumulative pct (decimal)
            "portfolio_worst_month": -0.18,
            "portfolio_best_month": 0.04,
            "holdings_included": [ticker, ...],
            "holdings_excluded": [{"ticker": ..., "reason": ...}, ...],
            "per_holding": [
                {"ticker":..., "weight_pct":..., "window_return":..., "contribution":...},
                ...
            ],
            "by_sector": {"Financials": -0.12, ...},
            "by_country": {"US": -0.09, ...},
        }
    """
    if scenario_key not in HISTORICAL_WINDOWS:
        return {"error": f"Unknown scenario '{scenario_key}'"}
    meta = HISTORICAL_WINDOWS[scenario_key]
    start = date.fromisoformat(meta["start"])
    end = date.fromisoformat(meta["end"])

    pid = str(portfolio_id)
    rs = await db.execute(text("""
        SELECT c.id AS cid, c.ticker, c.sector, c.country, COALESCE(h.weight,0) AS weight
          FROM portfolio_holdings h
          JOIN companies c ON c.id = h.company_id
         WHERE h.portfolio_id = :pid AND COALESCE(h.weight,0) > 0
         ORDER BY h.weight DESC
    """), {"pid": pid})
    raw = [(row.cid, row.ticker, row.sector, row.country, float(row.weight or 0))
           for row in rs]

    if not raw:
        return {
            "scenario_key": scenario_key, **meta,
            "portfolio_return": None, "error": "no holdings with weight > 0",
        }

    # Collect per-holding monthly returns + cumulative over the window.
    holding_series: dict[str, dict[tuple[int, int], float]] = {}
    excluded = []
    per_holding = []
    by_sector: dict[str, float] = {}
    by_country: dict[str, float] = {}

    for cid, ticker, sector, country, weight in raw:
        series = await _load_holding_monthly_returns_in_window(db, cid, start, end)
        if not series:
            excluded.append({"ticker": ticker, "reason": "no price history in window"})
            continue
        holding_series[ticker] = series

    if not holding_series:
        return {
            "scenario_key": scenario_key, **meta,
            "portfolio_return": None,
            "holdings_excluded": excluded,
            "error": "no holdings with data in window",
        }

    # Find months where at least one covered holding has data.
    all_months = sorted({m for rets in holding_series.values() for m in rets})
    if not all_months:
        return {
            "scenario_key": scenario_key, **meta,
            "portfolio_return": None,
            "holdings_excluded": excluded,
            "error": "no overlapping months",
        }

    # Compute per-month portfolio simple return using ONLY holdings present
    # in that month, rescaled so their weights sum to the covered total.
    holdings_by_ticker = {t: (cid, sec, cty, w) for (cid, t, sec, cty, w) in raw}

    portfolio_monthly: list[float] = []
    for m in all_months:
        num = 0.0
        cov_weight = 0.0
        for ticker, rets in holding_series.items():
            if m not in rets:
                continue
            _cid, _sec, _cty, w = holdings_by_ticker[ticker]
            num += (w / 100.0) * rets[m]
            cov_weight += (w / 100.0)
        if cov_weight > 0:
            # If coverage < 100% for this month, scale the portfolio return
            # up to represent a 100%-coverage-equivalent (i.e. the un-covered
            # portion is treated as if it matched the covered basket).
            portfolio_monthly.append(num / cov_weight)

    # Chain simple returns: (1+R1)(1+R2)... - 1
    cumulative = 1.0
    for r in portfolio_monthly:
        cumulative *= (1.0 + r)
    cumulative -= 1.0

    # Per-holding contribution: the holding's actual cumulative return in
    # the window × its portfolio weight. Sum approximates the portfolio
    # return when coverage is close to 100%.
    for ticker, rets in holding_series.items():
        cid, sector, country, weight = holdings_by_ticker[ticker]
        cumr = 1.0
        for m in sorted(rets):
            cumr *= (1.0 + rets[m])
        cumr -= 1.0
        contribution = (weight / 100.0) * cumr
        per_holding.append({
            "ticker": ticker,
            "sector": sector,
            "country": country,
            "weight_pct": round(weight, 3),
            "window_return": cumr,
            "contribution": contribution,
            "n_months": len(rets),
        })
        if sector:
            by_sector[sector] = by_sector.get(sector, 0.0) + contribution
        if country:
            by_country[country] = by_country.get(country, 0.0) + contribution

    per_holding.sort(key=lambda x: x["contribution"])

    return {
        "scenario_key": scenario_key,
        **meta,
        "n_months": len(portfolio_monthly),
        "months_covered": [f"{y:04d}-{m:02d}" for (y, m) in all_months[1:]],
        "portfolio_return": round(cumulative, 6),
        "portfolio_worst_month": round(min(portfolio_monthly), 6) if portfolio_monthly else None,
        "portfolio_best_month":  round(max(portfolio_monthly), 6) if portfolio_monthly else None,
        "holdings_included": list(holding_series.keys()),
        "holdings_excluded": excluded,
        "per_holding": per_holding,
        "by_sector": {k: round(v, 6) for k, v in sorted(by_sector.items(), key=lambda x: x[1])},
        "by_country": {k: round(v, 6) for k, v in sorted(by_country.items(), key=lambda x: x[1])},
    }
