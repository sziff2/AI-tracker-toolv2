"""
Portfolio analytics — real correlation / covariance / realised vol from
`price_records` + `fx_rates`.

Phase B of the portfolio analytics roadmap. Replaces the UI's sector/country
proximity correlation heuristic and the scenario-spread vol proxy with
actual monthly log-return statistics over a rolling window.

Design notes:
  - Uses only the Python standard library (math, statistics) — no numpy
    dependency. At 30 tickers × 60 months the math is trivial (~54k
    multiplications for the full correlation matrix).
  - Monthly resampling: for each (company, calendar month) we take the
    latest price_records row in that month as the month-end close.
    This merges the pre-2025-10 backfill (already EOM-aligned) with the
    post-2025-10 daily feed seamlessly.
  - USD conversion: local price × rate_to_usd. FX row for the same
    calendar month is preferred; if absent (e.g. current month not yet
    backfilled) we carry-forward the most recent FX rate on file.
  - A ticker is dropped from the correlation matrix if it has fewer
    than `min_returns` usable monthly returns in the window (default 12).
  - Results are cached in-memory with a 24h TTL keyed on
    (portfolio_id, window_months). Daily price refresh doesn't invalidate
    the cache — the next-day recompute happens naturally when the TTL
    expires. Mutating the portfolio flushes the cache entry.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession


# ─────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────
_CACHE_TTL_SECONDS = 24 * 60 * 60     # 24h


@dataclass
class _CacheEntry:
    payload: dict
    stored_at: float = field(default_factory=time.time)


_CORR_CACHE: dict[tuple[str, int], _CacheEntry] = {}
_VOL_CACHE: dict[tuple[str, int], _CacheEntry] = {}


def _cache_get(cache: dict, key: tuple) -> dict | None:
    entry = cache.get(key)
    if entry is None:
        return None
    if time.time() - entry.stored_at > _CACHE_TTL_SECONDS:
        cache.pop(key, None)
        return None
    return entry.payload


def _cache_put(cache: dict, key: tuple, payload: dict) -> None:
    cache[key] = _CacheEntry(payload=payload)


def flush_cache() -> None:
    """Drop all cached results — call after portfolio/holding edits."""
    _CORR_CACHE.clear()
    _VOL_CACHE.clear()


# ─────────────────────────────────────────────────────────────────
# Data access
# ─────────────────────────────────────────────────────────────────
def _year_month(d: date) -> tuple[int, int]:
    return (d.year, d.month)


async def _load_fx_by_month(db: AsyncSession) -> dict[str, dict[tuple[int, int], float]]:
    """Return {currency: {(yyyy, mm): rate_to_usd}} for every FX row on file.
    Used to convert local-currency price_records to USD."""
    rs = await db.execute(text(
        "SELECT currency, rate_date, rate_to_usd FROM fx_rates ORDER BY currency, rate_date"
    ))
    out: dict[str, dict[tuple[int, int], float]] = {}
    for ccy, rate_date, rate in rs:
        ym = _year_month(rate_date)
        out.setdefault(ccy, {})[ym] = float(rate)
    # Implicit USD identity row.
    return out


def _fx_for_month(
    fx_by_ccy: dict[str, dict[tuple[int, int], float]],
    currency: str,
    ym: tuple[int, int],
) -> float | None:
    """Return the USD conversion rate for <currency> at calendar month ym.
    Falls back to the most recent prior month if that exact month is
    missing (common for the current in-progress month)."""
    if currency.upper() == "USD":
        return 1.0
    # Normalise GBp (pence) — prices are already divided by 100 in backfill,
    # but the daily feed may store GBp. Downstream callers should pass
    # already-converted major-unit prices.
    series = fx_by_ccy.get(currency.upper())
    if not series:
        return None
    if ym in series:
        return series[ym]
    # Carry-forward: find the latest key <= ym
    latest_key = None
    for k in series:
        if k <= ym and (latest_key is None or k > latest_key):
            latest_key = k
    return series[latest_key] if latest_key else None


async def _load_monthly_series_usd(
    db: AsyncSession,
    company_id: UUID,
    window_months: int,
    fx_by_ccy: dict[str, dict[tuple[int, int], float]],
) -> dict[tuple[int, int], float]:
    """For one company, return {(year, month): usd_close} covering the last
    `window_months+1` calendar months (we need N+1 months to compute N returns).

    Rule: take the latest price_records row within each calendar month as
    that month's close. This cleanly merges EOM backfill rows with daily-
    feed rows. Prices stored as GBp are converted to GBP before FX.
    """
    rs = await db.execute(text("""
        SELECT price, currency, price_date
          FROM price_records
         WHERE company_id = :cid
         ORDER BY price_date ASC
    """), {"cid": str(company_id)})
    rows = rs.all()

    # Bucket by (year, month) and keep latest by price_date.
    # Currency tag is kept case-sensitive — "GBp" (pence) is distinct
    # from "GBP" (pounds). Yahoo's daily feed reports LSE quotes as
    # "GBp"; the backfill pre-divides by 100 and writes "GBP".
    by_month: dict[tuple[int, int], tuple[datetime, float, str]] = {}
    for price, ccy, pd_ in rows:
        if price is None or pd_ is None:
            continue
        ym = _year_month(pd_.date() if hasattr(pd_, "date") else pd_)
        if ym not in by_month or pd_ > by_month[ym][0]:
            by_month[ym] = (pd_, float(price), ccy or "USD")

    # Convert to USD.
    usd_series: dict[tuple[int, int], float] = {}
    for ym, (pd_, price, ccy) in by_month.items():
        if ccy == "GBp":
            local = price / 100.0
            ccy_std = "GBP"
        else:
            local = price
            ccy_std = ccy.upper()
        rate = _fx_for_month(fx_by_ccy, ccy_std, ym)
        if rate is None:
            continue
        usd_series[ym] = local * rate

    # Trim to the window: we want the N+1 most recent months.
    if not usd_series:
        return {}
    ordered = sorted(usd_series.keys())
    cutoff = ordered[-(window_months + 1):]
    return {ym: usd_series[ym] for ym in cutoff}


# ─────────────────────────────────────────────────────────────────
# Math
# ─────────────────────────────────────────────────────────────────
def _log_returns(series: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Compute log returns between consecutive calendar-month closes.
    Returns {month: log(p_t / p_{t-1})}, indexed on the later month."""
    if len(series) < 2:
        return {}
    ordered = sorted(series.keys())
    out: dict[tuple[int, int], float] = {}
    for prev, curr in zip(ordered[:-1], ordered[1:]):
        p0, p1 = series[prev], series[curr]
        if p0 and p0 > 0 and p1 > 0:
            out[curr] = math.log(p1 / p0)
    return out


def _aligned_returns(
    returns_by_ticker: dict[str, dict[tuple[int, int], float]],
    min_months: int,
) -> tuple[list[str], list[tuple[int, int]], list[list[float]]]:
    """Align each ticker's returns onto the common set of months where
    every ticker has a value. Drops tickers with fewer than `min_months`
    observations in the window before alignment.

    Returns (kept_tickers, months_used, matrix) where matrix[i] is the
    aligned return series for ticker i, matrix[i][j] for month j.
    """
    eligible = [t for t, r in returns_by_ticker.items() if len(r) >= min_months]
    if len(eligible) < 2:
        return eligible, [], []
    # Start with the intersection of month sets.
    common: set[tuple[int, int]] = set.intersection(
        *(set(returns_by_ticker[t].keys()) for t in eligible)
    )
    if len(common) < min_months:
        return eligible, [], []
    months = sorted(common)
    matrix = [[returns_by_ticker[t][m] for m in months] for t in eligible]
    return eligible, months, matrix


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _covariance(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return math.sqrt(_covariance(xs, xs))


def _corr(xs: list[float], ys: list[float]) -> float:
    sx, sy = _stdev(xs), _stdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    return _covariance(xs, ys) / (sx * sy)


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────
async def compute_correlation_matrix(
    db: AsyncSession,
    portfolio_id: str | UUID,
    window_months: int = 60,
    min_months: int = 12,
    use_cache: bool = True,
) -> dict:
    """Return the monthly-return covariance + correlation matrix for every
    holding in the portfolio, computed in USD over the last N months.

    Shape:
        {
            "portfolio_id": "...",
            "window_months": 60,
            "min_months": 12,
            "as_of": "2026-03-31T...",
            "tickers": ["LKQ US", "BNZL LN", ...],
            "months_used": ["2021-04", ..., "2026-03"],
            "n_months": 60,
            "corr": [[1.0, 0.42, ...], [...], ...],
            "cov":  [[...], [...], ...],
            "dropped": [{"ticker": "X", "reason": "insufficient history"}]
        }
    """
    pid = str(portfolio_id)
    cache_key = (pid, int(window_months))
    if use_cache:
        cached = _cache_get(_CORR_CACHE, cache_key)
        if cached:
            return cached

    # Holdings for this portfolio.
    rs = await db.execute(text("""
        SELECT c.id AS cid, c.ticker
          FROM portfolio_holdings h
          JOIN companies c ON c.id = h.company_id
         WHERE h.portfolio_id = :pid
         ORDER BY h.weight DESC NULLS LAST
    """), {"pid": pid})
    holdings = [(row.cid, row.ticker) for row in rs]
    if not holdings:
        return {
            "portfolio_id": pid, "window_months": window_months,
            "min_months": min_months, "tickers": [], "months_used": [],
            "n_months": 0, "corr": [], "cov": [], "dropped": [],
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    fx = await _load_fx_by_month(db)

    returns_by_ticker: dict[str, dict[tuple[int, int], float]] = {}
    dropped: list[dict] = []
    for cid, ticker in holdings:
        series = await _load_monthly_series_usd(db, cid, window_months, fx)
        if len(series) < min_months + 1:
            dropped.append({"ticker": ticker, "reason": f"only {len(series)} months of USD prices"})
            continue
        rets = _log_returns(series)
        if len(rets) < min_months:
            dropped.append({"ticker": ticker, "reason": f"only {len(rets)} monthly returns"})
            continue
        returns_by_ticker[ticker] = rets

    kept, months, matrix = _aligned_returns(returns_by_ticker, min_months)

    # Anything eligible but excluded by alignment?
    eligible_set = set(returns_by_ticker.keys())
    kept_set = set(kept)
    for t in eligible_set - kept_set:
        dropped.append({"ticker": t, "reason": "alignment dropped — overlap below min_months"})

    n = len(kept)
    corr_mtx = [[0.0] * n for _ in range(n)]
    cov_mtx = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_mtx[i][j] = 1.0
                cov_mtx[i][j] = _covariance(matrix[i], matrix[j])
            else:
                corr_mtx[i][j] = _corr(matrix[i], matrix[j])
                cov_mtx[i][j] = _covariance(matrix[i], matrix[j])

    payload = {
        "portfolio_id": pid,
        "window_months": window_months,
        "min_months": min_months,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "tickers": kept,
        "months_used": [f"{y:04d}-{m:02d}" for (y, m) in months],
        "n_months": len(months),
        "corr": corr_mtx,
        "cov": cov_mtx,
        "dropped": dropped,
    }
    if use_cache:
        _cache_put(_CORR_CACHE, cache_key, payload)
    return payload


async def compute_realised_vol(
    db: AsyncSession,
    company_id: str | UUID,
    ticker: str,
    window_months: int = 36,
    min_months: int = 12,
    use_cache: bool = True,
) -> dict:
    """Return annualised realised volatility of monthly USD log returns."""
    key = (f"{company_id}:{ticker}", int(window_months))
    if use_cache:
        cached = _cache_get(_VOL_CACHE, key)
        if cached:
            return cached

    fx = await _load_fx_by_month(db)
    series = await _load_monthly_series_usd(db, company_id, window_months, fx)
    rets = _log_returns(series)
    values = list(rets.values())
    if len(values) < min_months:
        payload = {
            "ticker": ticker, "window_months": window_months,
            "months_used": len(values), "monthly_vol": None,
            "annualised_vol": None,
            "status": "insufficient_history",
            "min_months": min_months,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        if use_cache:
            _cache_put(_VOL_CACHE, key, payload)
        return payload

    monthly_vol = _stdev(values)
    payload = {
        "ticker": ticker,
        "window_months": window_months,
        "months_used": len(values),
        "monthly_vol": monthly_vol,
        "annualised_vol": monthly_vol * math.sqrt(12),
        "status": "ok",
        "min_months": min_months,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    if use_cache:
        _cache_put(_VOL_CACHE, key, payload)
    return payload
