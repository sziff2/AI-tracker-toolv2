"""
Historical-price backfill — Phase A of portfolio analytics roadmap.

Populates `price_records` with 5 years of monthly EOM closes in local
currency for every active company, plus `fx_rates` with monthly EOM
rates for every non-USD currency we touch.

This is the prerequisite for real correlations / realised vol / risk
metrics (Phase B onwards). Without this, those endpoints have no price
history to compute from.

Design:
  - Idempotent: before inserting a (company, month_end) row, checks for
    any existing price_records row on the same calendar date. Re-running
    the script is safe.
  - Uses Yahoo Finance v8 chart API directly (same path as the daily
    price feed) — no pandas, no yfinance dependency.
  - GBp (London pence) → GBP conversion handled inline (÷100). FX is
    applied downstream in the analytics service, NOT here; price_records
    stores local-currency prices to mirror the daily feed.
  - Spliced tickers (e.g. Alpha Bank ALBKY → ALPHA.AT) are stitched from
    the SPLICED_TICKERS table in services/price_feed.py.
  - Source tag: 'backfill' so we can distinguish from 'api' (daily feed).

Usage (from repo root, with DATABASE_URL pointed at target env):
  python scripts/backfill_prices.py --dry-run                 # list only
  python scripts/backfill_prices.py --ticker "BNZL LN"        # one, dry-run
  python scripts/backfill_prices.py --ticker "BNZL LN" --apply
  python scripts/backfill_prices.py --apply                   # full run
  python scripts/backfill_prices.py --apply --years 5
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.price_feed import (                       # noqa: E402
    bloomberg_to_yahoo,
    is_pence_quoted,
    SPLICED_TICKERS,
    _EXCHANGE_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Bloomberg exchange code → settlement currency.
_EXCHANGE_CURRENCY = {
    "US": "USD",
    "LN": "GBP",       # pence handled separately
    "JP": "JPY",
    "KS": "KRW",
    "IM": "EUR",
    "NA": "EUR",
    "GR": "EUR",
    "GY": "EUR",
    "HK": "HKD",
    "CN": "CAD",
    "SW": "CHF",
    "SS": "SEK",
    "FP": "EUR",
    "SM": "EUR",
    "BB": "EUR",
    "GA": "EUR",
    "ID": "EUR",
}

# Local currency → Yahoo FX ticker (quote is: 1 local = X USD).
FX_YAHOO = {
    "GBP": "GBPUSD=X",
    "EUR": "EURUSD=X",
    "SEK": "SEKUSD=X",
    "HKD": "HKDUSD=X",
    "KRW": "KRWUSD=X",
    "JPY": "JPYUSD=X",
    "CAD": "CADUSD=X",
    "CHF": "CHFUSD=X",
}


@dataclass
class PricePoint:
    price_date: date          # calendar EOM date
    close: float              # local currency major unit (pence already divided)
    currency: str


def _month_end(d: date) -> date:
    """Return the last calendar day of d's month."""
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


async def _fetch_yahoo_history(
    client: httpx.AsyncClient,
    yahoo_ticker: str,
    start: date,
    end: date,
) -> list[tuple[date, float, str]]:
    """Fetch monthly-EOM history via Yahoo's DAILY chart and resample.

    We deliberately do NOT use Yahoo's interval=1mo endpoint: its candle
    timestamps are anchored at the exchange's local midnight which, for
    non-US exchanges (CET, KST, HKT, ...), lands on the PRIOR day in UTC.
    Bucketing those timestamps by `_month_end(date)` mis-labels every
    non-US monthly close by one calendar month, destroying cross-market
    correlations.

    Daily candles are timezone-safe: each daily timestamp is within the
    actual trading day. Bucketing by calendar (year, month) and keeping
    the last observation gives a correctly-labelled month-end series.

    Returns (last_trading_date, adj_close, currency) tuples — one row per
    calendar month that had any trading activity in the window.
    """
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
    params = {"period1": period1, "period2": period2, "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = await client.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        logger.warning("Yahoo %d for %s", resp.status_code, yahoo_ticker)
        return []
    data = resp.json()
    results = data.get("chart", {}).get("result") or []
    if not results:
        return []
    r = results[0]
    meta = r.get("meta", {}) or {}
    currency = meta.get("currency", "USD")
    gmtoff = int(meta.get("gmtoffset") or 0)   # seconds east of UTC
    timestamps = r.get("timestamp") or []
    indicators = r.get("indicators", {}) or {}
    # Prefer adjusted close (total-return series) where available.
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    raw = (indicators.get("quote") or [{}])[0].get("close")
    closes = adj if adj else raw
    if not timestamps or not closes:
        return []

    # Bucket daily observations by (year, month) IN THE EXCHANGE'S LOCAL
    # TIME and keep the latest date in each bucket.
    by_month: dict[tuple[int, int], tuple[date, float]] = {}
    for ts, cl in zip(timestamps, closes):
        if cl is None:
            continue
        d_local = datetime.fromtimestamp(ts + gmtoff, tz=timezone.utc).date()
        key = (d_local.year, d_local.month)
        if key not in by_month or d_local > by_month[key][0]:
            by_month[key] = (d_local, float(cl))

    rows: list[tuple[date, float, str]] = [
        (d, cl, currency) for _, (d, cl) in sorted(by_month.items())
    ]
    return rows


async def _fetch_monthly_prices(
    client: httpx.AsyncClient,
    ticker: str,                # Bloomberg ticker
    start: date,
    end: date,
) -> list[PricePoint]:
    """Fetch a single ticker's monthly EOM history, stitching spliced
    tickers and applying GBp → GBP divide where necessary."""

    # Spliced case — concatenate each leg, slicing by use_until.
    if ticker in SPLICED_TICKERS:
        points: list[PricePoint] = []
        last_cutoff: date | None = None
        for leg in SPLICED_TICKERS[ticker]:
            yahoo_tkr, ccy, is_pence, use_until = leg
            leg_start = last_cutoff + timedelta(days=1) if last_cutoff else start
            leg_end = use_until if use_until else end
            if leg_end < start or leg_start > end:
                last_cutoff = use_until
                continue
            rows = await _fetch_yahoo_history(
                client, yahoo_tkr,
                max(leg_start, start), min(leg_end, end),
            )
            for d, cl, yahoo_ccy in rows:
                price = cl / 100.0 if is_pence else cl
                # Trust our config over Yahoo's meta for spliced legs —
                # Yahoo sometimes reports the ADR in USD but the underlying
                # leg in the local currency; the config is authoritative.
                points.append(PricePoint(price_date=d, close=price, currency=ccy))
            last_cutoff = use_until
        return points

    # Normal single-ticker case.
    yahoo_tkr = bloomberg_to_yahoo(ticker)
    rows = await _fetch_yahoo_history(client, yahoo_tkr, start, end)
    parts = ticker.strip().split()
    exchange = parts[1] if len(parts) == 2 else ""
    expected_ccy = _EXCHANGE_CURRENCY.get(exchange, "USD")
    pence = is_pence_quoted(ticker)
    out: list[PricePoint] = []
    for d, cl, yahoo_ccy in rows:
        price = cl / 100.0 if pence else cl
        # Yahoo reports pence as GBp but our downstream analytics keys
        # off standard ISO codes — normalise to GBP after the divide.
        ccy = "GBP" if pence else expected_ccy
        if not pence and yahoo_ccy and yahoo_ccy.upper() not in ("GBP", "GBP"):
            # Trust Yahoo meta when no pence adjustment is needed — e.g.
            # some LSE ADRs quote in USD.
            ccy = yahoo_ccy
        out.append(PricePoint(price_date=d, close=price, currency=ccy))
    return out


async def _fetch_fx_monthly(
    client: httpx.AsyncClient,
    currency: str,
    start: date,
    end: date,
) -> list[tuple[date, float]]:
    """Fetch monthly EOM FX rates for <currency>/USD."""
    yahoo_tkr = FX_YAHOO.get(currency)
    if not yahoo_tkr:
        return []
    rows = await _fetch_yahoo_history(client, yahoo_tkr, start, end)
    return [(d, cl) for d, cl, _ in rows]


async def _load_companies(ticker_filter: str | None):
    from sqlalchemy import select
    from apps.api.database import AsyncSessionLocal
    from apps.api.models import Company

    async with AsyncSessionLocal() as db:
        # Include both real holdings ("active") and factor-shock proxy ETFs
        # ("factor") so a single backfill run keeps both up to date.
        q = select(Company).where(Company.coverage_status.in_(["active", "factor"]))
        if ticker_filter:
            q = q.where(Company.ticker == ticker_filter)
        rs = await db.execute(q)
        return rs.scalars().all()


async def _existing_price_dates(company_id) -> set[date]:
    """Return the set of calendar dates (no tz) where this company already
    has a price_records row. Used for idempotency."""
    from sqlalchemy import text
    from apps.api.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        rs = await db.execute(
            text("SELECT price_date FROM price_records WHERE company_id = :cid"),
            {"cid": str(company_id)},
        )
        out: set[date] = set()
        for row in rs:
            v = row[0]
            if v is None:
                continue
            out.add(v.date() if hasattr(v, "date") else v)
        return out


async def _existing_fx_dates(currency: str) -> set[date]:
    from sqlalchemy import text
    from apps.api.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        rs = await db.execute(
            text("SELECT rate_date FROM fx_rates WHERE currency = :c"),
            {"c": currency},
        )
        return {row[0] for row in rs if row[0] is not None}


async def _insert_prices(company_id, points: list[PricePoint]) -> int:
    from sqlalchemy import text
    from apps.api.database import AsyncSessionLocal

    if not points:
        return 0
    async with AsyncSessionLocal() as db:
        for p in points:
            await db.execute(
                text("""
                    INSERT INTO price_records
                        (id, company_id, price, currency, price_date, source)
                    VALUES
                        (:id, :cid, :price, :ccy, :pd, 'backfill')
                """),
                {
                    "id": str(uuid.uuid4()),
                    "cid": str(company_id),
                    "price": p.close,
                    "ccy": p.currency,
                    "pd": datetime.combine(p.price_date, datetime.min.time(), tzinfo=timezone.utc),
                },
            )
        await db.commit()
    return len(points)


async def _insert_fx(currency: str, rows: list[tuple[date, float]]) -> int:
    from sqlalchemy import text
    from apps.api.database import AsyncSessionLocal

    if not rows:
        return 0
    async with AsyncSessionLocal() as db:
        for d, rate in rows:
            await db.execute(
                text("""
                    INSERT INTO fx_rates
                        (id, currency, rate_date, rate_to_usd, source)
                    VALUES
                        (:id, :c, :d, :r, 'yahoo')
                    ON CONFLICT (currency, rate_date) DO NOTHING
                """),
                {
                    "id": str(uuid.uuid4()),
                    "c": currency,
                    "d": d,
                    "r": rate,
                },
            )
        await db.commit()
    return len(rows)


async def _wipe_backfill(company_id, ticker: str | None = None) -> int:
    """Delete all source='backfill' rows for a company (or all companies
    if company_id is None). Used before re-running with corrected logic."""
    from sqlalchemy import text
    from apps.api.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        if company_id is None:
            rs = await db.execute(text(
                "DELETE FROM price_records WHERE source = 'backfill'"
            ))
        else:
            rs = await db.execute(
                text("DELETE FROM price_records WHERE source = 'backfill' AND company_id = :cid"),
                {"cid": str(company_id)},
            )
        await db.commit()
        return rs.rowcount or 0


async def _wipe_fx() -> int:
    """Delete all fx_rates (they were written with the same timezone bug).
    Regenerated on next backfill from corrected fetch function."""
    from sqlalchemy import text
    from apps.api.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        rs = await db.execute(text("DELETE FROM fx_rates"))
        await db.commit()
        return rs.rowcount or 0


async def run_backfill(
    *,
    ticker: str | None = None,
    years: float = 5.0,
    apply: bool = False,
    clean: bool = False,
) -> dict:
    """Backfill entry point — callable from CLI or from a FastAPI route.
    Returns a summary dict with counts and skipped tickers."""
    end_date = date.today().replace(day=1) - timedelta(days=1)   # prior EOM
    start_date = end_date - timedelta(days=int(years * 365.25) + 31)
    start_date = _month_end(start_date)
    logger.info("Window: %s → %s (%.1f years)", start_date, end_date, years)

    companies = await _load_companies(ticker)
    logger.info("Scope: %d companies", len(companies))
    summary: dict = {
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "companies_scoped": len(companies),
        "price_rows_written": 0,
        "price_rows_deleted": 0,
        "fx_rows_written": 0,
        "fx_rows_deleted": 0,
        "per_ticker": [],
        "skipped": [],
        "fx_currencies": [],
        "mode": "APPLIED" if apply else "DRY-RUN",
        "clean": clean,
    }
    if not companies:
        logger.warning("No companies matched — nothing to do.")
        return summary

    # Optional: wipe existing backfill rows so a corrected run doesn't
    # leave stale data alongside freshly-inserted rows.
    if clean and apply:
        if ticker is None:
            summary["price_rows_deleted"] = await _wipe_backfill(None)
            summary["fx_rows_deleted"] = await _wipe_fx()
        else:
            for cid, _ in [(c.id, c.ticker) for c in companies]:
                summary["price_rows_deleted"] += await _wipe_backfill(cid)
        logger.info("Cleaned %d backfill price rows, %d fx rows",
                    summary["price_rows_deleted"], summary["fx_rows_deleted"])

    currencies_seen: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        for idx, c in enumerate(companies, 1):
            try:
                points = await _fetch_monthly_prices(client, c.ticker, start_date, end_date)
            except Exception as exc:
                logger.warning("[%d/%d] %s: fetch failed — %s", idx, len(companies), c.ticker, exc)
                summary["skipped"].append({"ticker": c.ticker, "reason": f"fetch: {exc}"})
                continue
            if not points:
                logger.warning("[%d/%d] %s: Yahoo returned no rows", idx, len(companies), c.ticker)
                summary["skipped"].append({"ticker": c.ticker, "reason": "no rows from Yahoo"})
                continue

            for p in points:
                currencies_seen.add(p.currency)

            existing = await _existing_price_dates(c.id)
            fresh = [p for p in points if p.price_date not in existing]
            if not fresh:
                logger.info("[%d/%d] %s: already fully backfilled (%d rows on file)",
                            idx, len(companies), c.ticker, len(existing))
                summary["per_ticker"].append({
                    "ticker": c.ticker, "new_rows": 0, "existing": len(existing),
                })
                continue

            sample = fresh[0]
            logger.info("[%d/%d] %s: %d new rows (%s .. %s, %s, sample=%.4f)",
                        idx, len(companies), c.ticker,
                        len(fresh), fresh[0].price_date, fresh[-1].price_date,
                        sample.currency, sample.close)
            summary["per_ticker"].append({
                "ticker": c.ticker,
                "new_rows": len(fresh),
                "first": fresh[0].price_date.isoformat(),
                "last": fresh[-1].price_date.isoformat(),
                "currency": sample.currency,
            })

            if apply:
                inserted = await _insert_prices(c.id, fresh)
                summary["price_rows_written"] += inserted

    # FX rates — one pull per distinct non-USD currency.
    non_usd = sorted(currencies_seen - {"USD"})
    summary["fx_currencies"] = non_usd
    if non_usd:
        logger.info("FX currencies to backfill: %s", ", ".join(non_usd))
        async with httpx.AsyncClient(timeout=30.0) as client:
            for ccy in non_usd:
                try:
                    rows = await _fetch_fx_monthly(client, ccy, start_date, end_date)
                except Exception as exc:
                    logger.warning("FX %s: fetch failed — %s", ccy, exc)
                    continue
                if not rows:
                    logger.warning("FX %s: Yahoo returned no rows", ccy)
                    continue
                existing = await _existing_fx_dates(ccy)
                fresh = [(d, r) for d, r in rows if d not in existing]
                logger.info("FX %s: %d new monthly EOM rates (of %d returned)",
                            ccy, len(fresh), len(rows))
                if apply and fresh:
                    summary["fx_rows_written"] += await _insert_fx(ccy, fresh)

    logger.info("─" * 60)
    logger.info("Done. %s: %d price rows, %d FX rows.",
                summary["mode"], summary["price_rows_written"], summary["fx_rows_written"])
    if summary["skipped"]:
        logger.warning("Skipped %d tickers:", len(summary["skipped"]))
        for item in summary["skipped"]:
            logger.warning("  %s — %s", item["ticker"], item["reason"])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backfill 5yr monthly EOM price history + FX rates.")
    ap.add_argument("--ticker", help="Only backfill this Bloomberg ticker")
    ap.add_argument("--years", type=float, default=5.0, help="Lookback window in years (default 5)")
    ap.add_argument("--apply", action="store_true", help="Write to DB (default: dry-run)")
    ap.add_argument("--clean", action="store_true",
                    help="Delete existing source='backfill' rows first (and all fx_rates if no --ticker)")
    cli_args = ap.parse_args()
    asyncio.run(run_backfill(
        ticker=cli_args.ticker, years=cli_args.years,
        apply=cli_args.apply, clean=cli_args.clean,
    ))
