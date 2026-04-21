"""
Automated price feed — fetches daily prices for all active companies.

Primary: Yahoo Finance (yfinance, free, no API key)
Fallback: EODHD (requires API key via EODHD_API_KEY env var)

Bloomberg ticker → Yahoo ticker mapping:
  LKQ US    → LKQ        (US: just ticker)
  BNZL LN   → BNZL.L     (London: .L suffix)
  3679 JP   → 3679.T     (Tokyo: .T suffix)
  005930 KS → 005930.KS  (Korea: .KS suffix)
  ENI IM    → ENI.MI     (Milan: .MI suffix)
  MT NA     → MT.AS      (Amsterdam: .AS suffix)
  FRE GR    → FRE.DE     (Frankfurt: .DE suffix)
  1 HK      → 0001.HK    (Hong Kong: .HK, pad to 4 digits)
  FFH CN    → FFH.TO     (Toronto: .TO suffix)
  UHR SW    → UHR.SW     (Swiss: .SW suffix)
  EXO NA    → EXO.AS     (Amsterdam: .AS suffix)
  SHBA SS   → SHBA.ST    (Stockholm: .ST suffix)
  SAN FP    → SAN.PA     (Paris: .PA suffix)
  HEIA NA   → HEIA.AS    (Amsterdam: .AS suffix)
  JDEP NA   → JDEP.AS    (Amsterdam: .AS suffix)
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta, date

import httpx

from configs.settings import settings

logger = logging.getLogger(__name__)

# Bloomberg exchange suffix → Yahoo suffix
_EXCHANGE_MAP = {
    "US": "",       # US tickers: no suffix
    "LN": ".L",     # London
    "JP": ".T",     # Tokyo
    "KS": ".KS",    # Korea
    "IM": ".MI",    # Milan
    "NA": ".AS",    # Amsterdam (Euronext)
    "GR": ".DE",    # Germany (XETRA)
    "GY": ".DE",    # Germany alt
    "HK": ".HK",    # Hong Kong
    "CN": ".TO",    # Toronto
    "SW": ".SW",    # Swiss
    "SS": ".ST",    # Stockholm
    "FP": ".PA",    # Paris
    "SM": ".MC",    # Madrid
    "BB": ".BR",    # Brussels
    "GA": ".AT",    # Athens
    "ID": ".IR",    # Ireland
}


# Tickers where Bloomberg symbol doesn't match Yahoo directly
_TICKER_OVERRIDES = {
    "SHBA SS": "SHB-A.ST",    # Svenska Handelsbanken A shares
    "BP/ LN": "BP.L",         # Slash in Bloomberg ticker
    "BT/A LN": "BT-A.L",     # Slash in Bloomberg ticker
    "RYA ID": "RYA.IR",        # Ryanair on Irish exchange
}


# Currencies that are quoted in pence / cents on Yahoo Finance and must be
# divided by 100 to reach the major unit before FX conversion. UK LSE is
# the dominant case; Israel (.TA) and South Africa (.JO) are similar but
# no current holdings trip those.
_PENCE_EXCHANGES = {"LN"}     # Bloomberg exchange codes


def is_pence_quoted(ticker: str) -> bool:
    """True if the Yahoo quote for this Bloomberg ticker will arrive in
    minor units (pence / cents) and must be divided by 100 before FX."""
    parts = ticker.strip().split()
    if len(parts) != 2:
        return False
    return parts[1] in _PENCE_EXCHANGES


# Splice table for tickers that changed symbol mid-history (corporate
# restructurings, ADR→local moves, etc.). Each entry is an ordered list
# of legs: (yahoo_ticker, currency, is_pence, use_until).
# use_until=None means "current leg — use up to today".
# Lookup is by Bloomberg ticker (stored in Company.ticker).
SPLICED_TICKERS: dict[str, list[tuple[str, str, bool, date | None]]] = {
    # Alpha Bank: ADR (ALBKY) delisted 2025-06-30 → local Athens line
    "ALPHA GA": [
        ("ALBKY",    "USD", False, date(2025, 6, 30)),
        ("ALPHA.AT", "EUR", False, None),
    ],
}


def bloomberg_to_yahoo(ticker: str) -> str:
    """Convert Bloomberg ticker (e.g. 'BNZL LN') to Yahoo format ('BNZL.L')."""
    if ticker in _TICKER_OVERRIDES:
        return _TICKER_OVERRIDES[ticker]
    parts = ticker.strip().split()
    if len(parts) != 2:
        return ticker  # can't parse, return as-is
    symbol, exchange = parts
    suffix = _EXCHANGE_MAP.get(exchange, "")
    # Hong Kong: pad to 4 digits
    if exchange == "HK" and symbol.isdigit():
        symbol = symbol.zfill(4)
    return f"{symbol}{suffix}"


async def fetch_price_yahoo(yahoo_ticker: str) -> dict | None:
    """Fetch latest price from Yahoo Finance v8 API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
    params = {"range": "5d", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                logger.warning("[PRICE] Yahoo returned %d for %s", resp.status_code, yahoo_ticker)
                return None
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None
            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice")
            currency = meta.get("currency", "USD")
            market_time = meta.get("regularMarketTime")
            if price is None:
                return None
            return {
                "price": float(price),
                "currency": currency,
                "price_date": datetime.fromtimestamp(market_time, tz=timezone.utc) if market_time else datetime.now(timezone.utc),
            }
    except Exception as exc:
        logger.warning("[PRICE] Yahoo fetch failed for %s: %s", yahoo_ticker, exc)
        return None


async def fetch_price_eodhd(ticker_symbol: str, exchange: str) -> dict | None:
    """Fetch latest price from EODHD API (fallback)."""
    api_key = settings.eodhd_api_key
    if not api_key:
        return None

    # EODHD uses SYMBOL.EXCHANGE format
    eodhd_exchanges = {"US": "US", "LN": "LSE", "JP": "TSE", "KS": "KO", "IM": "MI",
                       "NA": "AS", "GR": "XETRA", "GY": "XETRA", "HK": "HK",
                       "CN": "TO", "SW": "SW", "SS": "ST", "FP": "PA"}
    exch = eodhd_exchanges.get(exchange, exchange)
    url = f"https://eodhd.com/api/real-time/{ticker_symbol}.{exch}"
    params = {"api_token": api_key, "fmt": "json"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
            price = data.get("close") or data.get("previousClose")
            if price is None:
                return None
            return {
                "price": float(price),
                "currency": "USD",  # EODHD returns in local currency but we'll handle later
                "price_date": datetime.now(timezone.utc),
            }
    except Exception as exc:
        logger.warning("[PRICE] EODHD fetch failed for %s: %s", ticker_symbol, exc)
        return None


async def refresh_prices(tickers: list[str] | None = None) -> dict:
    """
    Refresh prices for all active companies (or specific tickers).
    Returns summary: {updated: int, failed: int, errors: []}
    """
    from sqlalchemy import select
    from apps.api.database import AsyncSessionLocal
    from apps.api.models import Company, PriceRecord

    async with AsyncSessionLocal() as db:
        query = select(Company).where(Company.coverage_status == "active")
        if tickers:
            query = query.where(Company.ticker.in_(tickers))
        result = await db.execute(query)
        companies = result.scalars().all()

    summary = {"updated": 0, "failed": 0, "skipped": 0, "errors": []}

    for company in companies:
        ticker = company.ticker
        parts = ticker.strip().split()
        if len(parts) != 2:
            summary["skipped"] += 1
            continue

        symbol, exchange = parts
        yahoo_ticker = bloomberg_to_yahoo(ticker)

        # Try Yahoo first
        price_data = await fetch_price_yahoo(yahoo_ticker)

        # Fallback to EODHD
        if price_data is None:
            price_data = await fetch_price_eodhd(symbol, exchange)

        if price_data is None:
            summary["failed"] += 1
            summary["errors"].append(f"{ticker}: no price found")
            logger.warning("[PRICE] No price for %s (yahoo=%s)", ticker, yahoo_ticker)
            continue

        # Save to DB
        try:
            async with AsyncSessionLocal() as db:
                from sqlalchemy import text
                await db.execute(text("""
                    INSERT INTO price_records (id, company_id, price, currency, price_date, source)
                    VALUES (:id, :cid, :price, :currency, :price_date, 'api')
                """), {
                    "id": str(uuid.uuid4()),
                    "cid": str(company.id),
                    "price": price_data["price"],
                    "currency": price_data["currency"],
                    "price_date": price_data["price_date"],
                })
                await db.commit()
            summary["updated"] += 1
            logger.info("[PRICE] %s = %.2f %s", ticker, price_data["price"], price_data["currency"])
        except Exception as exc:
            summary["failed"] += 1
            summary["errors"].append(f"{ticker}: DB error: {exc}")
            logger.error("[PRICE] Failed to save %s: %s", ticker, exc)

    logger.info("[PRICE] Refresh complete: %d updated, %d failed, %d skipped",
                summary["updated"], summary["failed"], summary["skipped"])

    # Auto-snapshot current prices against existing scenarios for chart tracking
    snapshots = await _snapshot_prices_against_scenarios()
    summary["snapshots"] = snapshots

    return summary


async def _snapshot_prices_against_scenarios() -> int:
    """For each company with scenarios + a fresh price, create a ScenarioSnapshot.
    This makes the valuation chart track price movement over time automatically."""
    from apps.api.database import AsyncSessionLocal
    from sqlalchemy import text

    count = 0
    try:
        async with AsyncSessionLocal() as db:
            # Get all companies that have both scenarios and a recent price
            rows = await db.execute(text("""
                SELECT vs.company_id, vs.scenario_type, vs.target_price, vs.probability, vs.currency,
                       pr.price AS current_price
                FROM valuation_scenarios vs
                JOIN LATERAL (
                    SELECT price FROM price_records
                    WHERE company_id = vs.company_id
                    ORDER BY price_date DESC LIMIT 1
                ) pr ON true
                WHERE vs.target_price IS NOT NULL
            """))
            scenarios = rows.all()

            now = datetime.now(timezone.utc)
            for s in scenarios:
                await db.execute(text("""
                    INSERT INTO scenario_snapshots (id, company_id, snapshot_date, scenario_type,
                        target_price, probability, current_price, currency, source)
                    VALUES (:id, :cid, :date, :type, :target, :prob, :price, :curr, 'auto_price')
                """), {
                    "id": str(uuid.uuid4()),
                    "cid": str(s.company_id),
                    "date": now,
                    "type": s.scenario_type,
                    "target": float(s.target_price),
                    "prob": float(s.probability) if s.probability else None,
                    "price": float(s.current_price),
                    "curr": s.currency or "USD",
                })
                count += 1
            await db.commit()
            logger.info("[PRICE] Created %d scenario snapshots", count)
    except Exception as exc:
        logger.warning("[PRICE] Scenario snapshot failed: %s", exc)
    return count
