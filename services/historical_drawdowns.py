"""
Historical drawdowns — empirical distribution of peak-to-trough moves.

Read by Bear Case and Bull Case agents (Tier 2.1) so downside / upside
scenarios anchor to what the stock has actually done, not narrative
stacking. Q1/Q2/Q3 2025 bear cases predicted -38% to -53% on a recovery
quarter; the stock's trailing 3Y worst drawdown was -28%. A bear case
requiring a larger drawdown than the empirical distribution needs to
justify why.

Reads from the `price_records` table populated by services/price_feed.py
(Yahoo primary, EODHD fallback; 25y backfill script seeds history). No
external API calls — everything comes from the local price store.

Splice / currency handling:
  Tickers that changed denomination mid-history (e.g. ALPHA GA: USD ADR
  → EUR local in 2025) produce a spurious price step when naively
  concatenated. We filter to the most-recent currency segment to avoid
  this. A one-time loss of history in the crossover month is an
  acceptable trade for not reporting a -90% "drawdown" that is really a
  currency rebase. Window counts are reported back so callers can see
  whether the history is intact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import PriceRecord

logger = logging.getLogger(__name__)


@dataclass
class DrawdownWindow:
    years: int
    max_drawdown_pct: float        # most negative peak-to-trough over window (e.g. -42.1)
    peak_date: datetime | None
    trough_date: datetime | None
    current_from_high_pct: float   # current price vs window high (≤0 by construction)
    observations: int              # price points used


def _compute_one_window(
    dates: list[datetime],
    prices: list[float],
    years: int,
    now: datetime,
) -> DrawdownWindow | None:
    cutoff = now - timedelta(days=365 * years)
    idx_start = next((i for i, d in enumerate(dates) if d >= cutoff), None)
    if idx_start is None:
        return None

    window_prices = prices[idx_start:]
    window_dates = dates[idx_start:]
    if len(window_prices) < 30:
        return None

    running_peak = window_prices[0]
    running_peak_date = window_dates[0]
    max_dd = 0.0
    max_dd_peak_date = running_peak_date
    max_dd_trough_date = running_peak_date

    for px, dt in zip(window_prices, window_dates):
        if px > running_peak:
            running_peak = px
            running_peak_date = dt
        if running_peak > 0:
            dd = (px - running_peak) / running_peak
            if dd < max_dd:
                max_dd = dd
                max_dd_peak_date = running_peak_date
                max_dd_trough_date = dt

    window_high = max(window_prices)
    current = window_prices[-1]
    current_from_high = (current - window_high) / window_high if window_high else 0.0

    return DrawdownWindow(
        years=years,
        max_drawdown_pct=max_dd * 100,
        peak_date=max_dd_peak_date,
        trough_date=max_dd_trough_date,
        current_from_high_pct=current_from_high * 100,
        observations=len(window_prices),
    )


async def compute_trailing_drawdowns(
    db: AsyncSession,
    company_id,
    windows_years: Iterable[int] = (1, 3, 5),
) -> dict[str, DrawdownWindow]:
    """Return `{"1y": DrawdownWindow, "3y": ..., "5y": ...}`.

    Missing windows are omitted (e.g. a ticker with only 2 years of
    history returns only `1y`). Companies with <30 total price points
    return `{}`.
    """
    windows = tuple(sorted(set(windows_years)))
    if not windows:
        return {}

    now = datetime.now(timezone.utc)
    earliest = now - timedelta(days=365 * max(windows) + 30)

    q = await db.execute(
        select(PriceRecord.price, PriceRecord.currency, PriceRecord.price_date)
        .where(PriceRecord.company_id == company_id)
        .where(PriceRecord.price_date >= earliest)
        .order_by(PriceRecord.price_date.asc())
    )
    rows = q.all()
    if len(rows) < 30:
        return {}

    # Splice guard: drop older segments in a different currency.
    latest_ccy = (rows[-1].currency or "").upper()
    filtered = [r for r in rows if (r.currency or "").upper() == latest_ccy]
    if len(filtered) < 30:
        # All-same-currency run is too short — skip rather than mix.
        return {}

    dates = [r.price_date for r in filtered]
    prices = [float(r.price) for r in filtered]

    out: dict[str, DrawdownWindow] = {}
    for years in windows:
        dw = _compute_one_window(dates, prices, years, now)
        if dw:
            out[f"{years}y"] = dw
    return out


def format_drawdowns_for_prompt(drawdowns: dict[str, DrawdownWindow]) -> str:
    """Compact block for insertion into bear / bull case prompts."""
    if not drawdowns:
        return "(no historical price data available — do not anchor scenarios to any implied distribution)"

    lines = ["Historical peak-to-trough drawdowns (from local price store):"]
    worst = None
    for key in ("1y", "3y", "5y"):
        dw = drawdowns.get(key)
        if not dw:
            continue
        peak = dw.peak_date.strftime("%Y-%m-%d") if dw.peak_date else "—"
        trough = dw.trough_date.strftime("%Y-%m-%d") if dw.trough_date else "—"
        lines.append(
            f"  Trailing {dw.years}Y: max peak-to-trough {dw.max_drawdown_pct:+.1f}% "
            f"({peak} → {trough}); now {dw.current_from_high_pct:+.1f}% from {dw.years}Y high "
            f"[{dw.observations} obs]"
        )
        if worst is None or dw.max_drawdown_pct < worst:
            worst = dw.max_drawdown_pct

    if worst is not None:
        lines.append(
            f"Interpretation: {worst:+.1f}% is the worst actually-realised drawdown "
            "in the observable window. Any bear scenario implying a larger drawdown "
            "must explicitly justify why this episode is different (new idiosyncratic "
            "risk, clearly worse macro than any event in the window, etc.)."
        )
    return "\n".join(lines)


async def build_historical_drawdowns_block(
    db: AsyncSession,
    company_id,
) -> str:
    """One-call helper for the context builder.

    Logs but never raises — a failure here must not block the agent
    pipeline (pricing history may be absent for factor proxies or for
    newly-onboarded holdings).
    """
    try:
        dw = await compute_trailing_drawdowns(db, company_id)
        return format_drawdowns_for_prompt(dw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("historical drawdowns failed for %s: %s", company_id, exc)
        return "(historical drawdowns unavailable)"
