"""Tests for services/historical_drawdowns — synthetic price series, known answers.

Uses a fake AsyncSession that returns hand-crafted `(price, currency, price_date)`
rows so we don't need a running Postgres.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import pytest

from services.historical_drawdowns import (
    DrawdownWindow,
    compute_trailing_drawdowns,
    format_drawdowns_for_prompt,
    _compute_one_window,
)


@dataclass
class _Row:
    price: float
    currency: str
    price_date: datetime


class _FakeResult:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def all(self) -> list[_Row]:
        return self._rows


class _FakeSession:
    """Minimal AsyncSession stand-in: .execute() returns whatever .all() needs."""

    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    async def execute(self, *_args, **_kwargs) -> _FakeResult:
        return _FakeResult(self._rows)


# ─────────────────────────────────────────────────────────────────
# _compute_one_window — pure function tests
# ─────────────────────────────────────────────────────────────────

def _daily_prices(pattern: list[float], days_back: int = 365) -> tuple[list[datetime], list[float]]:
    """Repeat a price pattern daily, ending 'now'. Returns (dates, prices)."""
    now = datetime.now(timezone.utc)
    dates = [now - timedelta(days=days_back - i) for i in range(days_back)]
    prices = [pattern[i % len(pattern)] for i in range(days_back)]
    return dates, prices


def test_compute_one_window_captures_peak_to_trough():
    """Price goes 100→120→80→100 — max drawdown should be -33.3% (120→80)."""
    now = datetime.now(timezone.utc)
    # 120 daily points — enough to pass the 30-obs floor
    prices = []
    for i in range(30):
        prices.append(100.0)
    for i in range(30):
        prices.append(100.0 + (i + 1) * (20 / 30))  # ramp to 120
    for i in range(30):
        prices.append(120.0 - (i + 1) * (40 / 30))  # drop to 80
    for i in range(30):
        prices.append(80.0 + (i + 1) * (20 / 30))  # recover to 100

    dates = [now - timedelta(days=len(prices) - i) for i in range(len(prices))]

    dw = _compute_one_window(dates, prices, years=1, now=now)
    assert dw is not None
    # 80 / 120 - 1 = -33.3%
    assert -34.0 < dw.max_drawdown_pct < -33.0
    # Current (100) vs window high (120) = -16.6%
    assert -17.0 < dw.current_from_high_pct < -16.0
    assert dw.observations == len(prices)


def test_compute_one_window_returns_none_when_too_few_points():
    now = datetime.now(timezone.utc)
    dates = [now - timedelta(days=i) for i in range(10)]
    prices = [100.0] * 10
    assert _compute_one_window(dates, prices, years=1, now=now) is None


def test_compute_one_window_flat_series_zero_drawdown():
    now = datetime.now(timezone.utc)
    dates = [now - timedelta(days=365 - i) for i in range(365)]
    prices = [100.0] * 365
    dw = _compute_one_window(dates, prices, years=1, now=now)
    assert dw is not None
    assert dw.max_drawdown_pct == 0.0
    assert dw.current_from_high_pct == 0.0


# ─────────────────────────────────────────────────────────────────
# compute_trailing_drawdowns — full async path
# ─────────────────────────────────────────────────────────────────

def test_insufficient_history_returns_empty():
    rows = [
        _Row(100.0, "USD", datetime.now(timezone.utc) - timedelta(days=i))
        for i in range(10)
    ]
    session = _FakeSession(rows)
    result = asyncio.run(compute_trailing_drawdowns(session, company_id="X"))
    assert result == {}


def test_mixed_currency_filters_to_latest_only():
    """Old rows in EUR, recent rows in USD — only USD segment should be used."""
    now = datetime.now(timezone.utc)
    rows: list[_Row] = []
    # 40 EUR rows (oldest) — would create a discontinuity if included
    for i in range(40):
        rows.append(_Row(50.0, "EUR", now - timedelta(days=400 - i)))
    # 100 USD rows (most recent)
    for i in range(100):
        rows.append(_Row(100.0, "USD", now - timedelta(days=100 - i)))

    session = _FakeSession(rows)
    result = asyncio.run(compute_trailing_drawdowns(session, company_id="X", windows_years=(1,)))
    assert "1y" in result
    dw = result["1y"]
    # Flat USD series at 100 → zero drawdown
    assert dw.max_drawdown_pct == 0.0
    assert dw.observations == 100


def test_drawdown_survives_realistic_dip():
    now = datetime.now(timezone.utc)
    rows: list[_Row] = []
    # 365 daily points: 60 at 100, 60 ramp to 150, 60 crash to 90, 185 recover to 120
    series = (
        [100.0] * 60
        + [100.0 + i for i in range(1, 51)]   # 101..150  (50 pts)
        + [150.0 - i for i in range(1, 61)]   # 149..90   (60 pts)
        + [90.0 + (i * 30 / 195) for i in range(195)]  # 90 → ~120
    )
    for i, p in enumerate(series):
        rows.append(_Row(p, "USD", now - timedelta(days=len(series) - i)))

    session = _FakeSession(rows)
    result = asyncio.run(compute_trailing_drawdowns(session, company_id="X", windows_years=(1,)))
    assert "1y" in result
    dw = result["1y"]
    # Peak-to-trough is 150 → 90 = -40%
    assert -41.0 < dw.max_drawdown_pct < -39.0


# ─────────────────────────────────────────────────────────────────
# format_drawdowns_for_prompt — output shape
# ─────────────────────────────────────────────────────────────────

def test_format_empty_gives_safe_string():
    out = format_drawdowns_for_prompt({})
    assert "no historical price data" in out.lower()
    # Agents should know not to anchor
    assert "do not anchor" in out.lower()


def test_format_populated_lists_windows_and_interpretation():
    now = datetime.now(timezone.utc)
    drawdowns = {
        "1y": DrawdownWindow(
            years=1, max_drawdown_pct=-12.3,
            peak_date=now - timedelta(days=200),
            trough_date=now - timedelta(days=150),
            current_from_high_pct=-4.1, observations=252,
        ),
        "3y": DrawdownWindow(
            years=3, max_drawdown_pct=-28.5,
            peak_date=now - timedelta(days=800),
            trough_date=now - timedelta(days=700),
            current_from_high_pct=-5.0, observations=756,
        ),
    }
    out = format_drawdowns_for_prompt(drawdowns)
    assert "Trailing 1Y" in out
    assert "Trailing 3Y" in out
    assert "-12.3%" in out
    assert "-28.5%" in out
    # Interpretation line mentions the worst figure
    assert "-28.5%" in out
    assert "worst actually-realised" in out
