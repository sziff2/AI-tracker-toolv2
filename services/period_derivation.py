"""
Period-shape derivation — fill gaps in the canonical taxonomy.

Given a set of extracted metrics tagged with the canonical shapes
(Q1, Q2, Q3, Q4, H1, H2, L3Q, FY), compute the SHAPES we don't have
from the ones we do, using additive identities:

    H1  = Q1 + Q2
    H2  = Q3 + Q4
    L3Q = Q1 + Q2 + Q3
    FY  = Q1 + Q2 + Q3 + Q4 = H1 + H2 = L3Q + Q4
    Q4  = FY - L3Q = FY - H1 - Q3 = H2 - Q3
    Q3  = L3Q - H1 = L3Q - Q1 - Q2
    Q1  = H1 - Q2
    Q2  = H1 - Q1
    H2  = FY - H1
    H1  = FY - H2
    L3Q = FY - Q4

Only applies to FLOW metrics (income-statement / cash-flow line items
that accumulate across a period). STOCK metrics (balance-sheet items
— total assets, debt, equity, cash balance) are point-in-time
snapshots and are NOT additive — those are excluded by name pattern.

LTM (trailing 12 months) is derived in a separate cross-year pass
(`_derive_ltm_rows`). Three as-of points are supported:
    LTM_Q3 = Q4(Y-1) + Q1(Y) + Q2(Y) + Q3(Y)
    LTM_Q2 = Q3(Y-1) + Q4(Y-1) + Q1(Y) + Q2(Y)
    LTM_Q1 = Q2(Y-1) + Q3(Y-1) + Q4(Y-1) + Q1(Y)
LTM-as-of-Q4 isn't computed since it equals FY for the same year.

Output rows are tagged is_derived=True with a source_snippet
explaining the formula used, so callers can flag them in the UI.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Stock-metric exclusion (NOT additive across periods)
# ─────────────────────────────────────────────────────────────────

_STOCK_PATTERNS = [
    # Balance sheet
    r"^cash_and_cash_equivalents",
    r"^cash_balance",
    r"^total_assets",
    r"^total_liabilities",
    r"^total_equity",
    r"^total_shareholders_equity",
    r"^stockholders_equity",
    r"^total_debt",
    r"^net_debt",
    r"^long_term_debt",
    r"^short_term_debt",
    r"^working_capital",
    r"^retained_earnings",
    r"^common_stock",
    r"^treasury_stock",
    r"^accumulated_",
    r"^right_of_use_asset",
    r"^total_operating_lease_liabilities",
    r"^segment_assets",
    r"^consolidated_assets",
    r"^total_segment_assets",
    r"^total_liabilities_and_equity",
    r"^capital_in_excess_of_par_value",
    # Headcounts / scale (point-in-time)
    r"^shares_outstanding",
    r"^shareholders_of_record",
    r"^employees_",
    r"^headcount",
    # Inventory / receivable / payable balances
    r"^inventory$",
    r"^inventories$",
    r"^accounts_receivable",
    r"^accounts_payable",
    # Cash-flow statement bookend balances
    r"_balance$",
    r"_at_(beginning|end)_of_(year|period|quarter)",
    # Per-share metrics (averages, not summable)
    r"^eps",
    r"^earnings_per_share",
    r"^book_value_per_share",
    # Ratios / percentages — generally not additive either
    r"_margin$",
    r"_rate$",
    r"_ratio$",
    r"_percentage$",
    r"_percent$",
    r"^return_on_",
    r"_yield$",
]


def is_stock_metric(metric_name: str) -> bool:
    """True when the metric is a point-in-time / per-share / ratio
    figure that does NOT add across periods."""
    n = (metric_name or "").lower()
    return any(re.search(p, n) for p in _STOCK_PATTERNS)


# ─────────────────────────────────────────────────────────────────
# Derivation rules
# ─────────────────────────────────────────────────────────────────
# Each rule: (target_shape, [(input_shape, sign), ...])
# Order matters — more specific / fewer-input rules first so we get
# the cleanest derivation when multiple paths are possible.

DERIVATION_RULES: list[tuple[str, list[tuple[str, int]]]] = [
    # Sums of quarters
    ("H1",  [("Q1", +1), ("Q2", +1)]),
    ("H2",  [("Q3", +1), ("Q4", +1)]),
    ("L3Q", [("Q1", +1), ("Q2", +1), ("Q3", +1)]),
    ("FY",  [("H1", +1), ("H2", +1)]),
    ("FY",  [("L3Q", +1), ("Q4", +1)]),
    ("FY",  [("Q1", +1), ("Q2", +1), ("Q3", +1), ("Q4", +1)]),
    # Subtractive rearrangements
    ("Q4",  [("FY", +1), ("L3Q", -1)]),
    ("Q4",  [("H2", +1), ("Q3", -1)]),
    ("Q4",  [("FY", +1), ("H1", -1), ("Q3", -1)]),
    ("Q3",  [("L3Q", +1), ("H1", -1)]),
    ("Q3",  [("L3Q", +1), ("Q1", -1), ("Q2", -1)]),
    ("Q3",  [("H2", +1), ("Q4", -1)]),
    ("Q2",  [("H1", +1), ("Q1", -1)]),
    ("Q1",  [("H1", +1), ("Q2", -1)]),
    ("L3Q", [("FY", +1), ("Q4", -1)]),
    ("H1",  [("FY", +1), ("H2", -1)]),
    ("H2",  [("FY", +1), ("H1", -1)]),
]


# ─────────────────────────────────────────────────────────────────
# Derivation engine
# ─────────────────────────────────────────────────────────────────

def derive_period_metrics(
    rows: list[dict],
    *,
    skip_stocks: bool = True,
    confidence_penalty: float = 0.90,
) -> list[dict]:
    """Compute missing period-shape values from additive identities.

    Args:
        rows: list of metric dicts. Each must have at least
            metric_name, period_label, period_frequency, metric_value.
            Optional: segment, unit, confidence.
        skip_stocks: when True (default), skip balance-sheet / point-
            in-time metrics that aren't additive across periods.
        confidence_penalty: derived rows multiply input confidence by
            this factor (0.90 → derived rows score 10% lower than the
            weakest input).

    Returns:
        list of derived metric dicts with is_derived=True. Each row's
        period_label is "{year}_{shape}" matching the new canonical
        taxonomy.
    """
    # Group rows by (metric_name, segment, year)
    groups: dict[tuple[str, str, str], dict[str, dict]] = {}
    for r in rows:
        name = r.get("metric_name", "")
        if not name:
            continue
        if skip_stocks and is_stock_metric(name):
            continue
        seg = (r.get("segment") or "") or ""
        label = (r.get("period_label") or "").upper()
        m = re.match(r"^(\d{4})_(Q[1-4]|H[12]|L3Q|FY)$", label)
        if not m:
            continue
        year, shape = m.group(1), m.group(2)
        key = (name, seg, year)
        # Don't overwrite an existing real row for this shape.
        groups.setdefault(key, {}).setdefault(shape, r)

    derived: list[dict] = []
    for (name, seg, year), shape_rows in groups.items():
        # Apply rules iteratively — a derivation can unlock a chain
        # (e.g. derive H1 from Q1+Q2, then derive FY from H1+H2).
        # Cap iterations so we can't loop forever on bad inputs.
        for _ in range(4):
            progress = False
            for target_shape, ingredients in DERIVATION_RULES:
                if target_shape in shape_rows:
                    continue
                inputs = []
                ok = True
                for shape, sign in ingredients:
                    if shape not in shape_rows:
                        ok = False
                        break
                    if shape_rows[shape].get("metric_value") is None:
                        ok = False
                        break
                    inputs.append((shape_rows[shape], sign))
                if not ok:
                    continue
                try:
                    value = sum(
                        sign * float(r["metric_value"])
                        for r, sign in inputs
                    )
                except (TypeError, ValueError):
                    continue
                template = inputs[0][0]
                # Build a human-readable formula label
                formula_parts = []
                for shape, sign in ingredients:
                    formula_parts.append(("+" if sign > 0 else "-") + shape)
                formula = " ".join(formula_parts)
                input_conf = min(
                    float(r.get("confidence") or 0.9) for r, _ in inputs
                )
                derived_row = {
                    "metric_name":      name,
                    "metric_value":     value,
                    "metric_text":      f"{value} (derived)",
                    "unit":             template.get("unit"),
                    "segment":          seg or None,
                    "period_label":     f"{year}_{target_shape}",
                    "period_frequency": target_shape,
                    "source_snippet":   f"derived: {target_shape} = {formula}",
                    "confidence":       round(input_conf * confidence_penalty, 3),
                    "is_derived":       True,
                }
                derived.append(derived_row)
                shape_rows[target_shape] = derived_row
                progress = True
            if not progress:
                break

    # ─────────────────────────────────────────────────────────────
    # LTM (trailing twelve months) — cross-year derivation
    # ─────────────────────────────────────────────────────────────
    # The same-year rules above can't reach LTM since it spans two
    # fiscal years (e.g. LTM ending Q3 2025 = Q4 2024 + Q1 2025 +
    # Q2 2025 + Q3 2025). We need a separate pass that groups by
    # (metric_name, segment) and walks across years.
    #
    # Convention: derived LTM rows are labelled `{year}_LTM` and
    # tagged with the as-of quarter in the source_snippet so the
    # consumer can disambiguate. Three useful as-of points are
    # supported (Q1, Q2, Q3 of `year`); LTM-as-of-Q4 is identical
    # to FY and would just duplicate it.
    derived.extend(
        _derive_ltm_rows(rows, groups, derived, skip_stocks, confidence_penalty)
    )

    if derived:
        logger.info(
            "Period-derivation: produced %d derived metric-rows across %d (metric, segment, year) groups",
            len(derived), len(groups),
        )
    return derived


def _derive_ltm_rows(
    rows: list[dict],
    groups: dict[tuple[str, str, str], dict[str, dict]],
    already_derived: list[dict],
    skip_stocks: bool,
    confidence_penalty: float,
) -> list[dict]:
    """LTM ending Q[1-3] of `year` = Q[1-3](year) + earlier quarters
    that span back into year-1. Computed as:

        LTM_Q3 = Q4(Y-1) + Q1(Y) + Q2(Y) + Q3(Y)
        LTM_Q2 = Q3(Y-1) + Q4(Y-1) + Q1(Y) + Q2(Y)
        LTM_Q1 = Q2(Y-1) + Q3(Y-1) + Q4(Y-1) + Q1(Y)

    Skips when an explicit LTM row already exists for the same
    (metric, segment, year) — never overwrite a real number."""
    # Re-index by (metric_name, segment) so we can walk across years.
    by_pair: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}
    for (name, seg, year), shape_rows in groups.items():
        pair = (name, seg)
        for shape, row in shape_rows.items():
            by_pair.setdefault(pair, {})[(year, shape)] = row

    # Existing LTM rows aren't captured by `groups` (the main regex
    # excludes LTM since within-year derivation doesn't apply). Walk
    # the original rows so we can refuse to overwrite a real LTM.
    existing_ltm: set[tuple[str, str, str]] = set()
    for r in rows:
        name = r.get("metric_name", "")
        if not name:
            continue
        if skip_stocks and is_stock_metric(name):
            continue
        seg = (r.get("segment") or "") or ""
        label = (r.get("period_label") or "").upper()
        m = re.match(r"^(\d{4})_LTM$", label)
        if m:
            existing_ltm.add((name, seg, m.group(1)))

    derived: list[dict] = []
    for (name, seg), cells in by_pair.items():
        years = sorted({y for (y, _s) in cells.keys()})
        for year in years:
            # Skip if a real or already-derived LTM exists for this year.
            if (year, "LTM") in cells or (name, seg, year) in existing_ltm:
                continue
            try:
                year_int = int(year)
            except ValueError:
                continue
            prev_year = str(year_int - 1)

            # Try LTM-as-of-Q3 first (most common analyst use), then Q2, Q1.
            for as_of in ("Q3", "Q2", "Q1"):
                ingredients_keys: list[tuple[str, str]] = []
                if as_of == "Q3":
                    ingredients_keys = [
                        (prev_year, "Q4"), (year, "Q1"), (year, "Q2"), (year, "Q3"),
                    ]
                elif as_of == "Q2":
                    ingredients_keys = [
                        (prev_year, "Q3"), (prev_year, "Q4"), (year, "Q1"), (year, "Q2"),
                    ]
                else:  # Q1
                    ingredients_keys = [
                        (prev_year, "Q2"), (prev_year, "Q3"), (prev_year, "Q4"), (year, "Q1"),
                    ]

                inputs: list[dict] = []
                ok = True
                for k in ingredients_keys:
                    r = cells.get(k)
                    if r is None or r.get("metric_value") is None:
                        ok = False
                        break
                    inputs.append(r)
                if not ok:
                    continue

                try:
                    value = sum(float(r["metric_value"]) for r in inputs)
                except (TypeError, ValueError):
                    continue

                template = inputs[0]
                formula = " + ".join(f"{k[0]}_{k[1]}" for k in ingredients_keys)
                input_conf = min(float(r.get("confidence") or 0.9) for r in inputs)
                derived_row = {
                    "metric_name":      name,
                    "metric_value":     value,
                    "metric_text":      f"{value} (derived LTM)",
                    "unit":             template.get("unit"),
                    "segment":          seg or None,
                    "period_label":     f"{year}_LTM",
                    "period_frequency": "LTM",
                    "source_snippet":   f"derived: LTM as of {year}_{as_of} = {formula}",
                    "confidence":       round(input_conf * confidence_penalty, 3),
                    "is_derived":       True,
                }
                derived.append(derived_row)
                cells[(year, "LTM")] = derived_row
                # One LTM per year — the most-recent as-of we can reach.
                break
    return derived
