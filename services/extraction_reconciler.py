"""
Extraction reconciler — cross-checks extracted financial data for internal consistency.

Checks:
1. Q4 + prior quarters ≈ FY for Revenue and Net Income (±2%)
2. Segment revenues sum to consolidated revenue (±2%)
3. Balance sheet equation: total assets ≈ total liabilities + equity (±1%)
4. Net income on P&L ≈ net income on CF statement (±2%)
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Metric name aliases (lowercase) for flexible matching
_REVENUE_ALIASES = {"revenue", "net sales", "total revenue", "net revenue", "sales"}
_NET_INCOME_ALIASES = {"net income", "net profit", "net loss", "net earnings", "profit for the period"}
_TOTAL_ASSETS_ALIASES = {"total assets"}
_TOTAL_LIABILITIES_ALIASES = {"total liabilities"}
_EQUITY_ALIASES = {
    "shareholders equity", "stockholders equity", "total equity",
    "shareholders' equity", "stockholders' equity", "equity",
}


def _normalise_key(key: str) -> str:
    """Lowercase and strip a metric key for matching."""
    return re.sub(r"[_\-]", " ", key.lower().strip())


def get_metric(
    results: dict,
    statement_type: str,
    period_contains: str,
    metric_name: str,
) -> float | None:
    """
    Retrieve a metric value from extraction results.

    Args:
        results: full extraction dict (from extract_all_statements)
        statement_type: key in results dict, e.g. "income_statements"
        period_contains: substring that must appear in the period label, e.g. "FY 2025"
        metric_name: the metric to look for (tries aliases)

    Returns:
        float value or None if not found
    """
    entries = results.get(statement_type, [])
    target_lower = period_contains.lower()

    for entry in entries:
        period = entry.get("period", "").lower()
        if target_lower not in period:
            continue

        data = entry.get("data", {})
        if not isinstance(data, dict):
            continue

        # Direct lookup
        for key, value in data.items():
            normalised = _normalise_key(key)
            if normalised == _normalise_key(metric_name):
                if isinstance(value, (int, float)):
                    return float(value)
            # Check aliases
            metric_lower = _normalise_key(metric_name)
            if metric_lower in _REVENUE_ALIASES and normalised in _REVENUE_ALIASES:
                if isinstance(value, (int, float)):
                    return float(value)
            if metric_lower in _NET_INCOME_ALIASES and normalised in _NET_INCOME_ALIASES:
                if isinstance(value, (int, float)):
                    return float(value)
            if metric_lower in _TOTAL_ASSETS_ALIASES and normalised in _TOTAL_ASSETS_ALIASES:
                if isinstance(value, (int, float)):
                    return float(value)
            if metric_lower in _TOTAL_LIABILITIES_ALIASES and normalised in _TOTAL_LIABILITIES_ALIASES:
                if isinstance(value, (int, float)):
                    return float(value)
            if metric_lower in _EQUITY_ALIASES and normalised in _EQUITY_ALIASES:
                if isinstance(value, (int, float)):
                    return float(value)

    return None


def get_all_segment_metrics(results: dict, metric_name: str) -> dict[str, float]:
    """
    Collect a metric across all segment entries.
    Returns: {"segment_name": value, ...}
    """
    segments = results.get("segments", [])
    out = {}
    metric_lower = _normalise_key(metric_name)

    for entry in segments:
        segment_name = entry.get("segment") or "unknown"
        data = entry.get("data", {})
        if not isinstance(data, dict):
            continue

        for key, value in data.items():
            normalised = _normalise_key(key)
            if normalised == metric_lower:
                if isinstance(value, (int, float)):
                    out[segment_name] = float(value)
                    break
            # Check revenue aliases
            if metric_lower in _REVENUE_ALIASES and normalised in _REVENUE_ALIASES:
                if isinstance(value, (int, float)):
                    out[segment_name] = float(value)
                    break

    return out


def _pct_diff(a: float, b: float) -> float:
    """Absolute percentage difference between two values."""
    if b == 0:
        return 0.0 if a == 0 else float("inf")
    return abs(a - b) / abs(b)


def _check_quarterly_sum_vs_fy(
    results: dict,
    year: int,
    metric_name: str,
    tolerance: float = 0.02,
) -> dict | None:
    """Check that Q1+Q2+Q3+Q4 ≈ FY for a given metric and year."""
    fy_val = get_metric(results, "income_statements", f"FY {year}", metric_name)
    if fy_val is None:
        return None

    q_sum = 0.0
    quarters_found = 0
    for q in ("Q1", "Q2", "Q3", "Q4"):
        q_val = get_metric(results, "income_statements", f"{q} {year}", metric_name)
        if q_val is not None:
            q_sum += q_val
            quarters_found += 1

    if quarters_found < 2:
        return None  # Not enough data to check

    diff = _pct_diff(q_sum, fy_val)
    if diff > tolerance:
        return {
            "check": f"quarterly_sum_vs_fy_{metric_name.lower().replace(' ', '_')}",
            "severity": "critical" if diff > 0.10 else "high",
            "detail": {
                "year": year,
                "metric": metric_name,
                "fy_value": fy_val,
                "quarterly_sum": q_sum,
                "quarters_found": quarters_found,
                "pct_diff": round(diff * 100, 2),
            },
        }
    return None


def _check_segment_sum(
    results: dict,
    metric_name: str,
    tolerance: float = 0.02,
) -> dict | None:
    """Check that segment values sum to consolidated value."""
    seg_values = get_all_segment_metrics(results, metric_name)
    if len(seg_values) < 2:
        return None

    seg_sum = sum(seg_values.values())

    # Find consolidated value — try income_statements for revenue
    consolidated = None
    for stmt_type in ("income_statements", "cash_flows"):
        for entry in results.get(stmt_type, []):
            if entry.get("segment") is not None:
                continue  # skip segment entries
            data = entry.get("data", {})
            if not isinstance(data, dict):
                continue
            for key, value in data.items():
                if _normalise_key(key) in _REVENUE_ALIASES and _normalise_key(metric_name) in _REVENUE_ALIASES:
                    if isinstance(value, (int, float)):
                        consolidated = float(value)
                        break
            if consolidated is not None:
                break

    if consolidated is None:
        return None

    diff = _pct_diff(seg_sum, consolidated)
    if diff > tolerance:
        return {
            "check": f"segment_sum_{metric_name.lower().replace(' ', '_')}",
            "severity": "high",
            "detail": {
                "metric": metric_name,
                "consolidated": consolidated,
                "segment_sum": seg_sum,
                "segments": seg_values,
                "pct_diff": round(diff * 100, 2),
            },
        }
    return None


def _check_balance_sheet_equation(
    results: dict,
    tolerance: float = 0.01,
) -> list[dict]:
    """Check total assets ≈ total liabilities + equity for each BS entry."""
    issues = []
    for entry in results.get("balance_sheets", []):
        data = entry.get("data", {})
        if not isinstance(data, dict):
            continue

        assets = None
        liabilities = None
        equity = None

        for key, value in data.items():
            norm = _normalise_key(key)
            if not isinstance(value, (int, float)):
                continue
            if norm in _TOTAL_ASSETS_ALIASES:
                assets = float(value)
            elif norm in _TOTAL_LIABILITIES_ALIASES:
                liabilities = float(value)
            elif norm in _EQUITY_ALIASES:
                equity = float(value)

        if assets is not None and liabilities is not None and equity is not None:
            rhs = liabilities + equity
            diff = _pct_diff(assets, rhs)
            if diff > tolerance:
                issues.append({
                    "check": "balance_sheet_equation",
                    "severity": "critical",
                    "detail": {
                        "period": entry.get("period", "unknown"),
                        "total_assets": assets,
                        "total_liabilities": liabilities,
                        "equity": equity,
                        "liabilities_plus_equity": rhs,
                        "pct_diff": round(diff * 100, 2),
                    },
                })

    return issues


def _check_net_income_cross_statement(
    results: dict,
    tolerance: float = 0.02,
) -> list[dict]:
    """Check net income on P&L ≈ net income on CF statement."""
    issues = []

    # Collect all periods from income statements
    for is_entry in results.get("income_statements", []):
        period = is_entry.get("period", "")
        if not period:
            continue

        # Find net income in P&L
        pl_ni = None
        data = is_entry.get("data", {})
        if isinstance(data, dict):
            for key, value in data.items():
                if _normalise_key(key) in _NET_INCOME_ALIASES and isinstance(value, (int, float)):
                    pl_ni = float(value)
                    break

        if pl_ni is None:
            continue

        # Find matching net income in CF
        cf_ni = get_metric(results, "cash_flows", period, "net income")
        if cf_ni is None:
            continue

        diff = _pct_diff(pl_ni, cf_ni)
        if diff > tolerance:
            issues.append({
                "check": "net_income_pl_vs_cf",
                "severity": "high",
                "detail": {
                    "period": period,
                    "pl_net_income": pl_ni,
                    "cf_net_income": cf_ni,
                    "pct_diff": round(diff * 100, 2),
                },
            })

    return issues


def reconcile_extractions(all_results: dict) -> dict:
    """
    Cross-check extracted financial data for internal consistency.

    Returns:
        {
            "issues": [...],
            "passed": bool,
            "checks_run": int,
            "checks_passed": int,
        }
    """
    issues = []
    checks_run = 0
    checks_passed = 0

    # ── 1. Quarterly sum vs FY ──
    # Try to find years in the data
    years = set()
    for entry in all_results.get("income_statements", []):
        period = entry.get("period", "")
        m = re.search(r"(20\d{2})", period)
        if m:
            years.add(int(m.group(1)))

    for year in years:
        for metric in ("revenue", "net income"):
            result = _check_quarterly_sum_vs_fy(all_results, year, metric)
            checks_run += 1
            if result:
                issues.append(result)
            else:
                checks_passed += 1

    # ── 2. Segment sum vs consolidated ──
    for metric in ("revenue",):
        result = _check_segment_sum(all_results, metric)
        checks_run += 1
        if result:
            issues.append(result)
        else:
            checks_passed += 1

    # ── 3. Balance sheet equation ──
    bs_issues = _check_balance_sheet_equation(all_results)
    bs_count = max(len(all_results.get("balance_sheets", [])), 1)
    checks_run += bs_count
    if bs_issues:
        issues.extend(bs_issues)
        checks_passed += bs_count - len(bs_issues)
    else:
        checks_passed += bs_count

    # ── 4. Net income P&L vs CF ──
    ni_issues = _check_net_income_cross_statement(all_results)
    ni_count = max(len(all_results.get("income_statements", [])), 1)
    checks_run += ni_count
    if ni_issues:
        issues.extend(ni_issues)
        checks_passed += ni_count - len(ni_issues)
    else:
        checks_passed += ni_count

    passed = len(issues) == 0

    logger.info(
        "Reconciliation: %d checks run, %d passed, %d issues found",
        checks_run, checks_passed, len(issues),
    )

    return {
        "issues": issues,
        "passed": passed,
        "checks_run": checks_run,
        "checks_passed": checks_passed,
    }
