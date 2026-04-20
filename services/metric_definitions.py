"""
Metric Validation Registry — per-metric reasonable-range and
denominator rules.

Two layers:
  - Universal (sector-agnostic) validators — EPS, Revenue, Operating
    Margin, FCF, Debt/Equity. Shared across all companies.
  - Sector-specific validators — routed via sector_kpi_config.py
    (banks: NIM, CET1; insurance: combined ratio; REITs: LTV; etc.)

Public surface:
  validate_metric(name, value, unit, sector, industry, denominator)
      → Optional[ValidationIssue]

Returns None if the metric passes. Returns a ValidationIssue with
severity info/warning/critical otherwise.

No LLM. Pure deterministic checks — fast to run, safe to invoke
from the reconciliation pre-flight gate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from services.sector_kpi_config import _match_subsector

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Issue structure
# ─────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """One flag raised by metric validation."""
    metric_name:       str
    value:             Optional[float]
    unit:              Optional[str]
    severity:          str                       # info | warning | critical
    rule_violated:     str
    suggested_check:   str
    sector:            Optional[str] = None
    industry:          Optional[str] = None
    extra:             dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metric_name":     self.metric_name,
            "value":           self.value,
            "unit":            self.unit,
            "severity":        self.severity,
            "rule_violated":   self.rule_violated,
            "suggested_check": self.suggested_check,
            "sector":          self.sector,
            "industry":        self.industry,
            "extra":           self.extra,
        }


# ─────────────────────────────────────────────────────────────────
# Universal validators — sector-agnostic rules
# ─────────────────────────────────────────────────────────────────

# Ranges are in the native extracted unit. For %-unit metrics we
# assume values are in percentage form (3.45) not decimal (0.0345),
# which matches how the extractor stores them today.
_UNIVERSAL_RULES: dict[str, dict] = {
    "eps": {
        # EPS can be genuinely any value; flag only truly absurd magnitudes.
        "reasonable_range": (-1000.0, 1000.0),
        "unit_hint": "currency/share",
    },
    "diluted eps": {
        "reasonable_range": (-1000.0, 1000.0),
        "unit_hint": "currency/share",
    },
    "operating margin": {
        "reasonable_range": (-50.0, 60.0),  # in %
        "unit_hint": "%",
    },
    "ebitda margin": {
        "reasonable_range": (-30.0, 70.0),
        "unit_hint": "%",
    },
    "gross margin": {
        "reasonable_range": (0.0, 95.0),
        "unit_hint": "%",
    },
    "net margin": {
        "reasonable_range": (-50.0, 50.0),
        "unit_hint": "%",
    },
    "debt/equity": {
        "reasonable_range": (0.0, 10.0),
        "unit_hint": "x",
    },
    "net debt/ebitda": {
        "reasonable_range": (-5.0, 15.0),   # negative = net cash
        "unit_hint": "x",
    },
    "roe": {
        "reasonable_range": (-40.0, 60.0),
        "unit_hint": "%",
    },
    "roic": {
        "reasonable_range": (-30.0, 60.0),
        "unit_hint": "%",
    },
    "revenue growth": {
        "reasonable_range": (-50.0, 200.0),
        "unit_hint": "%",
    },
    "organic revenue growth": {
        "reasonable_range": (-30.0, 50.0),
        "unit_hint": "%",
    },
}


# ─────────────────────────────────────────────────────────────────
# Name matching
# ─────────────────────────────────────────────────────────────────

def _clean_name(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    # Paren contents become spaces so "EPS (Diluted)" → "eps diluted"
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _match_rule(name: str, rules: dict) -> Optional[tuple[str, dict]]:
    """Return (matched_key, rule_dict) or None.

    Both the input name and each rule key are cleaned before matching,
    so "Charge_Off_Rate" matches a rule keyed "charge_off_rate" or
    "charge off rate". Match is exact (cleaned) then whole-word
    substring (so "operating margin" matches "core operating margin").
    """
    cleaned = _clean_name(name)
    if not cleaned:
        return None
    # Pre-clean rule keys once per call
    cleaned_rules = [(_clean_name(k), k, rule) for k, rule in rules.items()]
    # Exact cleaned match first
    for ck, original_key, rule in cleaned_rules:
        if ck == cleaned:
            return original_key, rule
    # Whole-word substring match
    for ck, original_key, rule in cleaned_rules:
        if ck and re.search(rf"\b{re.escape(ck)}\b", cleaned):
            return original_key, rule
    return None


# ─────────────────────────────────────────────────────────────────
# Sector routing
# ─────────────────────────────────────────────────────────────────

def _sector_rules(sector: str, industry: str) -> dict:
    """Pull kpi_validation dict for the subsector, or {} if none."""
    cfg = _match_subsector(sector or "", industry or "")
    if not cfg:
        return {}
    return cfg.get("kpi_validation", {}) or {}


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_metric(
    name: str,
    value: Optional[float],
    unit: Optional[str] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    denominator: Optional[str] = None,
) -> Optional[ValidationIssue]:
    """Check a single metric against universal + sector-specific rules.

    Returns None (passes) or a ValidationIssue describing the
    problem. Sector-specific rules take precedence over universal
    when both match (sector rules are tuned for real companies, not
    generic bounds).
    """
    if not name:
        return None
    if value is None:
        return None

    try:
        val = float(value)
    except (TypeError, ValueError):
        return None

    sec_rules = _sector_rules(sector or "", industry or "")

    # Sector-specific takes precedence
    match = _match_rule(name, sec_rules)
    if match is None:
        match = _match_rule(name, _UNIVERSAL_RULES)
        source = "universal"
    else:
        source = "sector"

    if match is None:
        return None

    matched_key, rule = match

    # Range check
    rng = rule.get("reasonable_range")
    if rng and isinstance(rng, (tuple, list)) and len(rng) == 2:
        low, high = rng
        if val < low or val > high:
            # Severity: how far outside the band?
            span = max(high - low, 1e-9)
            if val < low:
                margin = (low - val) / span
            else:
                margin = (val - high) / span
            severity = "critical" if margin > 0.5 else "warning"
            return ValidationIssue(
                metric_name=name,
                value=val,
                unit=unit,
                severity=severity,
                rule_violated=f"out_of_range ({source}:{matched_key})",
                suggested_check=(
                    f"{matched_key} = {val} is outside the typical range "
                    f"[{low}, {high}] for this sector — verify scale, unit, "
                    f"or extraction accuracy."
                ),
                sector=sector,
                industry=industry,
                extra={"range": [low, high], "margin": round(margin, 3)},
            )

    # Denominator check (only sector rules carry this)
    if denominator:
        valid_denoms = rule.get("valid_denominators")
        invalid_denoms = rule.get("invalid_denominators") or []
        denom_clean = _clean_name(denominator).replace(" ", "_")
        if denom_clean in {_clean_name(d).replace(" ", "_") for d in invalid_denoms}:
            return ValidationIssue(
                metric_name=name,
                value=val,
                unit=unit,
                severity="critical",
                rule_violated=f"invalid_denominator ({source}:{matched_key})",
                suggested_check=(
                    f"{matched_key} computed with denominator '{denominator}' "
                    f"— not one of the accepted denominators "
                    f"({valid_denoms}). This is almost certainly a "
                    f"fabricated or mis-constructed metric."
                ),
                sector=sector,
                industry=industry,
                extra={"denominator": denominator,
                       "valid_denominators": valid_denoms},
            )
        if valid_denoms and denom_clean not in {_clean_name(d).replace(" ", "_") for d in valid_denoms}:
            return ValidationIssue(
                metric_name=name,
                value=val,
                unit=unit,
                severity="warning",
                rule_violated=f"unknown_denominator ({source}:{matched_key})",
                suggested_check=(
                    f"{matched_key} denominator '{denominator}' is not in "
                    f"the known-valid list {valid_denoms} — verify this is "
                    f"not a misconstructed ratio."
                ),
                sector=sector,
                industry=industry,
                extra={"denominator": denominator,
                       "valid_denominators": valid_denoms},
            )

    return None


def validate_metrics_batch(
    metrics: list[dict],
    sector: Optional[str] = None,
    industry: Optional[str] = None,
) -> list[ValidationIssue]:
    """Validate a list of {metric_name, metric_value, unit, denominator} dicts.

    Returns only failures — callers can filter by severity.
    """
    issues: list[ValidationIssue] = []
    for m in metrics or []:
        issue = validate_metric(
            name=m.get("metric_name", ""),
            value=m.get("metric_value"),
            unit=m.get("unit"),
            sector=sector,
            industry=industry,
            denominator=m.get("denominator"),
        )
        if issue is not None:
            issues.append(issue)
    return issues
