"""
Prior quarter comparison — flags anomalies vs previous period.
Catches both hallucinations (impossible jumps) and missed extractions.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Thresholds for flagging
_GROWTH_ALERT = 2.0     # >200% change = suspicious
_DECLINE_ALERT = -0.8   # >80% decline = suspicious


def _safe_float(val) -> Optional[float]:
    """Try to convert a value to float, return None if impossible."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def compare_to_prior(
    current_items: list[dict],
    prior_items: list[dict],
    current_period: str,
    prior_period: str,
) -> dict:
    """
    Compare current extraction against prior period.

    Returns:
        {
            "anomalies": [{"metric", "current", "prior", "change_pct", "severity"}],
            "missing_in_current": [{"metric", "prior_value"}],
            "new_in_current": [{"metric", "current_value"}],
        }
    """
    # Build dicts of metric_name -> numeric value for both periods
    def _build_metric_dict(items: list[dict]) -> dict[str, float]:
        result = {}
        for item in items:
            name = (item.get("metric_name") or "").strip().lower()
            if not name:
                continue
            val = _safe_float(item.get("metric_value"))
            if val is not None:
                # Keep the first occurrence (highest confidence typically comes first)
                if name not in result:
                    result[name] = val
        return result

    current_dict = _build_metric_dict(current_items)
    prior_dict = _build_metric_dict(prior_items)

    current_keys = set(current_dict.keys())
    prior_keys = set(prior_dict.keys())

    anomalies = []
    missing_in_current = []
    new_in_current = []

    # Metrics present in both periods — check for anomalous changes
    for metric in current_keys & prior_keys:
        current_val = current_dict[metric]
        prior_val = prior_dict[metric]

        if prior_val == 0:
            # Can't compute % change from zero; flag large absolute jumps
            if abs(current_val) > 0:
                change_pct = None  # undefined
            else:
                continue  # 0 -> 0 is fine
        else:
            change_pct = (current_val - prior_val) / abs(prior_val)

        if change_pct is not None and (change_pct > _GROWTH_ALERT or change_pct < _DECLINE_ALERT):
            severity = "high" if (change_pct is not None and (change_pct > 5.0 or change_pct < -0.95)) else "medium"
            anomalies.append({
                "metric": metric,
                "current": current_val,
                "prior": prior_val,
                "change_pct": round(change_pct * 100, 1) if change_pct is not None else None,
                "severity": severity,
            })

    # Metrics in prior but not current — potential missed extractions
    for metric in prior_keys - current_keys:
        missing_in_current.append({
            "metric": metric,
            "prior_value": prior_dict[metric],
        })

    # Metrics in current but not prior — new items (not necessarily bad)
    for metric in current_keys - prior_keys:
        new_in_current.append({
            "metric": metric,
            "current_value": current_dict[metric],
        })

    return {
        "anomalies": anomalies,
        "missing_in_current": missing_in_current,
        "new_in_current": new_in_current,
    }
