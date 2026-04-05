"""
KPI Extractor — runs a Haiku pass on briefing output to extract
standardised KPIs and stores them in the kpi_actuals table.

Called automatically after synthesis completes in the background processor.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import AsyncSessionLocal
from services.llm_client import call_llm_json_async

logger = logging.getLogger(__name__)

# The 9 KPIs we extract from every briefing
KPI_DEFINITIONS = [
    {"name": "revenue_m", "type": "float", "description": "Total revenue in millions (local currency)"},
    {"name": "revenue_growth_yoy_pct", "type": "float", "description": "Year-over-year revenue growth percentage"},
    {"name": "ebit_margin_pct", "type": "float", "description": "EBIT margin as a percentage"},
    {"name": "net_income_m", "type": "float", "description": "Net income in millions (local currency)"},
    {"name": "eps", "type": "float", "description": "Earnings per share"},
    {"name": "eps_vs_consensus_pct", "type": "float", "description": "EPS beat/miss vs consensus as percentage (positive = beat)"},
    {"name": "guidance_raised", "type": "bool", "description": "Was full-year guidance raised? true/false/null if not mentioned"},
    {"name": "guidance_revenue_midpoint_m", "type": "float", "description": "Midpoint of revenue guidance range in millions"},
    {"name": "management_tone", "type": "text", "description": "One-sentence summary of management tone (e.g. 'cautiously optimistic', 'defensive')"},
]

EXTRACTION_PROMPT = """You are an investment research assistant. Extract the following KPIs from this earnings briefing.

Return a JSON object with these keys. Use null if the value is not mentioned or cannot be determined.
- revenue_m (number, in millions)
- revenue_growth_yoy_pct (number, percentage)
- ebit_margin_pct (number, percentage)
- net_income_m (number, in millions)
- eps (number)
- eps_vs_consensus_pct (number, percentage — positive means beat)
- guidance_raised (boolean or null)
- guidance_revenue_midpoint_m (number, in millions, or null)
- management_tone (string, one sentence, or null)

BRIEFING:
{briefing_text}

Return ONLY the JSON object, no markdown fences."""


def _parse_period(period_label: str) -> tuple[str, int]:
    """Parse '2025_Q2' into ('Q2', 2025)."""
    parts = period_label.split("_")
    if len(parts) == 2:
        try:
            return parts[1], int(parts[0])
        except (ValueError, IndexError):
            pass
    return period_label, 0


def _extract_briefing_text(output: dict) -> str | None:
    """Pull the briefing text from the pipeline output dict."""
    # Batch pipeline stores briefing under "synthesis" key
    synthesis = output.get("synthesis")
    if isinstance(synthesis, dict) and not synthesis.get("error"):
        # Try common fields in the synthesis output
        for key in ("executive_summary", "briefing", "summary", "analysis"):
            if key in synthesis:
                return str(synthesis[key])
        # Fall back to the entire synthesis as text
        return json.dumps(synthesis, default=str)

    # Single pipeline stores briefing under "briefing" key
    briefing = output.get("briefing")
    if isinstance(briefing, dict) and not briefing.get("error"):
        for key in ("executive_summary", "briefing", "summary", "analysis"):
            if key in briefing:
                return str(briefing[key])
        return json.dumps(briefing, default=str)

    return None


async def extract_kpis_from_briefing(
    company_id: uuid.UUID,
    period_label: str,
    output: dict,
    doc_id: uuid.UUID | None = None,
) -> dict:
    """Extract KPIs from a pipeline output and store in kpi_actuals.

    Args:
        company_id: The company UUID
        period_label: e.g. "2025_Q2"
        output: The full pipeline output dict (contains briefing/synthesis)
        doc_id: Optional source document ID

    Returns:
        Dict with extraction results or error info.
    """
    briefing_text = _extract_briefing_text(output)
    if not briefing_text:
        logger.info("[KPI] No briefing text found for %s/%s — skipping KPI extraction", company_id, period_label)
        return {"status": "skipped", "reason": "no_briefing_text"}

    # Truncate to avoid huge prompts (Haiku context is smaller)
    if len(briefing_text) > 12000:
        briefing_text = briefing_text[:12000] + "\n... [truncated]"

    try:
        prompt = EXTRACTION_PROMPT.format(briefing_text=briefing_text)
        kpi_data = await call_llm_json_async(
            prompt,
            max_tokens=1024,
            model="claude-3-5-haiku-20241022",
        )
    except Exception as e:
        logger.error("[KPI] LLM extraction failed for %s/%s: %s", company_id, period_label, e)
        return {"status": "error", "reason": str(e)[:200]}

    if not isinstance(kpi_data, dict):
        logger.warning("[KPI] LLM returned non-dict for %s/%s: %s", company_id, period_label, type(kpi_data))
        return {"status": "error", "reason": "unexpected_response_type"}

    period, year = _parse_period(period_label)
    now = datetime.now(timezone.utc)
    saved = 0

    try:
        async with AsyncSessionLocal() as db:
            for kpi_def in KPI_DEFINITIONS:
                kpi_name = kpi_def["name"]
                raw_val = kpi_data.get(kpi_name)

                value_float = None
                value_bool = None
                value_text = None

                if raw_val is None:
                    continue  # skip nulls

                if kpi_def["type"] == "float":
                    try:
                        value_float = float(raw_val)
                    except (TypeError, ValueError):
                        continue
                elif kpi_def["type"] == "bool":
                    if isinstance(raw_val, bool):
                        value_bool = raw_val
                    elif isinstance(raw_val, str):
                        value_bool = raw_val.lower() in ("true", "yes", "1")
                    else:
                        continue
                elif kpi_def["type"] == "text":
                    value_text = str(raw_val)

                # Upsert via DELETE + INSERT (asyncpg doesn't support ON CONFLICT easily via raw SQL)
                await db.execute(text(
                    "DELETE FROM kpi_actuals WHERE company_id = :cid AND period = :period AND year = :year AND kpi_name = :kpi"
                ), {"cid": company_id, "period": period, "year": year, "kpi": kpi_name})

                await db.execute(text(
                    """INSERT INTO kpi_actuals (id, company_id, period, year, kpi_name, value, value_bool, value_text, source_doc_id, extracted_at)
                    VALUES (:id, :cid, :period, :year, :kpi, :val, :vbool, :vtext, :doc_id, :now)"""
                ), {
                    "id": uuid.uuid4(),
                    "cid": company_id,
                    "period": period,
                    "year": year,
                    "kpi": kpi_name,
                    "val": value_float,
                    "vbool": value_bool,
                    "vtext": value_text,
                    "doc_id": doc_id,
                    "now": now,
                })
                saved += 1

            await db.commit()

        logger.info("[KPI] Extracted %d KPIs for %s/%s", saved, company_id, period_label)
        return {"status": "ok", "kpis_saved": saved, "data": kpi_data}

    except Exception as e:
        logger.error("[KPI] DB save failed for %s/%s: %s", company_id, period_label, e)
        return {"status": "error", "reason": str(e)[:200]}
