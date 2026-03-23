"""
Company CRUD endpoints (§8).
"""

import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db, get_company_or_404
from apps.api.models import Company, ThesisVersion
from schemas import CompanyCreate, CompanyOut, CompanyUpdate, ThesisCreate, ThesisOut

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=list[CompanyOut])
async def list_companies(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).order_by(Company.ticker))
    return result.scalars().all()


def _clean_ticker(raw: str) -> str:
    """Clean ticker: uppercase and trim outer whitespace. Keeps exchange suffix (e.g. 'ENI IM')."""
    return raw.strip().upper()


@router.get("/{ticker}", response_model=CompanyOut)
async def get_company(ticker: str, db: AsyncSession = Depends(get_db)):
    clean = _clean_ticker(ticker)
    result = await db.execute(select(Company).where(Company.ticker == clean))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    return company


@router.post("", response_model=CompanyOut, status_code=201)
async def create_company(body: CompanyCreate, db: AsyncSession = Depends(get_db)):
    company = Company(id=uuid.uuid4(), **body.model_dump())
    company.ticker = _clean_ticker(company.ticker)
    db.add(company)
    await db.commit()
    await db.refresh(company)
    return company


@router.patch("/{ticker}", response_model=CompanyOut)
async def update_company(ticker: str, body: CompanyUpdate, db: AsyncSession = Depends(get_db)):
    clean = _clean_ticker(ticker)
    result = await db.execute(select(Company).where(Company.ticker == clean))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(company, field, value)
    await db.commit()
    await db.refresh(company)
    return company


@router.delete("/{ticker}", status_code=204)
async def delete_company(ticker: str, db: AsyncSession = Depends(get_db)):
    """Delete a company and all associated data."""
    clean = _clean_ticker(ticker)
    result = await db.execute(select(Company).where(Company.ticker == clean))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    await db.delete(company)
    await db.commit()
    return None


# ─────────────────────────────────────────────────────────────────
# Thesis
# ─────────────────────────────────────────────────────────────────
@router.post("/{ticker}/thesis", response_model=ThesisOut, status_code=201)
async def create_thesis(ticker: str, body: ThesisCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Deactivate existing active theses
    existing = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id, ThesisVersion.active == True)
    )
    for tv in existing.scalars().all():
        tv.active = False

    thesis = ThesisVersion(
        id=uuid.uuid4(),
        company_id=company.id,
        **body.model_dump(),
        active=True,
    )
    db.add(thesis)
    await db.commit()
    await db.refresh(thesis)
    return thesis


@router.get("/{ticker}/thesis", response_model=list[ThesisOut])
async def list_theses(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    theses = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id).order_by(ThesisVersion.thesis_date.desc())
    )
    return theses.scalars().all()


@router.post("/{ticker}/seed-thesis", status_code=201)
async def seed_heineken_thesis(ticker: str, db: AsyncSession = Depends(get_db)):
    """One-click seed of the Heineken pilot thesis."""
    result = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    existing = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id, ThesisVersion.active == True)
    )
    if existing.scalar_one_or_none():
        return {"status": "thesis already exists"}
    thesis = ThesisVersion(
        id=uuid.uuid4(), company_id=company.id, thesis_date=date(2025, 12, 1),
        core_thesis="Heineken is a premium global brewer with pricing power and volume recovery tailwinds in emerging markets.",
        key_risks="1. Consumer downtrading in Europe\n2. FX headwinds in Africa\n3. Regulatory risk",
        valuation_framework="DCF with 7.5% WACC, cross-check EV/EBITDA 10-11x and FCF yield 6%+.",
        active=True,
    )
    db.add(thesis)
    await db.commit()
    return {"status": "thesis seeded", "thesis_id": str(thesis.id)}


@router.post("/{ticker}/generate-thesis", status_code=200)
async def generate_thesis(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Use the LLM to auto-generate an investment thesis based on
    extracted data already in the system, plus web knowledge.
    """
    from services.llm_client import call_llm_json

    result = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Gather any extracted metrics for context
    from apps.api.models import ExtractedMetric
    from sqlalchemy import func
    metrics_q = await db.execute(
        select(ExtractedMetric)
        .where(ExtractedMetric.company_id == company.id, ExtractedMetric.confidence >= 0.9)
        .order_by(ExtractedMetric.created_at.desc())
        .limit(50)
    )
    metrics = metrics_q.scalars().all()
    metrics_text = "\n".join(
        f"- {m.metric_name}: {m.metric_value} {m.unit or ''} ({m.period_label})"
        for m in metrics
    ) if metrics else "No extracted data available yet."

    prompt = f"""\
You are a senior buy-side equity analyst. Generate a structured investment thesis
for the following company based on your knowledge and any data provided.

COMPANY: {company.name} ({company.ticker})
SECTOR: {company.sector or 'Unknown'}
COUNTRY: {company.country or 'Unknown'}

AVAILABLE DATA FROM OUR SYSTEM:
{metrics_text}

Generate a comprehensive investment thesis. Be specific, opinionated, and actionable.
This is for an internal investment memo, not a marketing document.

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "core_thesis": "<3-4 sentences on why this is an attractive/unattractive investment>",
  "variant_perception": "<where does the market disagree with our view, and why are we right>",
  "key_risks": "<4-6 specific risks, numbered>",
  "debate_points": "<3-5 key debates around the stock, numbered>",
  "capital_allocation_view": "<how management allocates capital, and whether we agree>",
  "valuation_framework": "<preferred valuation approach with specific parameters>"
}}
"""
    try:
        thesis_data = call_llm_json(prompt, max_tokens=4096)
        return thesis_data
    except Exception as e:
        raise HTTPException(500, f"Thesis generation failed: {str(e)[:200]}")

@router.post("/{ticker}/llm-update-thesis")
async def llm_update_thesis(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Use the LLM to populate:
      - IC Summary fields (recommendation, conviction, catalyst, variant_perception)
      - What Would Make Us Wrong
      - Disconfirming Evidence
      - Positive / Negative Surprises

    Draws from: existing thesis + most recent 2 periods of analysis output.
    Saves results directly to the active ThesisVersion.
    """
    import json as _json
    from services.llm_client import call_llm_json
    from apps.api.models import ResearchOutput

    # Get company
    co_q = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = co_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Get active thesis
    th_q = await db.execute(
        select(ThesisVersion).where(
            ThesisVersion.company_id == company.id,
            ThesisVersion.active == True,
        ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = th_q.scalar_one_or_none()
    if not thesis:
        raise HTTPException(400, "No active thesis found. Create a thesis first.")

    # Get most recent 2 periods of analysis output
    outputs_q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == company.id,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(ResearchOutput.created_at.desc()).limit(2)
    )
    outputs = outputs_q.scalars().all()

    # Build context from recent outputs
    recent_context = ""
    for i, output in enumerate(outputs):
        if not output.content_json:
            continue
        try:
            data = _json.loads(output.content_json)
            period = output.period_label or f"Period {i+1}"
            syn = data.get("synthesis") or data.get("briefing") or {}
            parts = [f"=== {period} ==="]
            for key in ["headline", "what_happened", "management_message",
                        "thesis_status", "thesis_impact", "bottom_line",
                        "risks", "what_changed"]:
                if syn.get(key):
                    parts.append(f"{key.replace('_', ' ').title()}: {syn[key]}")
            surprises = data.get("surprises", [])
            if surprises:
                parts.append("Surprises: " + "; ".join(
                    f"{'↑' if s.get('direction') == 'positive' else '↓'} "
                    f"{s.get('metric_or_topic', '')}: {s.get('description', '')}"
                    for s in surprises[:5]
                ))
            recent_context += "\n".join(parts) + "\n\n"
        except Exception:
            pass

    if not recent_context:
        raise HTTPException(400, "No analysis output found. Run at least one analysis first.")

    prompt = f"""You are a senior buy-side equity analyst maintaining the investment file for {company.name} ({ticker}).

CURRENT INVESTMENT THESIS:
{thesis.core_thesis or "Not yet written."}

KEY RISKS ON FILE:
{thesis.key_risks or "None recorded."}

VALUATION FRAMEWORK:
{thesis.valuation_framework or "None recorded."}

RECENT RESULTS (most recent first):
{recent_context}

EXISTING IC SUMMARY FIELDS (leave unchanged if still accurate):
- Recommendation: {thesis.recommendation or "not set"}
- Conviction: {thesis.conviction or "not set"}
- Catalyst: {thesis.catalyst or "not set"}
- Variant Perception: {thesis.variant_perception or "not set"}

TASK:
Based on the thesis and recent results, populate the following fields.
Be specific, opinionated, and grounded in the evidence above.
Do NOT invent facts or numbers not present in the data above.

Respond ONLY with a JSON object. No preamble, no markdown fences.

{{
  "recommendation": "Buy" | "Hold" | "Trim" | "Exit" | "Under Review",
  "conviction": "High" | "Medium" | "Low",
  "catalyst": "<specific upcoming catalyst, e.g. Q2 2025 results — organic growth inflection>",
  "variant_perception": "<what we see that consensus does not — max 1 sentence>",
  "what_would_make_us_wrong": "<specific falsifiable conditions that would invalidate the thesis — 2-4 bullet points>",
  "disconfirming_evidence": "<data or events from recent results that have weakened the case — be honest>",
  "positive_surprises": "<positive developments vs initial thesis expectations>",
  "negative_surprises": "<negative developments or disappointments vs initial thesis expectations>"
}}"""

    try:
        result = call_llm_json(prompt, max_tokens=2000)
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {str(e)[:200]}")

    # Save each field to the thesis
    updatable_fields = [
        "recommendation", "conviction", "catalyst", "variant_perception",
        "what_would_make_us_wrong", "disconfirming_evidence",
        "positive_surprises", "negative_surprises",
    ]
    updated = []
    for field in updatable_fields:
        val = result.get(field)
        if val and isinstance(val, str) and val.strip():
            setattr(thesis, field, val.strip())
            updated.append(field)

    await db.commit()

    return {
        "status": "updated",
        "fields_updated": updated,
        "ticker": ticker,
    }


@router.patch("/{ticker}/thesis")
async def patch_thesis_field(ticker: str, body: dict, db: AsyncSession = Depends(get_db)):
    """Patch a single field on the active thesis. Used by inline editable fields in the UI."""
    co_q = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = co_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    th_q = await db.execute(
        select(ThesisVersion).where(
            ThesisVersion.company_id == company.id,
            ThesisVersion.active == True,
        ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = th_q.scalar_one_or_none()
    if not thesis:
        raise HTTPException(404, "No active thesis found")

    allowed_fields = {
        "core_thesis", "variant_perception", "key_risks", "debate_points",
        "capital_allocation_view", "valuation_framework",
        "recommendation", "catalyst", "conviction",
        "what_would_make_us_wrong", "disconfirming_evidence",
        "positive_surprises", "negative_surprises",
    }
    field = body.get("field")
    value = body.get("value")
    if field not in allowed_fields:
        raise HTTPException(400, f"Field '{field}' is not patchable")

    setattr(thesis, field, value)
    await db.commit()
    return {"status": "saved", "field": field}

