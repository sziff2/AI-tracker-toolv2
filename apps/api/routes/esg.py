"""
ESG Module Router — Environmental, Social, Governance data tracking.

Endpoints:
  GET  /companies/{ticker:path}/esg              — fetch stored ESG data
  PUT  /companies/{ticker:path}/esg              — upsert ESG data
  POST /companies/{ticker:path}/esg/ask          — ESG Research Assistant chat
  POST /companies/{ticker:path}/esg/analyse      — auto-generate summary + flag gaps
  GET  /esg/portfolio/pai-report            — aggregate PAI across all holdings
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, ESGData

logger = logging.getLogger(__name__)
router = APIRouter(tags=["esg"])


class ESGUpsertRequest(BaseModel):
    data: dict[str, Any]

class ESGAskRequest(BaseModel):
    question: str
    history: list[dict] = []

class ESGAnalyseResponse(BaseModel):
    summary: str
    suggested_fields: dict[str, str] = {}
    missing_fields: list[str] = []
    red_flags: list[str] = []


def _build_esg_context(data: dict, company: Company) -> str:
    filled = {k: v for k, v in data.items() if v not in (None, "")}
    if not filled:
        return "No ESG data has been entered yet."
    lines = [f"Company: {company.ticker} — {company.name} ({company.sector or 'N/A'})", ""]
    section_map = {
        "Environmental / PAI": [
            "ghgScope1", "ghgScope2", "ghgScope3", "ghgTotal", "carbonFootprint",
            "ghgIntensity", "fossilFuelPct", "nonRenewablePct",
            "biodiversity", "waterEmissions", "hazardousWaste", "sbti", "netZeroTarget",
        ],
        "Social / PAI": [
            "ungcViolations", "ungcProcesses", "genderPayGap", "ltir",
            "employeeTurnover", "unionisation", "humanRightsPolicy", "supplyChainAudit",
        ],
        "Governance / PAI": [
            "boardDiversity", "controversialWeapons", "ceoPayRatio",
            "independentDirectors", "auditQuality", "antiCorruption",
            "whistleblower", "dataPrivacy",
        ],
        "Ratings & Assessment": [
            "msciRating", "sustainalytics", "cdpScore", "issSoc", "issGov",
            "ftse4good", "djsi", "internalScore", "esgTrend", "redFlags", "esgNotes",
        ],
    }
    for section, keys in section_map.items():
        section_lines = [f"{k}: {filled[k]}" for k in keys if k in filled]
        if section_lines:
            lines.append(f"[{section}]")
            lines.extend(section_lines)
            lines.append("")
    return "\n".join(lines)


_ALL_IMPORTANT_KEYS = [
    "ghgScope1", "ghgScope2", "ghgScope3", "fossilFuelPct", "sbti",
    "biodiversity", "waterEmissions", "hazardousWaste",
    "ungcViolations", "ungcProcesses", "genderPayGap", "humanRightsPolicy",
    "boardDiversity", "controversialWeapons", "antiCorruption",
    "msciRating", "sustainalytics", "internalScore", "esgTrend",
]


async def _get_company(db, ticker):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    return company


async def _get_or_create_esg(db, company):
    result = await db.execute(select(ESGData).where(ESGData.company_id == company.id))
    row = result.scalar_one_or_none()
    if row is None:
        row = ESGData(id=uuid.uuid4(), company_id=company.id, data="{}")
        db.add(row)
        await db.commit()
        await db.refresh(row)
    return row


def _parse_data(row):
    if not row.data:
        return {}
    if isinstance(row.data, dict):
        return row.data
    try:
        return json.loads(row.data)
    except Exception:
        return {}


_FIELD_DEFINITIONS = {
    "env": {"label": "Environmental", "fields": [
        "ghgScope1","ghgScope2","ghgScope3","ghgTotal","carbonFootprint","ghgIntensity",
        "fossilFuelPct","nonRenewablePct","biodiversity","waterEmissions","hazardousWaste","sbti","netZeroTarget",
    ]},
    "soc": {"label": "Social", "fields": [
        "ungcViolations","ungcProcesses","genderPayGap","ltir","employeeTurnover",
        "unionisation","humanRightsPolicy","supplyChainAudit",
    ]},
    "gov": {"label": "Governance", "fields": [
        "boardDiversity","controversialWeapons","ceoPayRatio","independentDirectors",
        "auditQuality","antiCorruption","whistleblower","dataPrivacy",
    ]},
    "ratings": {"label": "Ratings", "fields": [
        "msciRating","sustainalytics","cdpScore","issSoc","issGov",
        "ftse4good","djsi","internalScore","esgTrend","redFlags","esgNotes",
    ]},
}
_ALL_FIELDS = [k for sec in _FIELD_DEFINITIONS.values() for k in sec["fields"]]


@router.get("/companies/{ticker:path}/esg")
async def get_esg(ticker: str, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    row = await _get_or_create_esg(db, company)
    data = _parse_data(row)
    filled = sum(1 for k in _ALL_FIELDS if data.get(k) not in (None, ""))
    completeness = round(filled / len(_ALL_FIELDS) * 100) if _ALL_FIELDS else 0
    return {
        "company_id": str(company.id), "ticker": company.ticker,
        "company_name": company.name, "sector": company.sector,
        "data": data, "ai_summary": row.ai_summary,
        "ai_summary_date": row.ai_summary_date.isoformat() if row.ai_summary_date else None,
        "completeness": completeness, "field_definitions": _FIELD_DEFINITIONS,
    }


class ESGFieldUpdate(BaseModel):
    key: str
    value: str


@router.patch("/companies/{ticker:path}/esg")
async def update_esg_field(ticker: str, body: ESGFieldUpdate, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    row = await _get_or_create_esg(db, company)
    data = _parse_data(row)
    data[body.key] = body.value
    row.data = json.dumps(data)
    await db.commit()
    return {"status": "saved", "key": body.key}


@router.put("/companies/{ticker:path}/esg")
async def upsert_esg(ticker: str, body: ESGUpsertRequest, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    row = await _get_or_create_esg(db, company)
    existing = _parse_data(row)
    incoming = {k: v for k, v in body.data.items() if v is not None}
    existing.update(incoming)
    row.data = json.dumps(existing)
    await db.commit()
    return {"status": "saved", "fields_updated": len(incoming)}


@router.post("/companies/{ticker:path}/esg/ask")
async def ask_esg(ticker: str, body: ESGAskRequest, db: AsyncSession = Depends(get_db)):
    from services.llm_client import call_llm_async
    company = await _get_company(db, ticker)
    row = await _get_or_create_esg(db, company)
    esg_context = _build_esg_context(_parse_data(row), company)
    history_text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in body.history[-6:])
    prompt = f"""You are an ESG analyst assistant for Oldfield Partners, a value-focused fund manager.

{esg_context}

Answer questions about ESG risks, SFDR PAI metrics, regulatory requirements, and sustainability.
Be concise and investment-focused. When data is missing, say so and suggest where to find it.

{history_text}

User: {body.question}"""
    try:
        answer = await call_llm_async(prompt, max_tokens=1024, timeout_seconds=25)
        return {"answer": answer}
    except TimeoutError:
        return {"answer": "The analysis is taking too long. Please try a simpler question."}
    except Exception as e:
        raise HTTPException(502, f"LLM error: {str(e)[:200]}")


@router.post("/companies/{ticker:path}/esg/analyse")
async def analyse_esg(ticker: str, db: AsyncSession = Depends(get_db)):
    from services.llm_client import call_llm_json_async
    company = await _get_company(db, ticker)
    row = await _get_or_create_esg(db, company)
    data = _parse_data(row)
    if not data or all(v in (None, "") for v in data.values()):
        raise HTTPException(400, "No ESG data entered yet.")
    esg_context = _build_esg_context(data, company)
    missing = [k for k in _ALL_IMPORTANT_KEYS if not data.get(k)]
    prompt = f"""You are a senior ESG analyst reviewing:

{esg_context}

Missing important fields: {', '.join(missing) if missing else 'None'}

Return a JSON object:
{{
  "summary": "<2-3 paragraph investment-grade ESG summary>",
  "suggested_fields": {{"<field_key>": "<suggested value with rationale>"}},
  "missing_fields": ["<field_key>"],
  "red_flags": ["<material ESG risk>"]
}}
Respond ONLY with JSON. No preamble."""
    try:
        parsed = await call_llm_json_async(prompt, max_tokens=2048)
    except Exception as e:
        raise HTTPException(502, f"Analysis failed: {str(e)[:200]}")
    row.ai_summary = parsed.get("summary", "")
    row.ai_summary_date = datetime.now(timezone.utc)
    await db.commit()
    return ESGAnalyseResponse(
        summary=parsed.get("summary", ""),
        suggested_fields=parsed.get("suggested_fields", {}),
        missing_fields=parsed.get("missing_fields", missing),
        red_flags=parsed.get("red_flags", []),
    )


@router.get("/esg/portfolio/pai-report")
async def portfolio_pai_report(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ESGData, Company).join(Company, ESGData.company_id == Company.id))
    rows = result.all()
    holdings, ungc, controv, scores = [], 0, 0, []
    pai_keys = ["ghgScope1","ghgScope2","ghgScope3","ghgTotal","carbonFootprint",
        "fossilFuelPct","biodiversity","ungcViolations","boardDiversity",
        "controversialWeapons","genderPayGap","msciRating","esgTrend","internalScore"]
    for esg, co in rows:
        d = _parse_data(esg)
        holdings.append({"company_id": str(co.id), "ticker": co.ticker, "company_name": co.name, **{k: d.get(k) for k in pai_keys}})
        if d.get("ungcViolations") == "Y": ungc += 1
        if d.get("controversialWeapons") == "Y": controv += 1
        try: scores.append(float(d.get("internalScore", "")))
        except (ValueError, TypeError): pass
    total = len(rows)
    return {
        "holdings": holdings, "coverage_pct": round((sum(1 for e, _ in rows if e.data and e.data != "{}") / total * 100), 1) if total else 0,
        "ungc_violations_count": ungc, "controversial_weapons_count": controv,
        "avg_internal_score": round(sum(scores) / len(scores), 1) if scores else None, "total_holdings": total,
    }


# Chat alias — UI calls /esg/chat, router has /esg/ask
@router.post("/companies/{ticker:path}/esg/chat")
async def chat_esg(ticker: str, body: ESGAskRequest, db: AsyncSession = Depends(get_db)):
    return await ask_esg(ticker, body, db)


# ─────────────────────────────────────────────────────────────────
# Bulk PAI import — populate ESG data from spreadsheet data
# ─────────────────────────────────────────────────────────────────
class PAIImportRow(BaseModel):
    company_name: str
    pai1_scope1: Optional[str] = None           # PAI:1 (Scope 1)
    pai2_scope1to3: Optional[str] = None        # PAI:2 (Scope 1 to 3)
    pai3: Optional[str] = None                  # PAI:3 (GHG intensity)
    sbti: Optional[str] = None                  # SBTi Yes/No
    pai4: Optional[str] = None                  # PAI:4 (fossil fuel)
    pai5: Optional[str] = None                  # PAI:5 (non-renewable energy)
    pai6: Optional[str] = None                  # PAI:6 (energy intensity)
    pai7: Optional[str] = None                  # PAI:7
    pai8: Optional[str] = None                  # PAI:8
    pai9: Optional[str] = None                  # PAI:9
    pai10_ungc: Optional[str] = None            # PAI:10 (UNGC violations)
    pai11_ungc_process: Optional[str] = None    # PAI:11 (UNGC processes)
    pai12: Optional[str] = None                 # PAI:12
    pai13_board_diversity: Optional[str] = None # PAI:13 (Board diversity)
    pai14_weapons: Optional[str] = None         # PAI:14 (Controversial weapons)


class PAIImportRequest(BaseModel):
    rows: list[PAIImportRow]


# Mapping from company names to tickers (fuzzy match helper)
_NAME_TICKER_MAP = {
    "chubb": "CB", "nov": "NOV", "lloyds": "LLOY LN", "arrow": "ARW",
    "samsung": "005930 KS", "sanofi": "SAN FP", "fresenius": "FRE GY",
    "exor": "EXO IM", "henkel": "HEN GY", "heineken": "HEIA",
    "handelsbanken": "SHBA SS", "southwest": "LUV", "disney": "DIS",
    "eni": "ENI IM", "ckh": "1 HK", "ckhutch": "1 HK",
    "tesco": "TSCO LN", "easyjet": "EZJ LN", "whitbread": "WTB LN",
    "arcelor": "MT NA", "lear": "LEA", "kyocera": "6971 JP",
    "bunzl": "BNZL", "swatch": "UHR SW", "fairfax": "FFH CN",
    "merck": "MRK GY", "pason": "PSI CN",
}


@router.post("/esg/import-pai")
async def import_pai_data(body: PAIImportRequest, db: AsyncSession = Depends(get_db)):
    """
    Bulk import PAI data from spreadsheet. Matches company names to existing
    companies in the database and populates their ESG fields.
    """
    results = []

    for row in body.rows:
        name_lower = row.company_name.strip().lower()

        # Try to find company by name match or ticker map
        company = None

        # First try exact ticker from map
        mapped_ticker = _NAME_TICKER_MAP.get(name_lower)
        if mapped_ticker:
            q = await db.execute(select(Company).where(Company.ticker == mapped_ticker))
            company = q.scalar_one_or_none()

        # Then try name contains
        if not company:
            q = await db.execute(
                select(Company).where(Company.name.ilike(f"%{name_lower}%"))
            )
            company = q.scalar_one_or_none()

        # Then try ticker contains
        if not company:
            q = await db.execute(
                select(Company).where(Company.ticker.ilike(f"%{name_lower}%"))
            )
            company = q.scalar_one_or_none()

        if not company:
            results.append({"company": row.company_name, "status": "not_found"})
            continue

        # Map PAI columns to ESG field keys
        esg_updates = {}

        def _set(key, val):
            if val and val.strip() and val.strip().upper() != "NA":
                esg_updates[key] = val.strip()

        _set("ghgScope1", row.pai1_scope1)
        _set("ghgTotal", row.pai2_scope1to3)
        _set("ghgIntensity", row.pai3)
        _set("sbti", row.sbti)
        _set("fossilFuelPct", row.pai4)
        _set("nonRenewablePct", row.pai5)
        # pai6 = energy intensity (no direct field, store as ghgIntensity if not set)
        # pai7-9 = biodiversity, water, hazardous waste
        _set("biodiversity", row.pai7)
        _set("waterEmissions", row.pai8)
        _set("hazardousWaste", row.pai9)
        _set("ungcViolations", row.pai10_ungc)
        _set("ungcProcesses", row.pai11_ungc_process)
        _set("genderPayGap", row.pai12)
        _set("boardDiversity", row.pai13_board_diversity)
        _set("controversialWeapons", row.pai14_weapons)

        if not esg_updates:
            results.append({"company": row.company_name, "ticker": company.ticker, "status": "no_data"})
            continue

        # Upsert ESG data
        esg_q = await db.execute(select(ESGData).where(ESGData.company_id == company.id))
        esg_row = esg_q.scalar_one_or_none()

        if esg_row:
            existing = json.loads(esg_row.data) if isinstance(esg_row.data, str) else (esg_row.data or {})
            existing.update(esg_updates)
            esg_row.data = json.dumps(existing)
        else:
            esg_row = ESGData(
                id=uuid.uuid4(), company_id=company.id,
                data=json.dumps(esg_updates),
            )
            db.add(esg_row)

        results.append({
            "company": row.company_name, "ticker": company.ticker,
            "status": "updated", "fields_set": len(esg_updates),
        })

    await db.commit()
    return {
        "imported": len([r for r in results if r["status"] == "updated"]),
        "not_found": len([r for r in results if r["status"] == "not_found"]),
        "results": results,
    }
