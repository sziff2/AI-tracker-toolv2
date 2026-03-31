"""
SEC EDGAR Harvester

Polls the SEC EDGAR submissions API for new filings.
No API key required — SEC provides a free REST API.

Covers: 10-K (annual), 10-Q (quarterly), 8-K (material events),
        20-F (foreign private issuer annual), 40-F (Canadian filers like Pason).

EDGAR rate limit: max 10 requests/second. We stay well under that.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Per-company EDGAR config
# CIK: SEC Central Index Key — look up at https://www.sec.gov/cgi-bin/browse-edgar
# ─────────────────────────────────────────────────────────────────
EDGAR_SOURCES: dict[str, dict] = {
    "PSI CN": {
        "name": "Pason Systems",
        "cik": "0001104485",
        "form_types": ["40-F", "6-K"],   # Canadian filer
    },
    "LKQ US": {
        "name": "LKQ Corporation",
        "cik": "0001065696",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "FFH CN": {
        "name": "Fairfax Financial",
        "cik": "0000915191",
        "form_types": ["40-F", "6-K"],   # Canadian filer
    },
    "ALLY US": {
        "name": "Ally Financial",
        "cik": "0000040729",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "LEA US": {
        "name": "Lear Corporation",
        "cik": "0000842162",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "ARW US": {
        "name": "Arrow Electronics",
        "cik": "0000007536",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "CB US": {
        "name": "Chubb Limited",
        "cik": "0000896159",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "PM US": {
        "name": "Philip Morris International",
        "cik": "0001413329",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "ALGT US": {
        "name": "Allegiant Travel",
        "cik": "0001362468",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "MT NA": {
        "name": "ArcelorMittal",
        "cik": "0001243429",
        "form_types": ["20-F", "6-K"],
    },
    "BP LN": {
        "name": "BP PLC",
        "cik": "0000313807",
        "form_types": ["20-F", "6-K"],
    },
    "SHEL LN": {
        "name": "Shell plc",
        "cik": "0001306965",
        "form_types": ["20-F", "6-K"],
    },
    "PYPL US": {
        "name": "PayPal",
        "cik": "0001633917",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "NOV US": {
        "name": "NOV Inc",
        "cik": "0001021860",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
    "DIS US": {
        "name": "Walt Disney",
        "cik": "0001744489",
        "form_types": ["10-K", "10-Q", "8-K"],
    },
}

# Form types that map to financial results
_EARNINGS_FORMS = {"10-K", "10-Q", "20-F", "40-F", "6-K"}
_MATERIAL_FORMS = {"8-K"}

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SUBMISSIONS = f"{EDGAR_BASE}/submissions/CIK{{cik}}.json"
EDGAR_FILING_INDEX = f"{EDGAR_BASE}/Archives/edgar/data/{{cik}}/{{accession_no_dashes}}/{{accession_no}}-index.htm"


def _norm_cik(cik: str) -> str:
    """Pad CIK to 10 digits as EDGAR expects."""
    return cik.lstrip("0").zfill(10)


def _accession_to_dashes(accession: str) -> str:
    """Convert 0000000000-00-000000 → 000000000000000000 (no dashes)."""
    return accession.replace("-", "")


def _period_from_form(form_type: str, filing_date: str, period_of_report: Optional[str]) -> Optional[str]:
    """
    Infer a human-readable period label from SEC filing metadata.
    period_of_report is in YYYY-MM-DD format from EDGAR.
    """
    if not period_of_report:
        return None

    try:
        dt = datetime.strptime(period_of_report, "%Y-%m-%d")
    except ValueError:
        return None

    year = dt.year
    month = dt.month

    if form_type in ("10-K", "20-F", "40-F"):
        return f"{year}_FY"

    if form_type in ("10-Q", "6-K"):
        # Map fiscal quarter end month to quarter label
        # Standard US fiscal year — adjust if needed per company
        if month in (1, 2, 3):
            return f"{year}_Q1"
        elif month in (4, 5, 6):
            return f"{year}_Q2"
        elif month in (7, 8, 9):
            return f"{year}_Q3"
        elif month in (10, 11, 12):
            return f"{year}_Q4"

    if form_type == "8-K":
        # Material event — label with filing date
        return f"{year}_Q{((month - 1) // 3) + 1}"

    return None


def _classify_document_type(form_type: str, items: str = "") -> str:
    """Classify SEC filing into document type.
    For 8-Ks, uses SEC item numbers to determine the event type:
      2.02 = Results of Operations (earnings release)
      5.02 = Director/officer changes
      1.01 = Material agreement
      Others = general material event
    """
    if form_type in ("10-K", "20-F", "40-F"):
        return "annual_report"
    if form_type == "10-Q":
        return "10-Q"
    if form_type == "6-K":
        # 6-K is the foreign private issuer catch-all — could be earnings, quarterly, or material event
        # Check description for clues
        if items and ("2.02" in items or "earnings" in items.lower() or "results" in items.lower()):
            return "earnings_release"
        return "other"
    if form_type == "ARS":
        return "annual_report"
    if form_type == "DEF 14A":
        return "proxy_statement"
    if form_type == "8-K":
        if "2.02" in items:
            return "earnings_release"
        return "other"
    return "other"


async def fetch_sec_edgar(ticker: str, max_filings: int = 20) -> list[dict]:
    """
    Fetch recent SEC filings for a given ticker.

    Returns a list of HarvestCandidate dicts compatible with dispatcher.py.
    Only returns filings from the last 2 years to avoid historical noise.
    """
    config = EDGAR_SOURCES.get(ticker)
    if not config:
        logger.debug("No EDGAR config for ticker %s", ticker)
        return []

    cik_raw = config["cik"]
    cik_padded = _norm_cik(cik_raw)
    cik_int = int(cik_raw)
    target_forms = set(config["form_types"])
    candidates = []

    headers = {
        "User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        url = EDGAR_SUBMISSIONS.format(cik=cik_padded)
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("[%s] EDGAR fetch failed: %s", ticker, exc)
            return []

        filings = data.get("filings", {}).get("recent", {})
        if not filings:
            logger.warning("[%s] No recent filings in EDGAR response", ticker)
            return []

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        periods = filings.get("periodOfReport", [])
        primary_docs = filings.get("primaryDocument", [])
        descriptions = filings.get("primaryDocDescription", [])
        items_list = filings.get("items", [])

        cutoff_year = datetime.now(timezone.utc).year - 2

        count = 0
        for i, form in enumerate(forms):
            if form not in target_forms:
                continue
            if count >= max_filings:
                break

            filing_date_str = dates[i] if i < len(dates) else ""
            # Skip very old filings
            try:
                filing_year = int(filing_date_str[:4])
                if filing_year < cutoff_year:
                    continue
            except (ValueError, IndexError):
                pass

            accession = accessions[i] if i < len(accessions) else ""
            period_of_report = periods[i] if i < len(periods) else None
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            description = descriptions[i] if i < len(descriptions) else form

            # Build the filing index URL (human-readable unique key)
            accession_dashes = accession  # e.g. "0001104485-24-000010"
            accession_nodash = _accession_to_dashes(accession)
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{accession_nodash}/{accession_dashes}-index.htm"
            )

            # Build direct document URL if we have a primary doc filename
            doc_url = None
            if primary_doc:
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                    f"{accession_nodash}/{primary_doc}"
                )

            published_at = None
            if filing_date_str:
                try:
                    published_at = datetime.strptime(filing_date_str, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    published_at = datetime.now(timezone.utc)

            period_label = _period_from_form(form, filing_date_str, period_of_report)
            headline = f"{config['name']} {form} — {period_of_report or filing_date_str}"

            candidates.append({
                "ticker": ticker,
                "source": "sec_edgar",
                "source_url": filing_url,           # dedup key
                "headline": headline,
                "description": f"{description} filed {filing_date_str}",
                "published_at": published_at or datetime.now(timezone.utc),
                "pdf_url": doc_url,                 # may be HTML — dispatcher handles both
                "period_label": period_label,
                "document_type": _classify_document_type(form, items_list[i] if i < len(items_list) else ""),
                "form_type": form,
            })
            count += 1

    logger.info("[%s] EDGAR found %d relevant filings", ticker, len(candidates))
    return candidates
