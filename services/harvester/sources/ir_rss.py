"""
IR RSS/Atom Feed Harvester (Generalised)

Polls IR RSS/Atom feeds for ALL companies in the DB.
Source URLs are discovered automatically via the discovery agent
and cached in the harvester_sources table.

No hardcoded company configs — works for any holding added to the DB.

Flow per company:
  1. Check harvester_sources table for cached RSS URL
  2. If not cached (or stale), run discovery agent → cache result
  3. Fetch and parse the RSS/Atom feed
  4. Return relevant HarvestCandidates

Keywords are intentionally broad — better to ingest a non-results
press release and skip it downstream than miss a results doc.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional
import xml.etree.ElementTree as ET

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import AsyncSessionLocal
from apps.api.models import Company, HarvesterSource
from services.harvester.discovery import discover_company_sources

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Relevance keywords — broad to avoid missing results docs
# ─────────────────────────────────────────────────────────────────
_RESULTS_KEYWORDS = [
    "results", "earnings", "revenue", "revenues", "profit", "profits",
    "quarterly", "annual", "full year", "half year", "interim",
    "trading update", "trading statement", "financial results",
    "Q1", "Q2", "Q3", "Q4", "H1", "H2",
    "10-K", "10-Q", "20-F", "40-F", "6-K",
    "production", "sales volume",
]

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc":   "http://purl.org/dc/elements/1.1/",
}

_CACHE_MAX_AGE_DAYS = 30

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
    "Accept": "application/rss+xml,application/atom+xml,application/xml,text/xml,*/*",
}


# ─────────────────────────────────────────────────────────────────
# Source cache
# ─────────────────────────────────────────────────────────────────

async def _get_or_discover_source(db: AsyncSession, company: Company) -> Optional[str]:
    """Return cached RSS URL, or discover and cache it if missing/stale."""
    result = await db.execute(
        select(HarvesterSource).where(HarvesterSource.company_id == company.id)
    )
    source = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)

    if source and source.rss_url:
        age_days = (now - source.last_checked_at).days if source.last_checked_at else 999
        if source.override or age_days < _CACHE_MAX_AGE_DAYS:
            return source.rss_url

    # Run discovery (LLM + HTTP probes)
    logger.info("[HARVEST] Discovering IR sources for %s (%s)…", company.name, company.ticker)
    discovered = await discover_company_sources(
        company_name=company.name,
        ticker=company.ticker,
        country=company.country,
    )

    import uuid as _uuid
    if source is None:
        source = HarvesterSource(id=_uuid.uuid4(), company_id=company.id)
        db.add(source)

    source.ir_url           = discovered.get("ir_url")
    source.rss_url          = discovered.get("rss_url")
    source.ir_reachable     = discovered.get("ir_reachable", False)
    source.discovery_method = discovered.get("discovery_method", "failed")
    source.last_checked_at  = now
    await db.commit()

    rss = discovered.get("rss_url")
    if rss:
        logger.info("[HARVEST] Discovered RSS for %s: %s", company.ticker, rss)
    else:
        logger.info("[HARVEST] No RSS found for %s (IR: %s)", company.ticker, discovered.get("ir_url"))
    return rss


# ─────────────────────────────────────────────────────────────────
# Period label inference
# ─────────────────────────────────────────────────────────────────

_PERIOD_PATTERNS = [
    (r"full[- ]?year\s+(\d{4})",              lambda m: f"{m.group(1)}_FY"),
    (r"(\d{4})\s+full[- ]?year",              lambda m: f"{m.group(1)}_FY"),
    (r"annual\s+(?:results\s+)?(\d{4})",      lambda m: f"{m.group(1)}_FY"),
    (r"(\d{4})\s+annual",                     lambda m: f"{m.group(1)}_FY"),
    (r"half[- ]?year\s+(\d{4})",              lambda m: f"{m.group(1)}_H1"),
    (r"(\d{4})\s+half[- ]?year",              lambda m: f"{m.group(1)}_H1"),
    (r"interim\s+(?:results\s+)?(\d{4})",     lambda m: f"{m.group(1)}_H1"),
    (r"H1\s+(\d{4})",                         lambda m: f"{m.group(1)}_H1"),
    (r"H2\s+(\d{4})",                         lambda m: f"{m.group(1)}_H2"),
    (r"Q([1-4])\s+(\d{4})",                   lambda m: f"{m.group(2)}_Q{m.group(1)}"),
    (r"(\d{4})\s+Q([1-4])",                   lambda m: f"{m.group(1)}_Q{m.group(2)}"),
    (r"(?:first|1st)\s+quarter\s+(\d{4})",    lambda m: f"{m.group(1)}_Q1"),
    (r"(?:second|2nd)\s+quarter\s+(\d{4})",   lambda m: f"{m.group(1)}_Q2"),
    (r"(?:third|3rd)\s+quarter\s+(\d{4})",    lambda m: f"{m.group(1)}_Q3"),
    (r"(?:fourth|4th)\s+quarter\s+(\d{4})",   lambda m: f"{m.group(1)}_Q4"),
    (r"(\d{4})\s+(?:first|1st)\s+quarter",    lambda m: f"{m.group(1)}_Q1"),
    (r"(\d{4})\s+(?:second|2nd)\s+quarter",   lambda m: f"{m.group(1)}_Q2"),
    (r"(\d{4})\s+(?:third|3rd)\s+quarter",    lambda m: f"{m.group(1)}_Q3"),
    (r"(\d{4})\s+(?:fourth|4th)\s+quarter",   lambda m: f"{m.group(1)}_Q4"),
]


def _infer_period_label(title: str) -> Optional[str]:
    for pattern, formatter in _PERIOD_PATTERNS:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            return formatter(m)
    return None


def _is_relevant(title: str, description: str) -> bool:
    text = (title + " " + (description or "")).lower()
    return any(kw.lower() in text for kw in _RESULTS_KEYWORDS)


def _classify_doc_type(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ["transcript", "call recording", "q&a"]):
        return "transcript"
    if any(k in t for k in ["presentation", "slides", "investor day", "capital markets day"]):
        return "presentation"
    return "earnings_release"


def _find_pdf_link(item_elem) -> Optional[str]:
    enclosure = item_elem.find("enclosure")
    if enclosure is not None:
        url = enclosure.get("url", "")
        if url.lower().endswith(".pdf"):
            return url
    for link in item_elem.findall("atom:link", _NS):
        href = link.get("href", "")
        if "pdf" in link.get("type", "").lower() or href.lower().endswith(".pdf"):
            return href
    for tag in ["description", "summary"]:
        el = item_elem.find(tag)
        if el is not None and el.text:
            m = re.search(r'https?://[^\s"\'<>]+\.pdf', el.text, re.IGNORECASE)
            if m:
                return m.group(0)
    return None


# ─────────────────────────────────────────────────────────────────
# Feed parser
# ─────────────────────────────────────────────────────────────────

async def _parse_feed(ticker: str, feed_url: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        try:
            resp = await client.get(feed_url, headers=_HEADERS)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("[%s] Feed fetch failed (%s): %s", ticker, feed_url, exc)
            return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        logger.warning("[%s] XML parse error: %s", ticker, exc)
        return []

    is_atom = root.tag.endswith("feed") or "atom" in root.tag.lower()
    if is_atom:
        items = root.findall("atom:entry", _NS) or root.findall("entry")
    else:
        channel = root.find("channel")
        items   = channel.findall("item") if channel is not None else root.findall("item")

    candidates = []
    for item in items:
        def _text(tag: str) -> str:
            el = item.find(tag) or item.find(f"atom:{tag}", _NS)
            return (el.text or "").strip() if el is not None else ""

        title       = _text("title")
        description = _text("description") or _text("summary")
        pub_raw     = _text("pubDate") or _text("published") or _text("updated")

        link = _text("link")
        if not link:
            link_el = item.find("atom:link", _NS) or item.find("link")
            if link_el is not None:
                link = link_el.get("href", "") or (link_el.text or "")

        if not title or not link:
            continue
        if not _is_relevant(title, description):
            continue

        candidates.append({
            "ticker":        ticker,
            "source":        "ir_rss",
            "source_url":    link,
            "headline":      title,
            "description":   description[:500],
            "published_at":  _parse_date(pub_raw) or datetime.now(timezone.utc),
            "pdf_url":       _find_pdf_link(item),
            "period_label":  _infer_period_label(title),
            "document_type": _classify_doc_type(title),
        })

    return candidates


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────

async def fetch_ir_rss_for_company(company: Company) -> list[dict]:
    """
    Discover (or retrieve cached) RSS feed for a company, fetch and parse it.
    Returns a list of HarvestCandidates.
    """
    async with AsyncSessionLocal() as db:
        rss_url = await _get_or_discover_source(db, company)

    if not rss_url:
        return []

    candidates = await _parse_feed(company.ticker, rss_url)
    logger.info("[%s] IR RSS returned %d candidates", company.ticker, len(candidates))
    return candidates
