"""
IR Website Scraper

Crawls a company's IR documents page looking for PDF links.
Works for any company with a standard static IR page — no RSS needed.

Strategy:
  1. Fetch the IR documents page
  2. Find all links to sub-pages (e.g. /annual-report-2025/) and PDF files
  3. For each sub-page, fetch it and find PDF download links
  4. Return HarvestCandidates for any PDFs not yet seen

This is the fallback source for companies that don't have RSS feeds.
Results are deduplicated by URL in the harvested_documents table as usual.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Sub-page URL patterns that suggest a results/reports landing page
_RESULTS_PAGE_PATTERNS = [
    r"annual.?report",
    r"half.?year",
    r"interim.?result",
    r"full.?year",
    r"results.?report",
    r"preliminary",
    r"trading.?update",
    r"q[1-4].{0,10}20\d\d",
    r"20\d\d.{0,10}(result|report|interim|annual)",
    r"investor.?present",
    r"capital.?market",
]

# PDF link text patterns that suggest a results document
_RESULTS_PDF_PATTERNS = [
    r"annual.?report",
    r"results",
    r"interim",
    r"half.?year",
    r"full.?year",
    r"preliminary",
    r"trading.?update",
    r"press.?release",
    r"earnings",
    r"presentation",
    r"appendix",
    r"slides",
]

_PERIOD_PATTERNS = [
    (r"full[- ]?year\s+(\d{4})",              lambda m: f"{m.group(1)}_FY"),
    (r"(\d{4})\s+full[- ]?year",              lambda m: f"{m.group(1)}_FY"),
    (r"annual\s+report\s+(\d{4})",            lambda m: f"{m.group(1)}_FY"),
    (r"annual\s+report\s*(\d{4})",            lambda m: f"{m.group(1)}_FY"),
    (r"(\d{4})\s+annual",                     lambda m: f"{m.group(1)}_FY"),
    (r"half[- ]?year\s+(\d{4})",              lambda m: f"{m.group(1)}_H1"),
    (r"(\d{4})\s+half[- ]?year",              lambda m: f"{m.group(1)}_H1"),
    (r"interim\s+(?:results?\s+)?(\d{4})",    lambda m: f"{m.group(1)}_H1"),
    (r"H1\s+(\d{4})",                         lambda m: f"{m.group(1)}_H1"),
    (r"H2\s+(\d{4})",                         lambda m: f"{m.group(1)}_H2"),
    (r"HY\s+(\d{4})",                         lambda m: f"{m.group(1)}_HY"),
    (r"(\d{4})[_-]HY",                        lambda m: f"{m.group(1)}_HY"),
    (r"Q([1-4])\s+(\d{4})",                   lambda m: f"{m.group(2)}_Q{m.group(1)}"),
    (r"(\d{4})\s+Q([1-4])",                   lambda m: f"{m.group(1)}_Q{m.group(2)}"),
    (r"(?:first|1st)\s+quarter\s+(\d{4})",    lambda m: f"{m.group(1)}_Q1"),
    (r"(?:second|2nd)\s+quarter\s+(\d{4})",   lambda m: f"{m.group(1)}_Q2"),
    (r"(?:third|3rd)\s+quarter\s+(\d{4})",    lambda m: f"{m.group(1)}_Q3"),
    (r"(?:fourth|4th)\s+quarter\s+(\d{4})",   lambda m: f"{m.group(1)}_Q4"),
    # Infer from URL slug e.g. /annual-report-2025/
    (r"(\d{4})",                              lambda m: f"{m.group(1)}_FY"),
]


def _base(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _infer_period(text: str) -> Optional[str]:
    for pattern, fmt in _PERIOD_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result = fmt(m)
            # Sanity check — year should be plausible
            year_match = re.search(r"(\d{4})", result)
            if year_match:
                year = int(year_match.group(1))
                if 2010 <= year <= 2030:
                    return result
    return None


def _classify_doc_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["transcript", "call", "q&a"]):
        return "transcript"
    if any(k in t for k in ["presentation", "slides", "investor day", "capital markets"]):
        return "presentation"
    return "earnings_release"


def _is_results_link(href: str, link_text: str) -> bool:
    combined = (href + " " + link_text).lower()
    return any(re.search(p, combined, re.IGNORECASE) for p in _RESULTS_PAGE_PATTERNS)


def _is_results_pdf(href: str, link_text: str) -> bool:
    combined = (href + " " + link_text).lower()
    return any(re.search(p, combined, re.IGNORECASE) for p in _RESULTS_PDF_PATTERNS)


def _extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract all (href, link_text) pairs from HTML."""
    links = []
    for m in re.finditer(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        html, re.IGNORECASE | re.DOTALL
    ):
        href = m.group(1).strip()
        text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        if not href or href.startswith('#') or href.startswith('mailto:'):
            continue
        if not href.startswith('http'):
            href = urljoin(base_url, href)
        links.append((href, text))
    return links


async def scrape_ir_page(
    ticker: str,
    ir_docs_url: str,
    max_subpages: int = 10,
) -> list[dict]:
    """
    Scrape an IR documents page and all linked sub-pages for PDF files.

    Args:
        ticker: Company ticker e.g. "BNZL LN"
        ir_docs_url: URL of the IR results/reports page
        max_subpages: Max number of sub-pages to crawl (prevents runaway)

    Returns:
        List of HarvestCandidates
    """
    candidates = []
    visited = set()
    base = _base(ir_docs_url)
    ir_domain = urlparse(ir_docs_url).netloc

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers=_HEADERS,
    ) as client:

        # ── Fetch main IR docs page ───────────────────────────────
        try:
            resp = await client.get(ir_docs_url)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("[SCRAPE] Failed to fetch %s: %s", ir_docs_url, e)
            return []

        visited.add(ir_docs_url)
        main_html = resp.text
        main_links = _extract_links(main_html, ir_docs_url)

        # ── Find direct PDF links on main page ────────────────────
        for href, text in main_links:
            if href.lower().endswith('.pdf') and _is_results_pdf(href, text):
                candidates.append(_make_candidate(ticker, href, text, ir_docs_url))

        # ── Find sub-pages to crawl ───────────────────────────────
        subpages = []
        for href, text in main_links:
            if href in visited:
                continue
            # Must be same domain
            if urlparse(href).netloc != ir_domain:
                continue
            # Must look like a results sub-page
            if _is_results_link(href, text):
                subpages.append((href, text))

        logger.info("[SCRAPE] %s — main page done, %d sub-pages to crawl", ticker, len(subpages))

        # ── Crawl sub-pages ───────────────────────────────────────
        for href, text in subpages[:max_subpages]:
            if href in visited:
                continue
            visited.add(href)

            try:
                sub_resp = await client.get(href)
                sub_resp.raise_for_status()
            except Exception as e:
                logger.debug("[SCRAPE] Sub-page failed %s: %s", href, e)
                continue

            sub_links = _extract_links(sub_resp.text, href)

            for pdf_href, pdf_text in sub_links:
                if not pdf_href.lower().endswith('.pdf'):
                    continue
                if pdf_href in {c['source_url'] for c in candidates}:
                    continue
                # Use page URL + text as context for period inference
                context = f"{text} {pdf_text} {href}"
                candidates.append(_make_candidate(ticker, pdf_href, pdf_text or text, context))

    logger.info("[SCRAPE] %s — found %d PDF candidates", ticker, len(candidates))
    return candidates


def _make_candidate(ticker: str, pdf_url: str, link_text: str, context: str) -> dict:
    """Build a HarvestCandidate dict from a found PDF link."""
    # Use context (page title + link text + URL) for period inference
    period_label = _infer_period(context)
    doc_type = _classify_doc_type(context)

    # Build a readable headline from link text + URL slug
    slug = urlparse(pdf_url).path.split('/')[-1].replace('-', ' ').replace('_', ' ')
    slug = re.sub(r'\.pdf$', '', slug, flags=re.IGNORECASE)
    headline = link_text.strip() if link_text.strip() else slug.title()

    return {
        "ticker":        ticker,
        "source":        "ir_scrape",
        "source_url":    pdf_url,          # dedup key
        "headline":      headline[:300],
        "description":   "",
        "published_at":  datetime.now(timezone.utc),
        "pdf_url":       pdf_url,
        "period_label":  period_label,
        "document_type": doc_type,
    }
