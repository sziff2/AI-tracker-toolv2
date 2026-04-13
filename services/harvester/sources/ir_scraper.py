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

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from services.harvester.http_retry import fetch_with_retry
from services.harvester.sources.robots_check import can_fetch, get_crawl_delay

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
    # ── German ─────────────────────────────────────────────────
    # German half-year report: "Halbjahresfinanzbericht 2024",
    # "Halbjahresbericht 2024", "Halbjahr 2024". Loose wildcard
    # between keyword and year to handle intervening words. Placed
    # first so it wins before the looser English/fallback patterns.
    (r"halbjahres?(?:finanz)?bericht.{0,40}(\d{4})", lambda m: f"{m.group(1)}_Q2"),
    (r"halbjahr.{0,40}(\d{4})",               lambda m: f"{m.group(1)}_Q2"),
    # German annual report: "Geschäftsbericht 2024", "Geschaeftsbericht 2024"
    (r"gesch(?:ä|ae)ftsbericht.{0,40}(\d{4})", lambda m: f"{m.group(1)}_Q4"),
    # German quarterly statement: "Quartalsmitteilung Q3 2024" —
    # already caught by the Q([1-4]) pattern below, but leave a
    # German keyword pass that falls through to those.

    # ── Full year — both word orders ──────────────────────────
    (r"full[- ]?year\s+(\d{4})",              lambda m: f"{m.group(1)}_Q4"),
    (r"(\d{4})[- ]full[- ]?year",             lambda m: f"{m.group(1)}_Q4"),
    (r"annual\s+report\s+(\d{4})",            lambda m: f"{m.group(1)}_Q4"),
    (r"(\d{4})\s+annual",                     lambda m: f"{m.group(1)}_Q4"),
    (r"FY\s*(\d{4})",                         lambda m: f"{m.group(1)}_Q4"),
    (r"(\d{4})[- ]FY",                        lambda m: f"{m.group(1)}_Q4"),
    # ── Half year — HY/H1/interim all map to Q2 ───────────────
    # Loose wildcard (up to 40 chars) between keyword and year to
    # handle titles like "Half-year financial report 2023" where
    # "financial report" sits between "half-year" and the year.
    (r"half[- ]?year.{0,40}(\d{4})",          lambda m: f"{m.group(1)}_Q2"),
    (r"(\d{4})\s+half[- ]?year",              lambda m: f"{m.group(1)}_Q2"),
    (r"interim\s+(?:results?\s+)?(\d{4})",    lambda m: f"{m.group(1)}_Q2"),
    (r"H1\s+(\d{4})",                         lambda m: f"{m.group(1)}_Q2"),
    (r"H2\s+(\d{4})",                         lambda m: f"{m.group(1)}_Q4"),
    (r"HY\s+(\d{4})",                         lambda m: f"{m.group(1)}_Q2"),
    (r"(\d{4})[_-]HY",                        lambda m: f"{m.group(1)}_Q2"),
    # Quarter — Q1/Q2/Q3/Q4 format, both word orders
    (r"Q([1-4])\s+(\d{4})",                   lambda m: f"{m.group(2)}_Q{m.group(1)}"),
    (r"(\d{4})\s+Q([1-4])",                   lambda m: f"{m.group(1)}_Q{m.group(2)}"),
    (r"(\d{4})[- ]Q([1-4])",                  lambda m: f"{m.group(1)}_Q{m.group(2)}"),
    # Quarter — ordinal words, both word orders
    (r"(?:first|1st)\s+quarter\s+(\d{4})",    lambda m: f"{m.group(1)}_Q1"),
    (r"(?:second|2nd)\s+quarter\s+(\d{4})",   lambda m: f"{m.group(1)}_Q2"),
    (r"(?:third|3rd)\s+quarter\s+(\d{4})",    lambda m: f"{m.group(1)}_Q3"),
    (r"(?:fourth|4th)\s+quarter\s+(\d{4})",   lambda m: f"{m.group(1)}_Q4"),
    (r"(\d{4})[- ](?:first|1st)[- ]quarter",  lambda m: f"{m.group(1)}_Q1"),
    (r"(\d{4})[- ](?:second|2nd)[- ]quarter", lambda m: f"{m.group(1)}_Q2"),
    (r"(\d{4})[- ](?:third|3rd)[- ]quarter",  lambda m: f"{m.group(1)}_Q3"),
    (r"(\d{4})[- ](?:fourth|4th)[- ]quarter", lambda m: f"{m.group(1)}_Q4"),
    # Quarter — "fourth quarter 2024 and full year" or "2025-fourth-quarter-results"
    (r"(?:fourth|4th)[- ]quarter[- ](\d{4})", lambda m: f"{m.group(1)}_Q4"),
    (r"(?:third|3rd)[- ]quarter[- ](\d{4})",  lambda m: f"{m.group(1)}_Q3"),
    (r"(?:second|2nd)[- ]quarter[- ](\d{4})", lambda m: f"{m.group(1)}_Q2"),
    (r"(?:first|1st)[- ]quarter[- ](\d{4})",  lambda m: f"{m.group(1)}_Q1"),
    # Short quarter formats: 2q22, 4q-23, 1q25, 3q-24 (common in filenames)
    (r"(?:^|[^0-9])([1-4])q[- ]?(\d{2})(?:[^0-9]|$)", lambda m: f"20{m.group(2)}_Q{m.group(1)}"),
    # Short FY format: fy25, FY24
    (r"(?:^|[^a-zA-Z])fy[- ]?(\d{2})(?:[^0-9]|$)", lambda m: f"20{m.group(1)}_Q4"),
    # Short half-year: h1-25 → Q2, h2-24 → Q4
    (r"(?:^|[^a-zA-Z])h1[- ]?(\d{2})(?:[^0-9]|$)", lambda m: f"20{m.group(1)}_Q2"),
    (r"(?:^|[^a-zA-Z])h2[- ]?(\d{2})(?:[^0-9]|$)", lambda m: f"20{m.group(1)}_Q4"),
    # Bare year as last resort
    (r"(\d{4})",                              lambda m: f"{m.group(1)}_Q4"),
]


_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}


def _extract_date(text: str) -> Optional[datetime]:
    """Try to extract a publication date from a URL/filename string.

    Matches: YYYY-MM-DD, YYYYMMDD, DD-Mon-YYYY, Mon-YYYY, YYYY/MM/DD
    Returns a timezone-aware datetime or None.
    """
    # YYYY-MM-DD or YYYY/MM/DD
    m = re.search(r'(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])', text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            pass

    # YYYYMMDD (8 consecutive digits starting with 20)
    m = re.search(r'(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])', text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            pass

    # DD-Mon-YYYY or DD Mon YYYY (e.g. 15-Jan-2025, 15 January 2025)
    m = re.search(r'(\d{1,2})[\s-]([A-Za-z]{3,9})[\s-](20\d{2})', text)
    if m:
        mon = _MONTH_MAP.get(m.group(2).lower())
        if mon:
            try:
                return datetime(int(m.group(3)), mon, int(m.group(1)), tzinfo=timezone.utc)
            except ValueError:
                pass

    # Mon-YYYY or Mon YYYY (e.g. Jan-2025, January 2025) — default to 1st
    m = re.search(r'([A-Za-z]{3,9})[\s-](20\d{2})', text)
    if m:
        mon = _MONTH_MAP.get(m.group(1).lower())
        if mon:
            return datetime(int(m.group(2)), mon, 1, tzinfo=timezone.utc)

    return None


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
    if any(k in t for k in ["transcript", "call transcript", "q&a", "questions and answers"]):
        return "transcript"
    if any(k in t for k in ["presentation", "slides", "investor day", "capital markets", "analyst slides"]):
        return "presentation"
    if any(k in t for k in ["annual report", "annual-report", "20-f", "40-f"]):
        return "annual_report"
    if any(k in t for k in ["proxy", "def 14", "agm", "general meeting"]):
        return "proxy_statement"
    if any(k in t for k in ["sustainability", "esg", "csr", "tcfd", "climate"]):
        return "sustainability_report"
    if any(k in t for k in ["broker", "research note", "analyst note"]):
        return "broker_note"
    if any(k in t for k in ["10-k", "10k"]):
        return "10-K"
    if any(k in t for k in ["10-q", "10q"]):
        return "10-Q"
    if any(k in t for k in ["press release", "earnings release", "results", "quarterly report"]):
        return "earnings_release"
    return "other"


def _is_results_link(href: str, link_text: str) -> bool:
    combined = (href + " " + link_text).lower()
    return any(re.search(p, combined, re.IGNORECASE) for p in _RESULTS_PAGE_PATTERNS)


def _is_results_pdf(href: str, link_text: str) -> bool:
    combined = (href + " " + link_text).lower()
    return any(re.search(p, combined, re.IGNORECASE) for p in _RESULTS_PDF_PATTERNS)


def _extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract all (href, link_text) pairs from HTML.
    Also decodes unicode-escaped HTML in JSON blobs (common in SPA sites)."""
    # First decode any unicode escapes in the raw HTML
    # This handles pages like ArcelorMittal where links are in JSON as \u003ca href=\u0022...\u0022\u003e
    try:
        decoded = html.encode('utf-8').decode('unicode_escape')
    except (UnicodeDecodeError, UnicodeEncodeError):
        decoded = ""

    # Combine original and decoded HTML for link extraction
    combined = html + "\n" + decoded

    links = []
    seen = set()
    for m in re.finditer(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        combined, re.IGNORECASE | re.DOTALL
    ):
        href = m.group(1).strip()
        text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        if not href or href.startswith('#') or href.startswith('mailto:'):
            continue
        if not href.startswith('http'):
            href = urljoin(base_url, href)
        if href not in seen:
            seen.add(href)
            links.append((href, text))
    return links


async def scrape_ir_page(
    ticker: str,
    ir_docs_url: str,
    max_subpages: int = 10,
) -> list[dict]:
    """
    Scrape IR documents pages and all linked sub-pages for PDF files.
    Supports multiple URLs separated by commas or newlines.

    Args:
        ticker: Company ticker e.g. "BNZL LN"
        ir_docs_url: URL(s) of the IR results/reports page(s), comma or newline separated
        max_subpages: Max number of sub-pages to crawl (prevents runaway)

    Returns:
        List of HarvestCandidates
    """
    candidates = []
    visited = set()

    # Support multiple URLs separated by comma, newline, or space
    import re as _re
    all_urls = [u.strip() for u in _re.split(r'[,\n\s]+', ir_docs_url) if u.strip().startswith('http')]
    if not all_urls:
        all_urls = [ir_docs_url]

    base = _base(all_urls[0])
    ir_domain = urlparse(all_urls[0]).netloc

    # Pages to scrape: configured URLs + fast sibling discovery
    pages_to_scrape = list(all_urls)

    # Probe sibling pages with a short timeout (no cloudscraper fallback)
    _SIBLING_PATTERNS = [
        "previous-results", "results-archive", "results",
        "reports-and-presentations", "quarterly-results",
    ]
    parsed_url = urlparse(all_urls[0])
    parent = '/'.join(parsed_url.path.rstrip('/').split('/')[:-1])
    async with httpx.AsyncClient(timeout=3.0, follow_redirects=True, headers=_HEADERS) as probe:
        for sibling in _SIBLING_PATTERNS:
            sib_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parent}/{sibling}"
            if sib_url in pages_to_scrape or sib_url == ir_docs_url:
                continue
            try:
                resp = await probe.head(sib_url)
                if resp.status_code == 200:
                    pages_to_scrape.append(sib_url)
                    logger.debug("[SCRAPE] %s — discovered sibling %s", ticker, sibling)
            except Exception:
                pass  # fast fail — don't retry on sibling discovery

    from services.doc_utils import async_fetch_page

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers=_HEADERS,
    ) as client:

        crawl_delay = get_crawl_delay(all_urls[0])

        all_main_links = []
        for page_url in pages_to_scrape:
            if page_url in visited:
                continue
            if not can_fetch(page_url):
                logger.warning("[SCRAPE] %s — robots.txt disallows %s, skipping", ticker, page_url)
                continue
            try:
                # Respect crawl-delay between requests
                if visited:
                    await asyncio.sleep(crawl_delay)
                # Use Cloudflare-aware fetcher for main pages
                html = await async_fetch_page(page_url, timeout=20)
                if not html:
                    if page_url == ir_docs_url:
                        logger.warning("[SCRAPE] Failed to fetch %s", ir_docs_url)
                        return []
                    continue
                visited.add(page_url)
                page_links = _extract_links(html, page_url)
                all_main_links.extend(page_links)
                logger.info("[SCRAPE] %s — fetched %s (%d links)", ticker, page_url, len(page_links))
            except Exception as e:
                if page_url == ir_docs_url:
                    logger.warning("[SCRAPE] Failed to fetch %s: %s", ir_docs_url, e)
                    return []
                continue

        main_links = all_main_links

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
            if not can_fetch(href):
                logger.warning("[SCRAPE] %s — robots.txt disallows sub-page %s, skipping", ticker, href)
                continue
            visited.add(href)

            try:
                await asyncio.sleep(crawl_delay)
                sub_resp = await fetch_with_retry(href, client, retries=2, timeout=15)
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
    # Build a readable headline from link text + URL slug
    slug = urlparse(pdf_url).path.split('/')[-1].replace('-', ' ').replace('_', ' ')
    slug = re.sub(r'\.pdf$', '', slug, flags=re.IGNORECASE)
    headline = link_text.strip() if link_text.strip() else slug.title()

    # Use full context (page title + link text + URL path + slug) for inference
    full_context = f"{context} {pdf_url} {slug} {headline}"
    period_label = _infer_period(full_context)
    doc_type = _classify_doc_type(full_context)

    # Try to extract a real publication date from URL/filename
    published_at = _extract_date(full_context) or datetime.now(timezone.utc)

    return {
        "ticker":        ticker,
        "source":        "ir_scrape",
        "source_url":    pdf_url,          # dedup key
        "headline":      headline[:300],
        "description":   "",
        "published_at":  published_at,
        "pdf_url":       pdf_url,
        "period_label":  period_label,
        "document_type": doc_type,
    }
