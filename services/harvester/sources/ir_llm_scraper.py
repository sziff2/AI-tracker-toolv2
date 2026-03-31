"""
LLM-Powered IR Document Scraper

For non-SEC companies, fetches the IR page HTML and uses an LLM to
identify all document links (PDFs, reports, presentations).

Works with:
  - JavaScript-rendered SPA pages (data embedded in JSON blobs)
  - Non-standard link formats
  - CDN/blob storage URLs
  - Multiple languages

The LLM returns structured document metadata that the user can
review before ingesting.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
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

# Headers that may trigger server-side rendering on SPA sites
_SSR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Max chars of page source to send to LLM (reduced to stay within Railway 30s gateway)
_MAX_PAGE_CHARS = 15000

# If main page already has this much content, skip sibling pages
_SKIP_SIBLINGS_THRESHOLD = 5000

# Minimum chars of visible text to consider a page as having real content
_THIN_CONTENT_THRESHOLD = 1000


def _clean_page_source(html: str) -> str:
    """Strip CSS, scripts, and navigation to reduce token count."""
    # Remove style blocks
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove script blocks (but keep inline JSON data)
    html = re.sub(r'<script[^>]*src=[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    # Remove nav, header, footer elements
    html = re.sub(r'<(nav|header|footer)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Collapse whitespace
    html = re.sub(r'\s+', ' ', html)
    return html.strip()


def _extract_visible_text(html: str) -> str:
    """Extract just the visible text from HTML (strip all tags)."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _find_api_urls(html: str, base_url: str) -> list[str]:
    """
    Look for API/JSON endpoints embedded in the HTML source that SPA pages
    call to load document data. Returns candidate URLs to try fetching.
    """
    api_urls = []
    base = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"

    # Match patterns like /api/..., /rest/..., or URLs ending in .json
    patterns = [
        r'["\'](/api/[^"\'>\s]{5,})["\']',
        r'["\'](/rest/[^"\'>\s]{5,})["\']',
        r'["\'](https?://[^"\'>\s]*?/api/[^"\'>\s]{5,})["\']',
        r'["\'](/[^"\'>\s]*?\.json)["\']',
        r'["\'](https?://[^"\'>\s]*?\.json)["\']',
    ]

    for pat in patterns:
        for match in re.finditer(pat, html):
            url = match.group(1)
            if not url.startswith("http"):
                url = urljoin(base, url)
            # Filter out common non-document API endpoints
            lower = url.lower()
            if any(skip in lower for skip in [
                "analytics", "tracking", "cookie", "consent", "login",
                "auth", "session", "cart", "search", "navigation",
                "manifest.json", "package.json", "tsconfig.json",
            ]):
                continue
            api_urls.append(url)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in api_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    return unique[:5]  # Limit to 5 most promising


async def _try_fetch_api_endpoints(
    client: httpx.AsyncClient,
    api_urls: list[str],
    ticker: str,
) -> str:
    """Try fetching discovered API endpoints and return any useful JSON data."""
    extra_content = ""
    for api_url in api_urls:
        try:
            resp = await client.get(api_url, headers={
                **_HEADERS,
                "Accept": "application/json, text/plain, */*",
            })
            if resp.status_code == 200 and len(resp.text) > 100:
                content_type = resp.headers.get("content-type", "")
                if "json" in content_type or resp.text.strip().startswith(("{", "[")):
                    chunk = resp.text[:3000]
                    extra_content += f"\n\n--- API DATA: {api_url} ---\n{chunk}"
                    logger.info("[LLM-SCRAPE] %s — fetched API endpoint: %s (%d chars)",
                                ticker, api_url, len(resp.text))
        except Exception:
            pass
    return extra_content


async def _try_ssr_fetch(
    client: httpx.AsyncClient,
    url: str,
    ticker: str,
) -> str | None:
    """
    Re-fetch the page with Googlebot-like headers. Some SPA sites serve
    pre-rendered HTML to crawlers.
    """
    try:
        resp = await client.get(url, headers=_SSR_HEADERS)
        if resp.status_code == 200:
            cleaned = _clean_page_source(resp.text)
            visible = _extract_visible_text(cleaned)
            if len(visible) > _THIN_CONTENT_THRESHOLD:
                logger.info("[LLM-SCRAPE] %s — SSR fetch returned richer content (%d chars)",
                            ticker, len(visible))
                return cleaned
    except Exception:
        pass
    return None


_LLM_PROMPT = """\
You are analysing an investor relations web page to find financial documents.

COMPANY: {company} ({ticker})
PAGE URL: {url}

Below is the HTML source of the page (may contain embedded JSON data with escaped HTML).
Find ALL downloadable financial documents — earnings releases, presentations, transcripts,
annual reports, quarterly reports, press releases, proxy statements.

For each document found, extract:
- url: the full download URL (resolve relative URLs using the base: {base})
- title: descriptive title
- document_type: one of: earnings_release, transcript, presentation, annual_report, proxy_statement, other
- period: the reporting period in format YYYY_Q1, YYYY_Q2, YYYY_Q3, YYYY_Q4, YYYY_FY, YYYY_H1, or empty if unclear
- date: publication date if visible (YYYY-MM-DD format) or empty

Important:
- Look for URLs ending in .pdf but also check for document links in JSON data
- Unicode-escaped URLs like \\u0022https://...\\u0022 should be decoded
- Only include documents from the last 3 years
- Do NOT include images, CSS, JS files, or navigation links

PAGE SOURCE:
---
{html}
---

Respond ONLY with a JSON array. No preamble, no markdown fences.
[
  {{"url": "...", "title": "...", "document_type": "...", "period": "...", "date": "..."}}
]

If no documents found, return an empty array: []"""


async def scrape_ir_with_llm(
    ticker: str,
    company_name: str,
    ir_docs_url: str,
) -> list[dict]:
    """
    Fetch an IR page and use the LLM to extract document links.

    Returns a list of HarvestCandidate dicts.
    """
    from services.llm_client import call_llm_json_async

    base = f"{urlparse(ir_docs_url).scheme}://{urlparse(ir_docs_url).netloc}"

    # Fetch the page (with Cloudflare bypass)
    from services.doc_utils import async_fetch_page

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers=_HEADERS,
    ) as client:
        raw_html = await async_fetch_page(ir_docs_url, timeout=20)
        if not raw_html:
            logger.warning("[LLM-SCRAPE] Failed to fetch %s", ir_docs_url)
            return []
        cleaned = _clean_page_source(raw_html)

        # ── JS-rendered page detection ────────────────────────────────
        visible_text = _extract_visible_text(cleaned)
        is_thin_content = len(visible_text) < _THIN_CONTENT_THRESHOLD

        if is_thin_content:
            logger.warning(
                "[LLM-SCRAPE] %s — thin content detected (%d visible chars), "
                "page may be JS-rendered: %s",
                ticker, len(visible_text), ir_docs_url,
            )

            # Strategy 1: Look for API/JSON endpoints in the HTML source
            api_urls = _find_api_urls(raw_html, ir_docs_url)
            if api_urls:
                logger.info("[LLM-SCRAPE] %s — found %d API endpoint(s) to try",
                            ticker, len(api_urls))
                api_content = await _try_fetch_api_endpoints(client, api_urls, ticker)
                if api_content:
                    cleaned += api_content

            # Strategy 2: Try SSR fetch with Googlebot headers
            ssr_content = await _try_ssr_fetch(client, ir_docs_url, ticker)
            if ssr_content and len(ssr_content) > len(cleaned):
                logger.info("[LLM-SCRAPE] %s — using SSR content instead", ticker)
                cleaned = ssr_content

        if len(cleaned) > _MAX_PAGE_CHARS:
            cleaned = cleaned[:_MAX_PAGE_CHARS]

        # ── Sibling pages (skip if main page already has enough) ──────
        if len(cleaned) < _SKIP_SIBLINGS_THRESHOLD:
            parsed = urlparse(ir_docs_url)
            path = parsed.path.rstrip('/')
            parent = '/'.join(path.split('/')[:-1])
            siblings = ["previous-results", "results", "reports", "results-archive"]
            for sib in siblings:
                sib_url = f"{base}{parent}/{sib}"
                if sib_url == ir_docs_url:
                    continue
                try:
                    sib_resp = await client.get(sib_url)
                    if sib_resp.status_code == 200:
                        sib_cleaned = _clean_page_source(sib_resp.text)
                        remaining = _MAX_PAGE_CHARS - len(cleaned)
                        if remaining > 2000:
                            cleaned += f"\n\n--- SIBLING PAGE: {sib_url} ---\n" + sib_cleaned[:remaining]
                            logger.info("[LLM-SCRAPE] Also fetched sibling: %s", sib_url)
                except Exception:
                    pass
        else:
            logger.info("[LLM-SCRAPE] %s — main page has %d chars, skipping sibling fetch",
                        ticker, len(cleaned))

    logger.info("[LLM-SCRAPE] %s — sending %d chars to LLM", ticker, len(cleaned))

    # Ask the LLM (with timeout for Railway 30s gateway limit)
    prompt = _LLM_PROMPT.format(
        company=company_name,
        ticker=ticker,
        url=ir_docs_url,
        base=base,
        html=cleaned,
    )

    try:
        documents = await asyncio.wait_for(
            call_llm_json_async(prompt, max_tokens=2048, model="claude-haiku-4-5-20251001", feature="llm_scan"),
            timeout=25.0,
        )
    except asyncio.TimeoutError:
        logger.error("[LLM-SCRAPE] %s — LLM call timed out (25s limit for Railway)", ticker)
        return []
    except Exception as e:
        logger.error("[LLM-SCRAPE] LLM call failed for %s: %s", ticker, str(e)[:200])
        return []

    if not isinstance(documents, list):
        logger.warning("[LLM-SCRAPE] LLM returned non-list: %s", type(documents))
        return []

    # Convert to HarvestCandidate format
    candidates = []
    seen_urls = set()
    for doc in documents:
        url = doc.get("url", "").strip()
        if not url or url in seen_urls:
            continue
        # Resolve relative URLs
        if not url.startswith("http"):
            url = urljoin(base, url)
        seen_urls.add(url)

        title = doc.get("title", "").strip()
        doc_type = doc.get("document_type", "other")
        period = doc.get("period", "")
        date_str = doc.get("date", "")

        # Validate document_type
        valid_types = {"earnings_release", "transcript", "presentation", "annual_report",
                       "proxy_statement", "broker_note", "10-K", "10-Q", "other"}
        if doc_type not in valid_types:
            doc_type = "other"

        candidates.append({
            "ticker": ticker,
            "source": "ir_llm",
            "source_url": url,
            "headline": title[:300] or url.split("/")[-1],
            "description": "",
            "published_at": datetime.now(timezone.utc),
            "pdf_url": url,
            "period_label": period,
            "document_type": doc_type,
            "date": date_str,
        })

    logger.info("[LLM-SCRAPE] %s — LLM found %d documents", ticker, len(candidates))
    return candidates
