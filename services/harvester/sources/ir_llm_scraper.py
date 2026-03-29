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

# Max chars of page source to send to LLM (keeps cost down)
_MAX_PAGE_CHARS = 30000


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
    from services.llm_client import call_llm_json

    base = f"{urlparse(ir_docs_url).scheme}://{urlparse(ir_docs_url).netloc}"

    # Fetch the page
    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers=_HEADERS,
    ) as client:
        try:
            resp = await client.get(ir_docs_url)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("[LLM-SCRAPE] Failed to fetch %s: %s", ir_docs_url, e)
            return []

    # Clean and truncate
    raw_html = resp.text
    cleaned = _clean_page_source(raw_html)
    if len(cleaned) > _MAX_PAGE_CHARS:
        cleaned = cleaned[:_MAX_PAGE_CHARS]

    # Also check sibling pages
    parsed = urlparse(ir_docs_url)
    path = parsed.path.rstrip('/')
    parent = '/'.join(path.split('/')[:-1])
    siblings = ["previous-results", "results", "reports", "results-archive"]
    for sib in siblings:
        sib_url = f"{base}{parent}/{sib}"
        if sib_url == ir_docs_url:
            continue
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=_HEADERS) as client:
                sib_resp = await client.get(sib_url)
                if sib_resp.status_code == 200:
                    sib_cleaned = _clean_page_source(sib_resp.text)
                    remaining = _MAX_PAGE_CHARS - len(cleaned)
                    if remaining > 5000:
                        cleaned += f"\n\n--- SIBLING PAGE: {sib_url} ---\n" + sib_cleaned[:remaining]
                        logger.info("[LLM-SCRAPE] Also fetched sibling: %s", sib_url)
        except Exception:
            pass

    logger.info("[LLM-SCRAPE] %s — sending %d chars to LLM", ticker, len(cleaned))

    # Ask the LLM
    prompt = _LLM_PROMPT.format(
        company=company_name,
        ticker=ticker,
        url=ir_docs_url,
        base=base,
        html=cleaned,
    )

    try:
        documents = call_llm_json(prompt, max_tokens=4096)
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
