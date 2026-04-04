"""
IR Website Discovery Agent

For any company (name + ticker), uses the Claude API to reason about
the most likely IR website URL, then validates it and attempts to find
an RSS/Atom feed programmatically.

This replaces the hardcoded IR_RSS_SOURCES dict — any company added
to the DB is automatically discoverable.

Discovery results are cached in the harvester_sources DB table so
Claude is only called once per company.
"""

import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from configs.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Known RSS patterns to try once we have the IR base URL
# ─────────────────────────────────────────────────────────────────
_RSS_PATH_CANDIDATES = [
    "/rss",
    "/rss.xml",
    "/feed",
    "/feed.xml",
    "/feeds/press-releases",
    "/feeds/news",
    "/news/rss",
    "/news/rss.xml",
    "/investors/rss",
    "/investors/news/rss",
    "/investors/press-releases/rss",
    "/media/press-releases/rss",
    "/media/news/rss",
    "/press-releases/rss",
    "/press-releases/rss.xml",
    "/regulatory-news/rss",
    "/en/investors/rss",
    "/en/media/press-releases/rss",
    "/en-IT/media/press-release/feed.xml",
    "/investor-relations/rss",
    "/investor-centre/news/rss",
]

_RSS_CONTENT_TYPES = {
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ─────────────────────────────────────────────────────────────────
# Step 1: Ask Claude to reason about the IR website URL
# ─────────────────────────────────────────────────────────────────

async def discover_ir_url_via_llm(company_name: str, ticker: str, country: Optional[str] = None) -> Optional[str]:
    """
    Ask Claude to infer the most likely IR website base URL for a company.
    Returns a URL string or None.

    Claude is prompted to reason step by step and return only the URL.
    This is fast (single short completion) and cheap.
    """
    country_hint = f" The company is listed/headquartered in {country}." if country else ""
    ticker_clean = ticker.split()[0]  # strip exchange suffix (e.g. "BNZL" from "BNZL LN")

    prompt = f"""You are helping locate the investor relations website for a publicly listed company.

Company name: {company_name}
Ticker: {ticker_clean}{country_hint}

Task: Return the base URL of this company's investor relations (IR) section.
This is typically at one of these patterns:
- https://www.companyname.com/investors
- https://www.companyname.com/investor-relations  
- https://ir.companyname.com
- https://investors.companyname.com
- https://corporate.companyname.com/investors

Rules:
1. Return ONLY the URL, nothing else.
2. Do not include trailing slashes.
3. If uncertain between two options, pick the more likely one.
4. Do not guess wildly — if you have no reasonable basis for the URL, return: UNKNOWN
5. Do not include any explanation.

Examples:
- Heineken → https://www.theheinekencompany.com/investors
- Bunzl → https://www.bunzl.com/investors
- Pason Systems → https://www.pason.com/investor-centre
- LKQ Corporation → https://investor.lkqcorp.com

URL:"""

    try:
        import anthropic
        from services.llm_client import _log_usage
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        message = await client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheap + fast for this task
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        _log_usage("claude-haiku-4-5-20251001", message.usage.input_tokens, message.usage.output_tokens,
                   feature="ir_discovery", ticker=ticker)
        url = message.content[0].text.strip().rstrip("/")

        if url == "UNKNOWN" or not url.startswith("http"):
            logger.info("[DISCOVER] Claude returned no URL for %s (%s)", company_name, ticker)
            return None

        # Basic sanity check
        parsed = urlparse(url)
        if not parsed.netloc:
            return None

        logger.info("[DISCOVER] Claude suggested IR URL for %s: %s", ticker, url)
        return url

    except Exception as exc:
        logger.warning("[DISCOVER] LLM discovery failed for %s: %s", ticker, exc)
        return None


# ─────────────────────────────────────────────────────────────────
# Step 2: Validate URL is reachable
# ─────────────────────────────────────────────────────────────────

async def validate_url(url: str, client: httpx.AsyncClient) -> bool:
    """Check that a URL returns a 2xx or 3xx response."""
    try:
        resp = await client.head(url, headers=_HEADERS, follow_redirects=True, timeout=8.0)
        return resp.status_code < 400
    except Exception:
        try:
            # HEAD sometimes blocked — try GET
            resp = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=8.0)
            return resp.status_code < 400
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────
# Step 3: Find RSS feed from IR base URL
# ─────────────────────────────────────────────────────────────────

def _extract_base(url: str) -> str:
    """Extract scheme + netloc from a URL."""
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


async def _find_rss_in_page(page_url: str, page_html: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Parse an HTML page looking for <link> tags pointing to RSS/Atom feeds,
    or <a href> links containing 'rss', 'feed', 'atom'.
    """
    base = _extract_base(page_url)

    # <link type="application/rss+xml" href="...">
    link_matches = re.findall(
        r'<link[^>]+(?:application/rss\+xml|application/atom\+xml)[^>]+href=["\']([^"\']+)["\']',
        page_html, re.IGNORECASE
    )
    link_matches += re.findall(
        r'<link[^>]+href=["\']([^"\']+)["\'][^>]+(?:application/rss\+xml|application/atom\+xml)',
        page_html, re.IGNORECASE
    )

    # <a href="...rss..."> or <a href="...feed...">
    a_matches = re.findall(
        r'href=["\']([^"\']*(?:rss|feed|atom)[^"\']*)["\']',
        page_html, re.IGNORECASE
    )

    candidates = link_matches + a_matches

    for href in candidates:
        if not href.startswith("http"):
            href = urljoin(base, href)
        if await _is_valid_rss(href, client):
            return href

    return None


async def _is_valid_rss(url: str, client: httpx.AsyncClient) -> bool:
    """Fetch a URL and check it looks like an RSS/Atom feed."""
    try:
        resp = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=8.0)
        if resp.status_code >= 400:
            return False
        ct = resp.headers.get("content-type", "").lower()
        body = resp.text[:2000]

        is_xml_ct = any(t in ct for t in _RSS_CONTENT_TYPES)
        is_feed_body = "<rss" in body or "<feed" in body or "<channel>" in body

        return is_xml_ct or is_feed_body
    except Exception:
        return False


async def find_rss_feed(ir_base_url: str) -> Optional[str]:
    """
    Given an IR base URL, attempt to discover an RSS/Atom feed by:
    1. Trying known path patterns
    2. Fetching the IR page and scanning for <link> feed tags or feed hrefs
    """
    base = _extract_base(ir_base_url)

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:

        # Try known path patterns first (fast)
        for path in _RSS_PATH_CANDIDATES:
            candidate = base + path
            if await _is_valid_rss(candidate, client):
                logger.info("[DISCOVER] RSS found via path probe: %s", candidate)
                return candidate

        # Also try appending paths to the IR sub-path
        if ir_base_url != base:
            ir_path = urlparse(ir_base_url).path.rstrip("/")
            for suffix in ["/rss", "/rss.xml", "/news/rss", "/press-releases/rss", "/feed"]:
                candidate = base + ir_path + suffix
                if await _is_valid_rss(candidate, client):
                    logger.info("[DISCOVER] RSS found via IR path probe: %s", candidate)
                    return candidate

        # Fetch the IR page and scan HTML for feed links
        try:
            resp = await client.get(ir_base_url, headers=_HEADERS, timeout=10.0)
            if resp.status_code < 400:
                rss_url = await _find_rss_in_page(ir_base_url, resp.text, client)
                if rss_url:
                    logger.info("[DISCOVER] RSS found via HTML scan: %s", rss_url)
                    return rss_url
        except Exception as exc:
            logger.debug("[DISCOVER] Could not fetch IR page %s: %s", ir_base_url, exc)

    logger.info("[DISCOVER] No RSS feed found for IR URL: %s", ir_base_url)
    return None


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

async def discover_company_sources(
    company_name: str,
    ticker: str,
    country: Optional[str] = None,
) -> dict:
    """
    Full discovery flow for a company.

    Returns:
    {
        "ticker": str,
        "ir_url": str | None,
        "rss_url": str | None,
        "ir_reachable": bool,
        "discovery_method": str,   # "llm+probe" | "llm_only" | "failed"
    }
    """
    result = {
        "ticker": ticker,
        "ir_url": None,
        "rss_url": None,
        "ir_reachable": False,
        "discovery_method": "failed",
    }

    # Step 1: Ask Claude for the IR URL
    ir_url = await discover_ir_url_via_llm(company_name, ticker, country)
    if not ir_url:
        return result

    result["ir_url"] = ir_url

    # Step 2: Validate it's reachable
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        reachable = await validate_url(ir_url, client)

    result["ir_reachable"] = reachable
    if not reachable:
        logger.info("[DISCOVER] IR URL not reachable for %s: %s", ticker, ir_url)
        result["discovery_method"] = "llm_only"
        return result

    # Step 3: Find RSS feed
    rss_url = await find_rss_feed(ir_url)
    result["rss_url"] = rss_url
    result["discovery_method"] = "llm+probe"

    return result
