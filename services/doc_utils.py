"""
Shared utilities for document title cleaning, URL normalisation, dedup,
and Cloudflare-aware page fetching.
Used by harvester dispatcher, ingest endpoint, and UI display.
"""

import logging
import re
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

logger = logging.getLogger(__name__)


def fetch_page(url: str, timeout: int = 20) -> str:
    """
    Fetch a web page. Uses httpx first, falls back to cloudscraper
    if Cloudflare bot protection is detected.
    Returns the page HTML as a string.
    """
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    }

    # Try httpx first (async-compatible, fast)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            # Check for Cloudflare block
            if "you have been blocked" in resp.text.lower() or "cf-chl-bypass" in resp.text.lower() or (len(resp.text) < 5000 and "challenge-platform" in resp.text):
                logger.info("[FETCH] Cloudflare detected on %s — trying cloudscraper", url)
                raise _CloudflareBlocked()
            return resp.text
    except _CloudflareBlocked:
        pass
    except Exception as e:
        logger.debug("[FETCH] httpx failed for %s: %s", url, str(e)[:100])

    # Fallback to cloudscraper
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper()
        resp = scraper.get(url, timeout=timeout)
        resp.raise_for_status()
        logger.info("[FETCH] cloudscraper succeeded for %s (%d bytes)", url, len(resp.text))
        return resp.text
    except Exception as e:
        logger.warning("[FETCH] cloudscraper also failed for %s: %s", url, str(e)[:100])
        return ""


class _CloudflareBlocked(Exception):
    pass


async def async_fetch_page(url: str, timeout: int = 20) -> str:
    """Async wrapper for fetch_page — runs in thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fetch_page(url, timeout))


def normalise_url(url: str) -> str:
    """
    Normalise a URL for dedup comparison.
    Strips query params, fragments, trailing slashes, and common tracking params.
    """
    if not url:
        return ""
    parsed = urlparse(url)
    # Strip fragment
    # Strip common tracking/cache-busting query params
    params = parse_qs(parsed.query)
    skip_params = {"v", "version", "utm_source", "utm_medium", "utm_campaign",
                   "cache", "t", "ts", "timestamp", "cb", "nocache", "ref"}
    cleaned_params = {k: v for k, v in params.items() if k.lower() not in skip_params}
    clean_query = urlencode(cleaned_params, doseq=True) if cleaned_params else ""
    # Rebuild without fragment, with cleaned query, strip trailing slash from path
    path = parsed.path.rstrip("/")
    normalised = urlunparse((parsed.scheme, parsed.netloc, path, "", clean_query, ""))
    return normalised


# ── Title cleaning ─────────────────────────────────────────────

def _strip_version_suffixes(title: str) -> str:
    """Remove version/status suffixes like _v2, _FINAL, _AMENDED from end of title."""
    suffixes = re.compile(r'[_\- ]+(final|draft|amended|revised|updated?|copy|new)\s*$', re.IGNORECASE)
    version = re.compile(r'[_\- ]+v(\d+)\s*$')  # case-sensitive: only lowercase v + digits
    prev = None
    while prev != title:
        prev = title
        title = suffixes.sub('', title)
        title = version.sub('', title)
    return title

_NOISE_PATTERNS = [
    (re.compile(r'[_\- ]*(?:pdf|htm|html|xlsx?|docx?|pptx?)$', re.I), ''),  # file extensions in title
    (re.compile(r'^(?:download|document|file|attachment)\s*$', re.I), ''),    # generic names
    (re.compile(r'\s+', re.I), ' '),                           # collapse whitespace
]

_LANG_SUFFIXES = re.compile(
    r'[_\- ]+(?:en|fr|de|es|it|pt|nl|ja|ko|zh|sv)$',
    re.IGNORECASE
)


def clean_title(title: str) -> str:
    """
    Clean a document title:
    - Strip version suffixes (_v2, _FINAL, _AMENDED, _draft)
    - Strip language suffixes (_fr, _de, _en)
    - Strip file extensions
    - Strip trailing numbers
    - Collapse whitespace
    - Title case if all caps or all lower
    """
    if not title:
        return title

    t = title.strip()

    # Strip version suffixes
    t = _strip_version_suffixes(t)

    # Strip language suffixes
    t = _LANG_SUFFIXES.sub('', t)

    # Apply noise patterns
    for pattern, replacement in _NOISE_PATTERNS:
        t = pattern.sub(replacement, t)

    t = t.strip()

    # Title case if all uppercase or all lowercase
    if t == t.upper() or t == t.lower():
        t = t.title()

    return t


def detect_language(title: str, url: str = "") -> str | None:
    """
    Detect non-English language from title or URL.
    Returns ISO code (fr, de, es, it, etc.) or None if English/unknown.
    """
    combined = f"{title} {url}".lower()

    # Check URL path for language codes
    lang_in_url = re.search(r'/(?:fr|de|es|it|pt|nl|ja|ko|zh|sv)/', combined)
    if lang_in_url:
        return lang_in_url.group(0).strip('/')

    # Check filename suffix
    lang_suffix = re.search(r'[-_](fr|de|es|it|pt|nl|ja|ko|zh|sv)\.(?:pdf|htm)', combined)
    if lang_suffix:
        return lang_suffix.group(1)

    # Check for common non-English words in title
    fr_words = ["résultats", "rapport", "annuel", "trimestriel", "semestriel", "communiqué"]
    de_words = ["ergebnis", "bericht", "geschäftsbericht", "quartal", "halbjahr"]
    es_words = ["resultados", "informe", "trimestral", "anual"]
    it_words = ["risultati", "relazione", "trimestrale", "annuale"]

    for word in fr_words:
        if word in combined:
            return "fr"
    for word in de_words:
        if word in combined:
            return "de"
    for word in es_words:
        if word in combined:
            return "es"
    for word in it_words:
        if word in combined:
            return "it"

    return None
