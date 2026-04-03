"""
Investegate RNS Harvester

Polls investegate.co.uk for UK regulatory announcements (RNS, MFN, etc).
Equivalent to the SEC EDGAR source but for UK-listed companies.

Strategy:
  1. Hit /company/{EPIC}/announcements — returns HTML table of recent filings
  2. Filter by headline keywords (results, trading update, annual report, etc.)
  3. Return HarvestCandidates pointing to the announcement URL
  4. Dispatcher downloads the full HTML text and ingests via the standard pipeline

No API key required. Investegate serves plain HTML, no Cloudflare.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Per-company config — epic is the Investegate/LSE ticker symbol
# ─────────────────────────────────────────────────────────────────
INVESTEGATE_SOURCES: dict[str, dict] = {
    "BNZL LN": {"epic": "BNZL"},
    "EZJ LN":  {"epic": "EZJ"},
    "JDW LN":  {"epic": "JDW"},
    "LLOY LN": {"epic": "LLOY"},
    "TSCO LN": {"epic": "TSCO"},
    "WTB LN":  {"epic": "WTB"},
    # BP and Shell also file via EDGAR (primary source), but Investegate
    # catches UK-specific announcements EDGAR misses (trading updates etc.)
    "BP LN":   {"epic": "BP."},
    "SHEL LN": {"epic": "SHEL"},
}

# ─────────────────────────────────────────────────────────────────
# Headline filters — only harvest results-relevant announcements
# ─────────────────────────────────────────────────────────────────
_RESULTS_KEYWORDS = [
    "final results", "preliminary results", "full year results",
    "half-year", "half year", "interim results", "interim report",
    "interim management statement",
    "trading update", "trading statement",
    "annual report", "annual results",
    "quarterly results", "quarterly report", "quarterly update",
    "q1 ", "q2 ", "q3 ", "q4 ",
    "q1&", "q2&", "q3&", "q4&",     # HTML-encoded ampersand in titles
    "1q ", "2q ", "3q ", "4q ",
    "results for", "results to",
    "form 20-f", "form 20f",
]

# Headlines to always skip even if they match a keyword
_SKIP_KEYWORDS = [
    "director", "pdmr", "shareholding", "holding(s)",
    "block listing", "transaction in own", "total voting",
    "form 8.3", "tr-1",
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "text/html, */*",
}

# ─────────────────────────────────────────────────────────────────
# Period inference from headline text
# ─────────────────────────────────────────────────────────────────
_PERIOD_PATTERNS = [
    # Q3 2025, Q1 2026
    (r"Q([1-4])\s*[\-&]?\s*(\d{4})", lambda m: f"{m.group(2)}_Q{m.group(1)}"),
    # 4Q25, 1Q26
    (r"([1-4])Q(\d{2})(?:\s|$)", lambda m: f"20{m.group(2)}_Q{m.group(1)}"),
    # FY2025, FY 2024
    (r"FY\s*(\d{4})", lambda m: f"{m.group(1)}_Q4"),
    # Year-first: "2025 Annual Report", "2025 Full Year Results"
    (r"(\d{4})\s+(?:annual|full[- ]?year)", lambda m: f"{m.group(1)}_Q4"),
    # Year-first: "2025 Half Year", "2025 Interim"
    (r"(\d{4})\s+(?:half[- ]?year|interim)", lambda m: f"{m.group(1)}_Q2"),
    # Full year / annual + year (keyword first)
    (r"(?:full[- ]?year|annual)\s+(?:results?\s+)?(?:report\s+)?(?:and\s+)?(?:accounts?\s+)?(?:for\s+)?(?:the\s+)?(?:year\s+)?(?:ended?\s+)?.*?(\d{4})",
     lambda m: f"{m.group(1)}_Q4"),
    # Half year / interim + year (keyword first)
    (r"(?:half[- ]?year|interim)\s+(?:results?\s+)?(?:report\s+)?(?:for\s+)?(?:the\s+)?.*?(\d{4})",
     lambda m: f"{m.group(1)}_Q2"),
    # H1 2025, H2 2024
    (r"H1\s*(\d{4})", lambda m: f"{m.group(1)}_Q2"),
    (r"H2\s*(\d{4})", lambda m: f"{m.group(1)}_Q4"),
    # "2025/26" in trading statements — use first year
    (r"(\d{4})/\d{2}", lambda m: f"{m.group(1)}_Q4"),
]


def _infer_period(title: str, date_str: str) -> Optional[str]:
    """Infer period label from announcement title, falling back to date."""
    for pattern, fmt in _PERIOD_PATTERNS:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            result = fmt(m)
            year_m = re.search(r"(\d{4})", result)
            if year_m and 2010 <= int(year_m.group(1)) <= 2030:
                return result

    # Fallback: infer from announcement date
    try:
        dt = datetime.strptime(date_str, "%d %b %Y")
        q = ((dt.month - 1) // 3) + 1
        return f"{dt.year}_Q{q}"
    except ValueError:
        return None


def _classify_doc_type(title: str) -> str:
    """Classify announcement into document type."""
    t = title.lower()
    if any(k in t for k in ["annual report", "annual results", "form 20-f"]):
        return "annual_report"
    if any(k in t for k in ["interim", "half-year", "half year"]):
        return "earnings_release"
    if any(k in t for k in ["trading update", "trading statement"]):
        return "earnings_release"
    if any(k in t for k in ["final results", "preliminary results", "full year"]):
        return "earnings_release"
    if "form 20" in t:
        return "annual_report"
    if any(k in t for k in ["quarterly"]):
        return "earnings_release"
    return "other"


def _is_relevant(title: str) -> bool:
    """Check if an announcement headline is results-relevant."""
    t = title.lower()
    # Skip noise first
    if any(k in t for k in _SKIP_KEYWORDS):
        return False
    return any(k in t for k in _RESULTS_KEYWORDS)


# ─────────────────────────────────────────────────────────────────
# Main fetch function
# ─────────────────────────────────────────────────────────────────

async def fetch_investegate(ticker: str, max_pages: int = 10) -> list[dict]:
    """
    Fetch recent RNS announcements from Investegate for a UK-listed company.

    Returns a list of HarvestCandidate dicts compatible with dispatcher.py.
    Scans up to max_pages of announcements (50 per page).
    High-volume filers like LLOY post daily share buyback notices that
    push results announcements deep into the pagination.
    """
    config = INVESTEGATE_SOURCES.get(ticker)
    if not config:
        logger.debug("No Investegate config for ticker %s", ticker)
        return []

    epic = config["epic"]
    candidates = []
    seen_titles = set()  # dedup by normalised headline
    cutoff_year = datetime.now(timezone.utc).year - 2

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=_HEADERS) as client:
        for page in range(1, max_pages + 1):
            url = f"https://www.investegate.co.uk/company/{epic}/announcements"
            if page > 1:
                url += f"?page={page}"

            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("[%s] Investegate fetch failed (page %d): %s", ticker, page, exc)
                break

            html = resp.text
            if "/announcement/" not in html:
                break  # No more announcements

            # Parse table rows
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
            page_count = 0

            for row in rows:
                date_match = re.search(r"<td>(\d+ \w+ \d+)</td>", row)
                link_match = re.search(
                    r'<a class="announcement-link" href="([^"]+)">([^<]+)</a>', row
                )
                if not date_match or not link_match:
                    continue

                date_str = date_match.group(1)
                ann_url = link_match.group(1)
                title = link_match.group(2).replace("&amp;", "&").strip()

                # Year cutoff
                try:
                    ann_year = int(date_str.split()[-1])
                    if ann_year < cutoff_year:
                        # Past cutoff — stop pagination
                        page = max_pages  # break outer loop
                        break
                except (ValueError, IndexError):
                    pass

                if not _is_relevant(title):
                    continue

                # Dedup: skip if we already have this headline
                # (e.g. annual report filed as multiple RNS parts)
                norm_title = re.sub(r"[^a-z0-9]", "", title.lower())
                if norm_title in seen_titles:
                    continue
                seen_titles.add(norm_title)

                # Ensure full URL
                if not ann_url.startswith("http"):
                    ann_url = f"https://www.investegate.co.uk{ann_url}"

                period_label = _infer_period(title, date_str)
                doc_type = _classify_doc_type(title)

                # Parse published_at
                try:
                    published_at = datetime.strptime(date_str, "%d %b %Y").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    published_at = datetime.now(timezone.utc)

                candidates.append({
                    "ticker": ticker,
                    "source": "investegate",
                    "source_url": ann_url,              # dedup key
                    "headline": title[:300],
                    "description": f"RNS announcement {date_str}",
                    "published_at": published_at,
                    "pdf_url": ann_url,                  # dispatcher downloads HTML
                    "period_label": period_label,
                    "document_type": doc_type,
                })
                page_count += 1

            # Don't break early on empty pages — high-volume filers
            # (e.g. LLOY with daily buyback notices) may have results
            # announcements scattered across many pages.

    logger.info("[%s] Investegate found %d relevant announcements", ticker, len(candidates))
    return candidates
