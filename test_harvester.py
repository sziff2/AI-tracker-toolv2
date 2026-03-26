"""
Standalone Harvester Test Script
=================================
Tests the three core harvester functions WITHOUT needing the full app,
database, Celery, or Docker running.

Requirements (install once):
    pip install httpx anthropic

Usage:
    # Set your API key first:
    set ANTHROPIC_API_KEY=sk-ant-...          (Windows CMD)
    $env:ANTHROPIC_API_KEY="sk-ant-..."       (PowerShell)

    # Run all tests:
    python test_harvester.py

    # Test a specific company:
    python test_harvester.py --company "Bunzl" --ticker "BNZL LN" --country "UK"

    # Skip the LLM call, test RSS with a known URL directly:
    python test_harvester.py --rss-url "https://www.bunzl.com/investors/regulatory-news/rss"

    # Test EDGAR only:
    python test_harvester.py --edgar --ticker "LKQX"
"""

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import httpx

# ─────────────────────────────────────────────────────────────────
# Inline copies of the key functions — no app imports needed
# ─────────────────────────────────────────────────────────────────

_RSS_PATH_CANDIDATES = [
    "/rss", "/rss.xml", "/feed", "/feed.xml",
    "/feeds/press-releases", "/feeds/news",
    "/news/rss", "/news/rss.xml",
    "/investors/rss", "/investors/news/rss", "/investors/press-releases/rss",
    "/media/press-releases/rss", "/media/news/rss",
    "/press-releases/rss", "/press-releases/rss.xml",
    "/regulatory-news/rss",
    "/en/investors/rss", "/en/media/press-releases/rss",
    "/en-IT/media/press-release/feed.xml",
    "/investor-relations/rss", "/investor-centre/news/rss",
]

_RESULTS_KEYWORDS = [
    "results", "earnings", "revenue", "revenues", "profit",
    "quarterly", "annual", "full year", "half year", "interim",
    "trading update", "Q1", "Q2", "Q3", "Q4", "H1", "H2",
    "10-K", "10-Q", "20-F", "40-F", "6-K", "production",
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
    "Accept": "application/rss+xml,application/atom+xml,application/xml,text/xml,*/*",
}

_NS = {"atom": "http://www.w3.org/2005/Atom"}

EDGAR_BASE = "https://data.sec.gov"


# ── Helpers ───────────────────────────────────────────────────────

def _extract_base(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _is_relevant(title: str, description: str = "") -> bool:
    text = (title + " " + description).lower()
    return any(kw.lower() in text for kw in _RESULTS_KEYWORDS)


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(s.strip(), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


_PERIOD_PATTERNS = [
    (r"full[- ]?year\s+(\d{4})",           lambda m: f"{m.group(1)}_FY"),
    (r"(\d{4})\s+full[- ]?year",           lambda m: f"{m.group(1)}_FY"),
    (r"annual\s+(?:results\s+)?(\d{4})",   lambda m: f"{m.group(1)}_FY"),
    (r"half[- ]?year\s+(\d{4})",           lambda m: f"{m.group(1)}_H1"),
    (r"H1\s+(\d{4})",                      lambda m: f"{m.group(1)}_H1"),
    (r"H2\s+(\d{4})",                      lambda m: f"{m.group(1)}_H2"),
    (r"Q([1-4])\s+(\d{4})",               lambda m: f"{m.group(2)}_Q{m.group(1)}"),
    (r"(\d{4})\s+Q([1-4])",               lambda m: f"{m.group(1)}_Q{m.group(2)}"),
    (r"(?:first|1st)\s+quarter\s+(\d{4})", lambda m: f"{m.group(1)}_Q1"),
    (r"(?:second|2nd)\s+quarter\s+(\d{4})",lambda m: f"{m.group(1)}_Q2"),
    (r"(?:third|3rd)\s+quarter\s+(\d{4})", lambda m: f"{m.group(1)}_Q3"),
    (r"(?:fourth|4th)\s+quarter\s+(\d{4})",lambda m: f"{m.group(1)}_Q4"),
]


def _infer_period(title: str) -> Optional[str]:
    for pattern, fmt in _PERIOD_PATTERNS:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            return fmt(m)
    return None


# ── Test 1: LLM Discovery ─────────────────────────────────────────

async def test_llm_discovery(company_name: str, ticker: str, country: Optional[str] = None) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ✗ ANTHROPIC_API_KEY not set — skipping LLM test")
        return None

    ticker_clean = ticker.split()[0]
    country_hint = f" The company is listed/headquartered in {country}." if country else ""

    prompt = f"""You are helping locate the investor relations website for a publicly listed company.

Company name: {company_name}
Ticker: {ticker_clean}{country_hint}

Return ONLY the base URL of this company's investor relations section. No explanation.
If uncertain, return: UNKNOWN

URL:"""

    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        url = msg.content[0].text.strip().rstrip("/")
        return None if (url == "UNKNOWN" or not url.startswith("http")) else url
    except Exception as e:
        print(f"  ✗ LLM call failed: {e}")
        return None


# ── Test 2: RSS probe ─────────────────────────────────────────────

async def _is_valid_rss(url: str, client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=8.0)
        if r.status_code >= 400:
            return False
        ct = r.headers.get("content-type", "").lower()
        body = r.text[:2000]
        return any(t in ct for t in ["rss", "atom", "xml"]) or "<rss" in body or "<feed" in body
    except Exception:
        return False


async def test_rss_probe(ir_base_url: str) -> Optional[str]:
    base = _extract_base(ir_base_url)
    ir_path = urlparse(ir_base_url).path.rstrip("/")

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        # Try standard paths
        for path in _RSS_PATH_CANDIDATES:
            for prefix in [base, base + ir_path]:
                candidate = prefix + path
                if await _is_valid_rss(candidate, client):
                    return candidate

        # Scan HTML for feed links
        try:
            r = await client.get(ir_base_url, headers=_HEADERS, timeout=10.0)
            if r.status_code < 400:
                # Look for <link type="application/rss+xml"> tags
                for match in re.finditer(
                    r'<link[^>]+(?:rss\+xml|atom\+xml)[^>]+href=["\']([^"\']+)["\']',
                    r.text, re.IGNORECASE
                ):
                    href = match.group(1)
                    if not href.startswith("http"):
                        href = urljoin(base, href)
                    if await _is_valid_rss(href, client):
                        return href

                # Look for <a href="...rss..."> links
                for match in re.finditer(
                    r'href=["\']([^"\']*(?:rss|feed|atom)[^"\']*)["\']',
                    r.text, re.IGNORECASE
                ):
                    href = match.group(1)
                    if not href.startswith("http"):
                        href = urljoin(base, href)
                    if await _is_valid_rss(href, client):
                        return href
        except Exception:
            pass

    return None


# ── Test 3: Parse RSS feed ────────────────────────────────────────

async def test_parse_rss(ticker: str, rss_url: str, max_items: int = 10) -> list[dict]:
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        try:
            r = await client.get(rss_url, headers=_HEADERS)
            r.raise_for_status()
        except Exception as e:
            print(f"  ✗ Feed fetch failed: {e}")
            return []

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        print(f"  ✗ XML parse error: {e}")
        return []

    is_atom = root.tag.endswith("feed") or "atom" in root.tag.lower()
    if is_atom:
        items = root.findall("atom:entry", _NS) or root.findall("entry")
    else:
        ch = root.find("channel")
        items = ch.findall("item") if ch is not None else root.findall("item")

    results = []
    for item in items[:max_items * 3]:  # check more than max to find relevant ones
        def _t(tag):
            el = item.find(tag) or item.find(f"atom:{tag}", _NS)
            return (el.text or "").strip() if el is not None else ""

        title = _t("title")
        desc  = _t("description") or _t("summary")
        link  = _t("link")
        if not link:
            le = item.find("atom:link", _NS) or item.find("link")
            if le is not None:
                link = le.get("href", "") or (le.text or "")

        pub   = _parse_date(_t("pubDate") or _t("published") or _t("updated"))

        results.append({
            "title":        title,
            "link":         link,
            "published_at": pub,
            "period_label": _infer_period(title),
            "relevant":     _is_relevant(title, desc),
        })

    return results


# ── Test 4: SEC EDGAR ─────────────────────────────────────────────

EDGAR_CIK_MAP = {
    "LKQX":   "0001065696",
    "PSI CN": "0001104485",
    "FFH CN": "0000915191",
}

async def test_edgar(ticker: str, cik: Optional[str] = None) -> list[dict]:
    cik = cik or EDGAR_CIK_MAP.get(ticker)
    if not cik:
        print(f"  ✗ No CIK known for {ticker}. Pass --cik XXXXXXXXXX")
        return []

    cik_padded = cik.lstrip("0").zfill(10)
    url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.get(url, headers={
                "User-Agent": "Oldfield Partners research@oldfieldpartners.com",
                "Accept": "application/json",
            })
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ✗ EDGAR request failed: {e}")
            return []

    filings = data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    dates   = filings.get("filingDate", [])
    accns   = filings.get("accessionNumber", [])
    periods = filings.get("periodOfReport", [])

    target_forms = {"10-K", "10-Q", "20-F", "40-F", "6-K", "8-K"}
    results = []
    for i, form in enumerate(forms):
        if form not in target_forms:
            continue
        results.append({
            "form":        form,
            "date":        dates[i] if i < len(dates) else "",
            "period":      periods[i] if i < len(periods) else "",
            "accession":   accns[i] if i < len(accns) else "",
        })
        if len(results) >= 10:
            break

    return results


# ─────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg):print(f"  {RED}✗{RESET} {msg}")
def info(msg):print(f"  {CYAN}→{RESET} {msg}")
def head(msg):print(f"\n{BOLD}{msg}{RESET}")


# ─────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────

async def run_tests(args):
    print(f"\n{'='*60}")
    print(f"  Harvester Test Suite")
    print(f"{'='*60}")

    # ── Companies to test ──────────────────────────────────────
    if args.company and args.ticker:
        companies = [{"name": args.company, "ticker": args.ticker, "country": args.country}]
    else:
        # Default test set — covers European IR, North American IR, EDGAR
        companies = [
            {"name": "Bunzl",          "ticker": "BNZL LN",  "country": "UK"},
            {"name": "Heineken",       "ticker": "HEIA",     "country": "Netherlands"},
            {"name": "ENI",            "ticker": "ENI IM",   "country": "Italy"},
            {"name": "Pason Systems",  "ticker": "PSI CN",   "country": "Canada"},
            {"name": "LKQ Corporation","ticker": "LKQX",     "country": "USA"},
        ]

    for co in companies:
        name    = co["name"]
        ticker  = co["ticker"]
        country = co.get("country")

        print(f"\n{'─'*60}")
        print(f"  {BOLD}{name} ({ticker}){RESET}")
        print(f"{'─'*60}")

        rss_url = args.rss_url  # override if passed via CLI

        # ── Step 1: LLM discovery ──────────────────────────────
        if not rss_url:
            head("Step 1: LLM IR URL Discovery")
            ir_url = await test_llm_discovery(name, ticker, country)
            if ir_url:
                ok(f"IR URL: {ir_url}")
            else:
                warn("LLM returned no URL (check API key or try --rss-url directly)")
                continue

            # ── Step 2: RSS probe ──────────────────────────────
            head("Step 2: RSS Feed Probe")
            info(f"Probing {_extract_base(ir_url)} for RSS feeds…")
            rss_url = await test_rss_probe(ir_url)
            if rss_url:
                ok(f"RSS found: {rss_url}")
            else:
                warn("No RSS feed found at this domain")
                info("Tip: try passing --rss-url manually if you know the feed URL")
                continue
        else:
            head("RSS Feed (manual URL)")
            ok(f"Using: {rss_url}")

        # ── Step 3: Parse feed ─────────────────────────────────
        head("Step 3: Feed Parse + Relevance Filter")
        items = await test_parse_rss(ticker, rss_url)
        if not items:
            fail("No items parsed from feed")
            continue

        relevant   = [i for i in items if i["relevant"]]
        irrelevant = [i for i in items if not i["relevant"]]

        ok(f"Total items parsed: {len(items)}")
        ok(f"Relevant (financial results): {len(relevant)}")
        info(f"Filtered out (non-results): {len(irrelevant)}")

        if relevant:
            print(f"\n  {BOLD}Relevant items:{RESET}")
            for item in relevant[:5]:
                period = item["period_label"] or "period unknown"
                pub    = item["published_at"].strftime("%Y-%m-%d") if item["published_at"] else "date unknown"
                print(f"    • [{pub}] [{period:>10}]  {item['title'][:70]}")
                print(f"      {CYAN}{item['link'][:80]}{RESET}")
        else:
            warn("No relevant items found — check keyword list or feed content")

        if irrelevant:
            print(f"\n  {BOLD}Sample filtered-out items:{RESET}")
            for item in irrelevant[:3]:
                print(f"    · {item['title'][:70]}")

        args.rss_url = None  # reset for next company if set globally

    # ── EDGAR test ─────────────────────────────────────────────
    if args.edgar or (not args.company):
        edgar_tickers = [args.ticker] if args.ticker else ["LKQX", "PSI CN"]
        for ticker in edgar_tickers:
            print(f"\n{'─'*60}")
            head(f"SEC EDGAR: {ticker}")
            cik = args.cik
            filings = await test_edgar(ticker, cik)
            if filings:
                ok(f"Found {len(filings)} recent filings:")
                for f in filings:
                    print(f"    • {f['form']:6s}  {f['date']}  period: {f['period']}")
                    print(f"      accession: {f['accession']}")
            else:
                fail("No filings found")

    print(f"\n{'='*60}")
    print(f"  Test complete")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test harvester components standalone")
    parser.add_argument("--company",  help="Company name, e.g. 'Bunzl'")
    parser.add_argument("--ticker",   help="Ticker, e.g. 'BNZL LN'")
    parser.add_argument("--country",  help="Country hint for LLM, e.g. 'UK'")
    parser.add_argument("--rss-url",  help="Skip discovery and test a known RSS URL directly")
    parser.add_argument("--edgar",    action="store_true", help="Run EDGAR test only")
    parser.add_argument("--cik",      help="SEC CIK for EDGAR test, e.g. '0001065696'")
    args = parser.parse_args()

    try:
        import anthropic
    except ImportError:
        print(f"{RED}Missing dependency. Run:{RESET}  pip install httpx anthropic")
        sys.exit(1)

    asyncio.run(run_tests(args))


if __name__ == "__main__":
    main()
