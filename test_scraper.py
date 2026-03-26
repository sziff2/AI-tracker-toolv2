"""
Standalone test for IR website scraper.
Tests scraping a company IR page directly for PDF links.

Usage:
    python test_scraper.py
    python test_scraper.py --url "https://www.bunzl.com/investors/results-reports-and-presentations/" --ticker "BNZL LN"
    python test_scraper.py --url "https://www.theheinekencompany.com/investors/results-reports-webcasts-and-presentations" --ticker "HEIA"
    python test_scraper.py --url "https://www.eni.com/en-IT/investors/reporting-financial-statements.html" --ticker "ENI IM"
"""

import argparse
import asyncio
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

# ── Inline scraper (no app imports) ──────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

_RESULTS_PAGE_PATTERNS = [
    r"annual.?report", r"half.?year", r"interim", r"full.?year",
    r"results", r"preliminary", r"trading.?update",
    r"q[1-4].{0,10}20\d\d", r"20\d\d.{0,10}(result|report|interim|annual)",
    r"investor.?present", r"capital.?market", r"presentation",
]

_RESULTS_PDF_PATTERNS = [
    r"annual.?report", r"results", r"interim", r"half.?year",
    r"full.?year", r"preliminary", r"trading.?update",
    r"press.?release", r"earnings", r"presentation", r"slides", r"appendix",
]

_PERIOD_PATTERNS = [
    (r"full[- ]?year\s+(\d{4})",           lambda m: f"{m.group(1)}_FY"),
    (r"(\d{4})\s+full[- ]?year",           lambda m: f"{m.group(1)}_FY"),
    (r"annual.report.(\d{4})",             lambda m: f"{m.group(1)}_FY"),
    (r"(\d{4}).annual",                    lambda m: f"{m.group(1)}_FY"),
    (r"half[- ]?year\s+(\d{4})",           lambda m: f"{m.group(1)}_H1"),
    (r"(\d{4})\s+half[- ]?year",           lambda m: f"{m.group(1)}_H1"),
    (r"interim.{0,20}(\d{4})",             lambda m: f"{m.group(1)}_H1"),
    (r"H1\s+(\d{4})",                      lambda m: f"{m.group(1)}_H1"),
    (r"Q([1-4])\s+(\d{4})",               lambda m: f"{m.group(2)}_Q{m.group(1)}"),
    (r"(\d{4})\s+Q([1-4])",               lambda m: f"{m.group(1)}_Q{m.group(2)}"),
    (r"(\d{4})",                           lambda m: f"{m.group(1)}_FY"),
]

GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"

def ok(m):   print(f"  {GREEN}✓{RESET} {m}")
def warn(m): print(f"  {YELLOW}⚠{RESET} {m}")
def fail(m): print(f"  {RED}✗{RESET} {m}")
def info(m): print(f"  {CYAN}→{RESET} {m}")
def head(m): print(f"\n{BOLD}{m}{RESET}")


def _base(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"

def _infer_period(text):
    for pattern, fmt in _PERIOD_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result = fmt(m)
            yr = re.search(r"(\d{4})", result)
            if yr and 2010 <= int(yr.group(1)) <= 2030:
                return result
    return None

def _extract_links(html, base_url):
    links = []
    for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                         html, re.IGNORECASE | re.DOTALL):
        href = m.group(1).strip()
        text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        if not href or href.startswith('#') or href.startswith('mailto:'):
            continue
        if not href.startswith('http'):
            href = urljoin(base_url, href)
        links.append((href, text))
    return links

def _is_results_link(href, text):
    combined = (href + " " + text).lower()
    return any(re.search(p, combined, re.IGNORECASE) for p in _RESULTS_PAGE_PATTERNS)

def _is_results_pdf(href, text):
    combined = (href + " " + text).lower()
    return any(re.search(p, combined, re.IGNORECASE) for p in _RESULTS_PDF_PATTERNS)


async def run_scraper_test(ticker: str, ir_docs_url: str, max_subpages: int = 8):
    print(f"\n{'='*60}")
    print(f"  IR Scraper Test — {ticker}")
    print(f"  {ir_docs_url}")
    print(f"{'='*60}")

    ir_domain = urlparse(ir_docs_url).netloc
    visited = set()
    all_pdfs = []

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=_HEADERS) as client:

        # Step 1: fetch main page
        head("Step 1: Fetch main IR page")
        try:
            resp = await client.get(ir_docs_url)
            resp.raise_for_status()
            ok(f"HTTP {resp.status_code} — {len(resp.text):,} chars")
        except Exception as e:
            fail(f"Failed: {e}")
            return

        visited.add(ir_docs_url)
        main_links = _extract_links(resp.text, ir_docs_url)
        ok(f"Found {len(main_links)} total links on page")

        # Direct PDFs on main page
        direct_pdfs = [(h, t) for h, t in main_links if h.lower().endswith('.pdf')]
        if direct_pdfs:
            ok(f"{len(direct_pdfs)} direct PDF links on main page")

        # Sub-pages
        subpages = [
            (h, t) for h, t in main_links
            if not h.lower().endswith('.pdf')
            and urlparse(h).netloc == ir_domain
            and h not in visited
            and _is_results_link(h, t)
        ]
        ok(f"{len(subpages)} results sub-pages found to crawl")

        if subpages:
            print(f"\n  {BOLD}Sub-pages identified:{RESET}")
            for h, t in subpages[:15]:
                print(f"    · {t[:50]:50s}  {CYAN}{h[:70]}{RESET}")

        # Step 2: crawl sub-pages
        head(f"Step 2: Crawl sub-pages (max {max_subpages})")
        for href, text in subpages[:max_subpages]:
            visited.add(href)
            try:
                sr = await client.get(href)
                sr.raise_for_status()
                sub_links = _extract_links(sr.text, href)
                pdfs = [(h, t) for h, t in sub_links if h.lower().endswith('.pdf')]
                results_pdfs = [(h, t) for h, t in pdfs if _is_results_pdf(h, t)]
                info(f"{text[:45]:45s} → {len(pdfs)} PDFs, {len(results_pdfs)} relevant")
                all_pdfs.extend(results_pdfs)
            except Exception as e:
                warn(f"  Sub-page failed {href[:60]}: {e}")

        # Also include direct PDFs from main page
        all_pdfs.extend([(h, t) for h, t in direct_pdfs if _is_results_pdf(h, t)])

        # Deduplicate
        seen_urls = set()
        unique_pdfs = []
        for h, t in all_pdfs:
            if h not in seen_urls:
                seen_urls.add(h)
                unique_pdfs.append((h, t))

        # Step 3: show results
        head("Step 3: Results")
        if not unique_pdfs:
            fail("No relevant PDFs found")
            warn("The site may use JavaScript rendering — static scraping won't work")
            return

        ok(f"Found {len(unique_pdfs)} unique relevant PDFs:")
        print()
        for pdf_url, link_text in unique_pdfs:
            slug = urlparse(pdf_url).path
            context = f"{link_text} {slug}"
            period = _infer_period(context) or "period unknown"
            print(f"    {GREEN}•{RESET} [{period:>12}]  {link_text[:55]:55s}")
            print(f"              {CYAN}{pdf_url[:90]}{RESET}")

    print(f"\n{'='*60}")
    print(f"  Done — {len(unique_pdfs)} PDFs found")
    print(f"{'='*60}\n")


DEFAULT_COMPANIES = [
    {
        "ticker": "BNZL LN",
        "url": "https://www.bunzl.com/investors/results-reports-and-presentations/",
    },
    {
        "ticker": "HEIA",
        "url": "https://www.theheinekencompany.com/investors/results-reports-webcasts-and-presentations",
    },
    {
        "ticker": "ENI IM",
        "url": "https://www.eni.com/en-IT/investors/reporting-financial-statements.html",
    },
]


async def main(args):
    if args.url and args.ticker:
        await run_scraper_test(args.ticker, args.url)
    else:
        for co in DEFAULT_COMPANIES:
            await run_scraper_test(co["ticker"], co["url"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Ticker e.g. 'BNZL LN'")
    parser.add_argument("--url",    help="IR documents page URL")
    args = parser.parse_args()

    try:
        import httpx
    except ImportError:
        print("Run: pip install httpx")
        exit(1)

    asyncio.run(main(args))
