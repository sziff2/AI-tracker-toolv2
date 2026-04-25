"""
Sprint C-prep — Tier 5.6 three-way PDF parser A/B.

Arms:
  1. baseline          — existing pymupdf + pdfplumber
  2. opendataloader    — Java-core local parser (opendataloader-pdf)
  3. native_claude_pdf — Anthropic document block

Deliverable: a markdown report at Dev plans/_sprint-c-prep-ab-results.md
(not in the code repo — lives in the plans folder).

Usage:
  # from repo root
  python -m scripts.pdf_ab_test --phase baseline
  python -m scripts.pdf_ab_test --phase opendataloader
  python -m scripts.pdf_ab_test --phase native --max-pdfs 3
  python -m scripts.pdf_ab_test --phase report  # compile final markdown

Run phases individually so a cost-incurring arm (native) requires an
explicit invocation. Results are persisted as JSON under
storage/ab_test/<phase>.json so each phase's output survives between
runs and the final report compiles what exists.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pdf_ab")

# Where to find test PDFs
BLP_DIR = Path("C:/blp/data")

# Where to persist results
AB_STORE = Path(__file__).resolve().parent.parent / "storage" / "ab_test"
AB_STORE.mkdir(parents=True, exist_ok=True)

# Test filings — selected for variety: US 10-Q, Canadian stmts (failure case),
# Japanese tanshin, UK transcript, LKQ deck for narrative read, LKQ transcript.
TEST_FILINGS: list[dict[str, str]] = [
    {
        "key":  "NWC_CN_Q3_FY2025",
        "label": "Canadian condensed financial statements (NWC CN Q3 FY2025) — confirmed pdfplumber failure",
        "file": "20251210_North_West_Co_Inc-The-_Financial_Statements_2025-12-10_SE000000003088536099.pdf",
        "doc_type": "financial_statements",
    },
    {
        "key":  "LKQ_10Q_Q3_2025",
        "label": "US 10-Q (LKQ Corp Q3 2025)",
        "file": "20251030_LKQ_Corp-_10_Q_2025-10-30.pdf",
        "doc_type": "10-Q",
    },
    {
        "key":  "LKQ_transcript_Q3_2025",
        "label": "Earnings call transcript (LKQ Corp Q3 2025)",
        "file": "20251030_LKQ_Corp-_Earnings_Call_2025-10-30_DN000000003082660721.pdf.pdf",
        "doc_type": "transcript",
    },
    {
        "key":  "LKQ_deck_Q3_2025",
        "label": "Investor presentation deck (LKQ Corp Q3 2025)",
        "file": "20251030_LKQ_Corp-_Company_Presentation_2025-10-30_WC000000003082631839.pdf",
        "doc_type": "presentation",
    },
    {
        "key":  "TSE_3679_2026Q1",
        "label": "Japanese TSE tanshin (3679 JP)",
        "file": "TSE_ 3679@JP_020226_63033.pdf",
        "doc_type": "earnings_release",
    },
    {
        "key":  "TESCO_call_2025",
        "label": "UK earnings call (Tesco H1 2025)",
        "file": "Tesco PLC Earnings Call 20251002 SD000000003078912516.pdf.pdf",
        "doc_type": "transcript",
    },
]


# ─────────────────────────────────────────────────────────────────
# Result schema — identical across arms for easy diffing
# ─────────────────────────────────────────────────────────────────

@dataclass
class ArmResult:
    arm:          str
    key:          str
    status:       str             # ok | skip | error
    page_count:   int = 0
    char_count:   int = 0
    table_count:  int = 0
    headings_count: int = 0
    first_1k_chars: str = ""
    tables_sample: list = None
    elapsed_s:    float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error_msg:    str = ""

    def __post_init__(self):
        if self.tables_sample is None:
            self.tables_sample = []


def _save_results(arm: str, results: list[ArmResult]) -> Path:
    path = AB_STORE / f"{arm}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    return path


def _load_results(arm: str) -> list[dict]:
    path = AB_STORE / f"{arm}.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pdf_path(filing: dict) -> Path:
    return BLP_DIR / filing["file"]


def _pdf_exists(filing: dict) -> bool:
    return _pdf_path(filing).exists()


# ─────────────────────────────────────────────────────────────────
# Arm 1: baseline (pymupdf + pdfplumber)
# ─────────────────────────────────────────────────────────────────

def run_baseline_one(filing: dict) -> ArmResult:
    path = _pdf_path(filing)
    if not path.exists():
        return ArmResult(arm="baseline", key=filing["key"], status="skip", error_msg=f"file not found: {path.name}")

    from services.document_parser import extract_text_pymupdf, extract_tables_pdfplumber
    import re

    t0 = time.time()
    try:
        pages = extract_text_pymupdf(str(path))
        tables = extract_tables_pdfplumber(str(path))
    except Exception as exc:  # noqa: BLE001
        return ArmResult(
            arm="baseline", key=filing["key"], status="error",
            error_msg=str(exc)[:200], elapsed_s=time.time() - t0,
        )

    full_text = "\n".join(p["text"] for p in pages)
    # Naive heading count — ALL-CAPS lines or title-case short lines
    heading_pattern = re.compile(r"^[A-Z][A-Z0-9\s,\-]{3,80}$", re.MULTILINE)
    headings = heading_pattern.findall(full_text)
    # Show the first table as a sample (first 3 rows × 4 cols) for readability
    tables_sample = []
    for t in tables[:2]:
        if t.get("tables"):
            for tbl in t["tables"][:1]:
                tables_sample.append({
                    "page": t["page"],
                    "rows": [row[:4] for row in tbl[:3]],
                })

    return ArmResult(
        arm="baseline", key=filing["key"], status="ok",
        page_count=len(pages),
        char_count=len(full_text),
        table_count=sum(len(t.get("tables", [])) for t in tables),
        headings_count=len(headings),
        first_1k_chars=full_text[:1000],
        tables_sample=tables_sample,
        elapsed_s=time.time() - t0,
    )


def run_baseline_phase() -> list[ArmResult]:
    out = []
    for filing in TEST_FILINGS:
        logger.info("[baseline] %s", filing["key"])
        r = run_baseline_one(filing)
        out.append(r)
        logger.info("  → %s | pages=%d tables=%d chars=%d elapsed=%.2fs",
                    r.status, r.page_count, r.table_count, r.char_count, r.elapsed_s)
    _save_results("baseline", out)
    return out


# ─────────────────────────────────────────────────────────────────
# Arm 2: opendataloader-pdf
# ─────────────────────────────────────────────────────────────────

def _opendataloader_available() -> tuple[bool, str]:
    """Probe: Python pkg installed + JVM reachable."""
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError:
        return False, "pip package 'opendataloader-pdf' not installed"
    import subprocess
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return False, "java returned non-zero"
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"JRE/JDK not on PATH ({type(exc).__name__}) — opendataloader needs Java 11+"
    return True, "ok"


def run_opendataloader_one(filing: dict) -> ArmResult:
    path = _pdf_path(filing)
    if not path.exists():
        return ArmResult(arm="opendataloader", key=filing["key"], status="skip", error_msg=f"file not found: {path.name}")

    import tempfile
    import re
    t0 = time.time()
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            from opendataloader_pdf import convert
            convert(
                input_path=str(path),
                output_dir=out_dir,
                format="markdown,json",
                quiet=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ArmResult(
                arm="opendataloader", key=filing["key"], status="error",
                error_msg=str(exc)[:200], elapsed_s=time.time() - t0,
            )

        # Locate the emitted markdown + json. Both land in out_dir with the
        # PDF basename and .md / .json suffix.
        stem = Path(path).stem
        md_path = Path(out_dir) / f"{stem}.md"
        json_path = Path(out_dir) / f"{stem}.json"
        md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        meta = {}
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    # Best-effort table count from markdown — rows that are | foo | bar |
    table_rows = sum(1 for line in md.splitlines() if line.startswith("|") and line.count("|") >= 3)
    # Each table is roughly a cluster of contiguous rows — approximate
    table_count = 0
    in_table = False
    for line in md.splitlines():
        is_row = line.startswith("|") and line.count("|") >= 3
        if is_row and not in_table:
            table_count += 1
            in_table = True
        elif not is_row and in_table:
            in_table = False

    # Headings from markdown: lines starting with one-or-more '#' followed by space
    heading_lines = [ln for ln in md.splitlines() if re.match(r"^#+\s", ln)]

    return ArmResult(
        arm="opendataloader", key=filing["key"], status="ok",
        page_count=int(meta.get("page_count", 0) or 0),
        char_count=len(md),
        table_count=table_count,
        headings_count=len(heading_lines),
        first_1k_chars=md[:1000],
        tables_sample=[{"table_rows_total": table_rows}],
        elapsed_s=time.time() - t0,
    )


def run_opendataloader_phase() -> list[ArmResult]:
    available, msg = _opendataloader_available()
    if not available:
        logger.warning("[opendataloader] unavailable: %s — recording skip results", msg)
        out = [
            ArmResult(arm="opendataloader", key=f["key"], status="skip",
                      error_msg=msg)
            for f in TEST_FILINGS
        ]
        _save_results("opendataloader", out)
        return out

    out = []
    for filing in TEST_FILINGS:
        logger.info("[opendataloader] %s", filing["key"])
        r = run_opendataloader_one(filing)
        out.append(r)
        logger.info("  → %s | pages=%d tables=%d chars=%d",
                    r.status, r.page_count, r.table_count, r.char_count)
    _save_results("opendataloader", out)
    return out


# ─────────────────────────────────────────────────────────────────
# Arm 3: native Claude PDF
# ─────────────────────────────────────────────────────────────────

# Deliberately small strategic set — cost control.
NATIVE_STRATEGIC_KEYS = [
    "NWC_CN_Q3_FY2025",
    "LKQ_deck_Q3_2025",
    "LKQ_transcript_Q3_2025",
]

# Prompt asks Claude to summarise the structural parts of the PDF — this
# mirrors what a narrative deep-read agent would need (headings, tables,
# chart callouts, slide flow). It deliberately asks Claude to report what
# it can see that the text-only baseline would miss.
NATIVE_PROMPT = """\
You are benchmarking PDF comprehension for an investment-research parser
A/B. Treat this PDF as an arbitrary financial document.

Return STRICT JSON, no preamble:
{
  "doc_pages": <int>,
  "heading_count": <int>,
  "main_headings": ["...", "..."],       // up to 10 top-level section names
  "table_count": <int>,                   // major tables visible in the PDF
  "key_table_summary": "<1-2 sentences on the biggest table(s)>",
  "visible_charts": <int>,
  "chart_descriptions": ["...", "..."],   // up to 5 short descriptions
  "layout_signals": ["...", "..."],       // things the text-only baseline would miss (column breaks, chart callouts, deck-style flow, waterfall visuals)
  "likely_document_type": "<one of: 10-Q, 10-K, condensed_financials, earnings_release, transcript, presentation, tanshin, RNS, other>",
  "sample_first_1000_chars": "<verbatim from page 1>"
}
"""


def _estimate_pdf_tokens(page_count: int) -> int:
    """Anthropic doc-block pricing: ~2,250 tokens per PDF page (midpoint of
    1,500-3,000 quoted in our Tier 2.3 scope). Conservative estimate for
    budget tracking."""
    return page_count * 2250


def run_native_one(filing: dict, dry_run: bool = False) -> ArmResult:
    path = _pdf_path(filing)
    if not path.exists():
        return ArmResult(arm="native_claude_pdf", key=filing["key"], status="skip",
                         error_msg=f"file not found: {path.name}")

    # Quick page count probe from baseline so cost estimate is real
    try:
        import fitz
        doc = fitz.open(str(path))
        page_count = len(doc)
        doc.close()
    except Exception as exc:  # noqa: BLE001
        return ArmResult(arm="native_claude_pdf", key=filing["key"], status="error",
                         error_msg=f"fitz probe failed: {exc}")

    if page_count > 100:
        return ArmResult(arm="native_claude_pdf", key=filing["key"], status="skip",
                         page_count=page_count,
                         error_msg=f"{page_count} pages exceeds 100-page doc-block cap")

    est_tokens = _estimate_pdf_tokens(page_count)
    est_cost   = est_tokens * 3.0 / 1_000_000  # Sonnet $3/M input
    logger.info("  est_tokens=%d  est_cost=$%.3f  page_count=%d",
                est_tokens, est_cost, page_count)

    if dry_run:
        return ArmResult(arm="native_claude_pdf", key=filing["key"], status="skip",
                         page_count=page_count,
                         input_tokens=est_tokens,
                         estimated_cost_usd=est_cost,
                         error_msg="dry_run")

    # Load + base64 encode
    with path.open("rb") as f:
        pdf_b64 = base64.standard_b64encode(f.read()).decode("ascii")

    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    t0 = time.time()
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",  # current Sonnet (4.5/4 retire June 15 2026)
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {"type": "text", "text": NATIVE_PROMPT},
                ],
            }],
        )
    except Exception as exc:  # noqa: BLE001
        return ArmResult(
            arm="native_claude_pdf", key=filing["key"], status="error",
            page_count=page_count, error_msg=str(exc)[:300],
            elapsed_s=time.time() - t0,
        )

    elapsed = time.time() - t0
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    input_tokens = resp.usage.input_tokens
    output_tokens = resp.usage.output_tokens
    cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    # Parse the JSON
    parsed: dict[str, Any] = {}
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(stripped)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  JSON parse failed: %s", exc)

    return ArmResult(
        arm="native_claude_pdf", key=filing["key"], status="ok",
        page_count=int(parsed.get("doc_pages") or page_count),
        char_count=len(parsed.get("sample_first_1000_chars", "")),
        table_count=int(parsed.get("table_count", 0)),
        headings_count=int(parsed.get("heading_count", 0)),
        first_1k_chars=parsed.get("sample_first_1000_chars", "")[:1000],
        tables_sample=[{
            "main_headings":       parsed.get("main_headings", []),
            "key_table_summary":   parsed.get("key_table_summary", ""),
            "chart_descriptions":  parsed.get("chart_descriptions", []),
            "visible_charts":      parsed.get("visible_charts", 0),
            "layout_signals":      parsed.get("layout_signals", []),
            "likely_document_type": parsed.get("likely_document_type", ""),
            "raw_response_len":    len(text),
        }],
        elapsed_s=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
    )


def run_native_phase(max_pdfs: int = 3, dry_run: bool = False) -> list[ArmResult]:
    out = []
    targets = [f for f in TEST_FILINGS if f["key"] in NATIVE_STRATEGIC_KEYS][:max_pdfs]
    total_est = 0.0
    for filing in targets:
        logger.info("[native%s] %s", "-dry" if dry_run else "", filing["key"])
        r = run_native_one(filing, dry_run=dry_run)
        total_est += r.estimated_cost_usd
        out.append(r)
    logger.info("[native] total estimated cost: $%.3f", total_est)
    _merge_and_save_native(out)
    return out


def _merge_and_save_native(new_results: list[ArmResult]) -> None:
    existing = {e.get("key"): e for e in _load_results("native_claude_pdf")}
    merged = list(existing.values())
    for r in new_results:
        merged = [m for m in merged if m.get("key") != r.key]
        merged.append(asdict(r))
    path = AB_STORE / "native_claude_pdf.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, default=str)


def run_native_phase_remote(
    base_url: str, password: str, max_pdfs: int = 3,
) -> list[ArmResult]:
    """Post the strategic PDFs to the deployed /admin/pdf-ab/native
    endpoint on Railway. The ANTHROPIC_API_KEY never leaves production.

    Authenticates via the same /auth/login flow used by the browser
    (password body → session cookie). Returns ArmResult list in the
    same shape as the local path so the report compiler doesn't care.
    """
    import httpx

    targets = [f for f in TEST_FILINGS if f["key"] in NATIVE_STRATEGIC_KEYS][:max_pdfs]
    payload_pdfs = []
    for filing in targets:
        path = _pdf_path(filing)
        if not path.exists():
            logger.warning("skip %s: file missing", filing["key"])
            continue
        with path.open("rb") as f:
            pdf_b64 = base64.standard_b64encode(f.read()).decode("ascii")
        payload_pdfs.append({
            "key":        filing["key"],
            "filename":   filing["file"],
            "pdf_base64": pdf_b64,
        })
    if not payload_pdfs:
        logger.error("no PDFs to send")
        return []

    logger.info("[native-remote] sending %d PDFs to %s", len(payload_pdfs), base_url)
    with httpx.Client(base_url=base_url, timeout=300.0, follow_redirects=False) as client:
        login = client.post("/auth/login", json={"password": password})
        if login.status_code != 200:
            raise RuntimeError(f"login failed: {login.status_code} {login.text[:200]}")

        resp = client.post(
            "/api/v1/admin/pdf-ab/native",
            json={"pdfs": payload_pdfs},
            cookies=login.cookies,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"extraction failed: {resp.status_code} {resp.text[:300]}")
        body = resp.json()

    logger.info("[native-remote] total cost: $%.4f  input=%d output=%d",
                body.get("total_cost_usd", 0),
                body.get("total_input_tokens", 0),
                body.get("total_output_tokens", 0))

    out: list[ArmResult] = []
    for row in body.get("results", []):
        if row.get("status") != "ok":
            out.append(ArmResult(
                arm="native_claude_pdf", key=row.get("key", ""), status="error",
                error_msg=row.get("error", "")[:300],
                elapsed_s=row.get("elapsed_s", 0),
            ))
            continue
        parsed = row.get("parsed", {}) or {}
        out.append(ArmResult(
            arm="native_claude_pdf",
            key=row.get("key", ""),
            status="ok",
            page_count=int(parsed.get("doc_pages") or 0),
            char_count=len(parsed.get("sample_first_1000_chars", "")),
            table_count=int(parsed.get("table_count", 0) or 0),
            headings_count=int(parsed.get("heading_count", 0) or 0),
            first_1k_chars=parsed.get("sample_first_1000_chars", "")[:1000],
            tables_sample=[{
                "main_headings":      parsed.get("main_headings", []),
                "key_table_summary":  parsed.get("key_table_summary", ""),
                "chart_descriptions": parsed.get("chart_descriptions", []),
                "visible_charts":     parsed.get("visible_charts", 0),
                "layout_signals":     parsed.get("layout_signals", []),
                "likely_document_type": parsed.get("likely_document_type", ""),
                "raw_response_len":   row.get("raw_response_len", 0),
            }],
            elapsed_s=row.get("elapsed_s", 0),
            input_tokens=row.get("input_tokens", 0),
            output_tokens=row.get("output_tokens", 0),
            estimated_cost_usd=row.get("cost_usd", 0),
        ))
    _merge_and_save_native(out)
    return out


# ─────────────────────────────────────────────────────────────────
# Report compilation
# ─────────────────────────────────────────────────────────────────

REPORT_PATH = Path("C:/Users/sam/OneDrive - Oldfield Partners/Reading/AI agent/Dev plans/_sprint-c-prep-ab-results.md")


def _fmt_n(n: int | None) -> str:
    return "—" if n is None else str(n)


def _lookup(results: list[dict], key: str) -> dict | None:
    for r in results:
        if r.get("key") == key:
            return r
    return None


def compile_report() -> Path:
    baseline = _load_results("baseline")
    openloader = _load_results("opendataloader")
    native = _load_results("native_claude_pdf")

    lines: list[str] = []
    lines.append("# Sprint C-prep — Tier 5.6 PDF Parser A/B Results")
    lines.append("")
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} by scripts/pdf_ab_test.py")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("Decide two questions ahead of Sprint C (Tier 1.3) and Sprint D2 (Tier 2.3):")
    lines.append("1. Does any parser arm materially beat the baseline on condensed "
                 "financial statements (the NWC CN Q3 FY2025 failure case)? Answer gates "
                 "Tier 1.3's implementation path.")
    lines.append("2. Does native Claude PDF materially beat the current truncated-text read on "
                 "narrative deep-reads (deck + transcript)? Answer gates Tier 2.3 go/no-go.")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("| Arm | What it is | Cost | Env dependencies |")
    lines.append("|---|---|---|---|")
    lines.append("| **baseline** | Current pipeline: `pymupdf` for text + `pdfplumber` for tables + regex section_splitter | free | none new |")
    lines.append("| **opendataloader** | Java-core local parser (opendataloader-pdf) — native heading hierarchy, bounding boxes, Markdown out | free per-call | JRE install on Railway image |")
    lines.append("| **native_claude_pdf** | Anthropic document content block — PDF sent verbatim to Sonnet | ~$0.10-0.60 per PDF per agent call; prompt caching brings 2-5 follow-up calls to 10% | none (API only) |")
    lines.append("")

    # Summary table
    lines.append("## Per-PDF headline comparison")
    lines.append("")
    lines.append("Columns are the 3 arms; rows are PDFs. Cells show `pages / tables / headings / chars`.")
    lines.append("")
    lines.append("| PDF | Baseline | opendataloader | native_claude_pdf |")
    lines.append("|---|---|---|---|")
    for filing in TEST_FILINGS:
        b  = _lookup(baseline, filing["key"])
        ol = _lookup(openloader, filing["key"])
        n  = _lookup(native, filing["key"])
        def _cell(r):
            if not r or r.get("status") == "skip":
                return f"_skip_ ({(r or {}).get('error_msg', 'n/a')[:60]})"
            if r.get("status") == "error":
                return f"**error** ({r.get('error_msg','')[:60]})"
            return (f"{_fmt_n(r.get('page_count'))}p / "
                    f"{_fmt_n(r.get('table_count'))}t / "
                    f"{_fmt_n(r.get('headings_count'))}h / "
                    f"{_fmt_n(r.get('char_count'))}c")
        lines.append(f"| **{filing['key']}** ({filing['doc_type']}) | {_cell(b)} | {_cell(ol)} | {_cell(n)} |")
    lines.append("")

    # Detailed per-PDF section
    lines.append("## Per-PDF detail")
    lines.append("")
    for filing in TEST_FILINGS:
        lines.append(f"### {filing['key']}")
        lines.append(f"_{filing['label']}_")
        lines.append("")
        for arm_name, arm_results in (
            ("baseline", baseline),
            ("opendataloader", openloader),
            ("native_claude_pdf", native),
        ):
            r = _lookup(arm_results, filing["key"])
            if not r:
                lines.append(f"- **{arm_name}**: (not run)")
                continue
            if r.get("status") != "ok":
                lines.append(f"- **{arm_name}**: {r.get('status')} — {r.get('error_msg','')[:200]}")
                continue
            lines.append(f"- **{arm_name}**: {r.get('page_count')}p, {r.get('table_count')} tables, "
                         f"{r.get('headings_count')} headings, {r.get('char_count')} chars, "
                         f"{r.get('elapsed_s',0):.2f}s"
                         + (f", ${r.get('estimated_cost_usd',0):.3f}"
                            if r.get('estimated_cost_usd') else ""))
            # Show tables sample / native structured response
            ts = r.get("tables_sample") or []
            if ts and arm_name == "native_claude_pdf":
                s = ts[0]
                lines.append(f"    - detected doc_type: `{s.get('likely_document_type','?')}`")
                if s.get("main_headings"):
                    lines.append(f"    - main headings: {', '.join(s['main_headings'][:6])}")
                if s.get("key_table_summary"):
                    lines.append(f"    - key tables: {s['key_table_summary']}")
                if s.get("visible_charts"):
                    lines.append(f"    - {s['visible_charts']} visible charts")
                if s.get("layout_signals"):
                    lines.append(f"    - layout signals baseline would miss: {', '.join(s['layout_signals'][:4])}")
        lines.append("")

    # Decision criteria
    lines.append("## Decision rules")
    lines.append("")
    lines.append("Apply per doc-type, not per PDF:")
    lines.append("")
    lines.append("- **Condensed financial statements** (NWC_CN_Q3_FY2025) — whichever arm recovers "
                 "≥40 absolute metrics (the exit criterion from Tier 1.3) wins. Baseline's `tables=0` "
                 "here is the documented failure. If opendataloader or native-PDF recovers the tables, "
                 "it takes the Tier 1.3 integration path.")
    lines.append("- **US 10-Q** (LKQ_10Q_Q3_2025) — baseline is known-good for SEC filings. If neither "
                 "alternative is a clean win here, keep baseline on US 10-K/10-Q and only swap where a "
                 "specific doc type demands it.")
    lines.append("- **Decks** (LKQ_deck_Q3_2025) — native PDF should show visible charts / layout signals "
                 "baseline misses. If it does AND delivers richer heading structure, Tier 2.3 greenlit.")
    lines.append("- **Transcripts** (LKQ_transcript_Q3_2025, TESCO_call_2025) — all three arms should be "
                 "roughly equivalent (transcripts are flat text). Native PDF without improvement here is "
                 "wasted spend; baseline stays.")
    lines.append("- **Tanshin** (TSE_3679_2026Q1) — edge case for baseline. If either alternative reads the "
                 "Japanese columnar format cleanly, that's a point for it on APAC filings.")
    lines.append("")
    lines.append("## Findings & recommendation")
    lines.append("")
    lines.append("_To complete once all arms have run._")
    lines.append("")
    lines.append("### Tier 1.3 (extraction) implementation path")
    lines.append("_Fill in: baseline + pdfplumber-fallback, or opendataloader, or native PDF._")
    lines.append("")
    lines.append("### Tier 2.3 (narrative deep-reads) go/no-go")
    lines.append("_Fill in: ship native PDF for decks+transcripts, or drop 2.3._")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report written: %s", REPORT_PATH)
    return REPORT_PATH


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "opendataloader", "native", "report", "all"],
                        default="report")
    parser.add_argument("--max-pdfs", type=int, default=3, help="For native arm only")
    parser.add_argument("--dry-run", action="store_true", help="Native arm: estimate cost but do not call API")
    parser.add_argument("--mode", choices=["local", "remote"], default="local",
                        help="Native arm: local uses ANTHROPIC_API_KEY; remote POSTs to deployed /admin/pdf-ab/native")
    parser.add_argument("--base-url", default="https://ai-tracker-tool-production.up.railway.app",
                        help="Remote mode: Railway URL")
    parser.add_argument("--password", default=os.environ.get("APP_PASSWORD", ""),
                        help="Remote mode: APP_PASSWORD (env APP_PASSWORD if not passed)")
    args = parser.parse_args()

    # Fix console encoding on Windows for non-ASCII printout
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.phase in ("baseline", "all"):
        run_baseline_phase()
    if args.phase in ("opendataloader", "all"):
        run_opendataloader_phase()
    if args.phase in ("native", "all"):
        if args.mode == "remote":
            if not args.password:
                logger.error("remote mode needs --password or env APP_PASSWORD")
                sys.exit(2)
            run_native_phase_remote(
                base_url=args.base_url,
                password=args.password,
                max_pdfs=args.max_pdfs,
            )
        else:
            run_native_phase(max_pdfs=args.max_pdfs, dry_run=args.dry_run)
    if args.phase in ("report", "all"):
        compile_report()


if __name__ == "__main__":
    main()
