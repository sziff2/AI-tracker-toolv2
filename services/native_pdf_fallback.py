"""
Native Claude PDF fallback — Tier 1.3 production path.

When the baseline parser (pymupdf + pdfplumber) returns zero tables on
a filing that should have them — Canadian condensed consolidated
statements, UK RNS half-years, any table-heavy non-SEC document —
send the PDF directly to Claude as a document content block and
recover the tables that way.

Why a fallback, not a wholesale swap:
  - Baseline is confirmed working on US SEC 10-Q/10-K: 60 tables /
    14.6s on LKQ Corp Q3 2025. Swapping that path to native PDF would
    cost ~$0.20 per call for zero quality gain.
  - Sprint C-prep A/B showed baseline HANGS for 507s on NWC CN Q3
    FY2025 before returning 0 tables. Native PDF reads the same file
    in 31s and recovers 28 tables with real numbers ($83,031 cash,
    $112,390 AR, etc.). So the trigger is narrow: "baseline returned
    nothing on a type that should have tables".

Cost control:
  - 4096 max_tokens + retry-on-empty-parse (2048 truncated the JSON
    mid-value in the A/B — silent failure).
  - 100-page doc-block cap per Anthropic constraint — over-cap docs
    stay on the text-only baseline path.
  - Concurrency bounded by the same global semaphore as every other
    Anthropic call.

Ship behind `settings.native_pdf_fallback = False` default. Flip to
True after shadow-running for 2 weeks with both paths logged and only
baseline output used. See Tier 1.3 in Dev plans/_consolidated_roadmap.md.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from configs.settings import settings

logger = logging.getLogger(__name__)


# Doc types where a 0-table baseline result is a strong signal of
# baseline failure rather than a genuinely table-free document. US
# 10-Q/10-K stay in this set — if baseline ever returns 0 tables on
# one of those we want the fallback to fire. Transcripts and
# presentations stay OUT: 0 tables on a transcript is expected.
DEFAULT_FALLBACK_DOC_TYPES = {
    "financial_statements",
    "condensed_financials",
    "earnings_release",
    "10-Q",
    "10-K",
    "annual_report",
    "rns",
    "interim_report",
    "half_year_report",
}


# Prompt asks Claude to emit each table as a markdown block with a
# stable delimiter so we can parse them back into the 2D-list shape
# the existing baseline path produces. We also ask for a per-table
# page number and an optional caption — the existing downstream
# (services/financial_statement_segmenter.py) keys on the caption.
_FALLBACK_PROMPT = """\
You are extracting financial tables from a PDF so a downstream
pipeline can route them to statement-specific metric extractors.

For EACH table visible in the document, emit one object in a JSON
array. Do NOT summarise, do NOT skip small tables. Preserve every
row and every column exactly as shown, including units in headers,
footnote references, and subtotals.

Return STRICT JSON. No preamble. No markdown fences:
{{
  "tables": [
    {{
      "page": <int, 1-indexed>,
      "caption": "<the heading/title above the table, e.g. 'Condensed Consolidated Balance Sheets'>",
      "rows": [
        ["<cell 1>", "<cell 2>", "..."],
        ["<cell 1>", "<cell 2>", "..."]
      ]
    }}
  ]
}}

Rules:
- Every row is an ARRAY of strings (not objects), one cell per column.
- Keep empty cells as empty strings "".
- Do NOT merge or reshape rows; output them in document order.
- If a cell contains a number with thousand separators or currency
  symbols, keep them verbatim ("$1,234.56", not "1234.56").
- Include header rows and totals rows as regular rows.
- If the document contains NO tables, return {{"tables": []}}.
"""


async def extract_tables_from_pdf(
    file_path: str,
    *,
    max_pages: int | None = None,
    max_tokens: int = 4096,
    model: str | None = None,
) -> dict[str, Any]:
    """Extract tables from a PDF via Anthropic document content block.

    Returns the shape
        {
          "tables": [{"page": int, "caption": str, "rows": list[list[str]]}, ...],
          "pdf_bytes":       int,
          "page_count":      int,
          "input_tokens":    int,
          "output_tokens":   int,
          "cost_usd":        float,
          "status":          "ok" | "skip" | "error",
          "reason":          str,   # non-empty on skip/error
          "retried":         bool,  # True if the second attempt was used
        }

    Never raises for operational failures — callers get a structured
    status so the outer extraction pipeline can degrade gracefully.
    """
    path = Path(file_path)
    if not path.exists():
        return {"tables": [], "status": "error", "reason": f"file not found: {path.name}"}

    # Probe page count via fitz (already a hard dep of the pipeline)
    try:
        import fitz  # type: ignore
        doc = fitz.open(str(path))
        page_count = len(doc)
        doc.close()
    except Exception as exc:  # noqa: BLE001
        return {"tables": [], "status": "error", "reason": f"fitz page-count probe failed: {exc}"}

    max_pages = max_pages if max_pages is not None else settings.native_pdf_max_pages
    if page_count > max_pages:
        return {
            "tables": [], "page_count": page_count, "status": "skip",
            "reason": f"{page_count} pages exceeds {max_pages}-page fallback cap",
        }

    if not settings.anthropic_api_key:
        return {"tables": [], "page_count": page_count, "status": "error",
                "reason": "ANTHROPIC_API_KEY not set"}

    pdf_bytes = path.read_bytes()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")

    # Route through the existing global concurrency semaphore so this
    # doesn't bypass the rate limiter everyone else observes.
    from services.llm_client import _get_semaphore  # local import — avoids circular at module load
    sem = _get_semaphore()

    # Use AsyncAnthropic directly — this is a doc-block call, not plain
    # text, so the standard llm_client path doesn't cover it.
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = model or settings.native_pdf_model or settings.agent_default_model

    async def _one_call(tokens: int) -> tuple[dict[str, Any] | None, int, int, str]:
        async with sem:
            resp = await client.messages.create(
                model=model,
                max_tokens=tokens,
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
                        {"type": "text", "text": _FALLBACK_PROMPT},
                    ],
                }],
            )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        parsed = _safe_parse_json(raw)
        return parsed, resp.usage.input_tokens, resp.usage.output_tokens, raw

    total_in = 0
    total_out = 0
    retried = False
    try:
        parsed, in_toks, out_toks, raw_text = await _one_call(max_tokens)
    except Exception as exc:  # noqa: BLE001
        return {
            "tables": [], "page_count": page_count, "status": "error",
            "reason": f"anthropic call failed: {str(exc)[:200]}",
        }
    total_in += in_toks
    total_out += out_toks

    tables = (parsed or {}).get("tables") or []

    # Retry once on empty-parse or JSON failure — max_tokens=2048 in
    # the Sprint C-prep A/B silently truncated NWC's response
    # mid-value. If that happens we bump tokens and try again once.
    if not isinstance(tables, list) or (not tables and (parsed is None or "tables" not in (parsed or {}))):
        retried = True
        try:
            parsed2, in2, out2, raw2 = await _one_call(max_tokens + 4096)
            total_in += in2
            total_out += out2
            tables = (parsed2 or {}).get("tables") or []
            raw_text = raw2
        except Exception as exc:  # noqa: BLE001
            logger.warning("native_pdf_fallback retry failed: %s", str(exc)[:200])

    # Normalise: every row must be a list[str]
    clean_tables: list[dict] = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        rows = t.get("rows")
        if not isinstance(rows, list):
            continue
        clean_rows = []
        for row in rows:
            if not isinstance(row, list):
                continue
            clean_rows.append([_cell_str(c) for c in row])
        if clean_rows:
            clean_tables.append({
                "page":    int(t.get("page", 0) or 0),
                "caption": str(t.get("caption", "") or ""),
                "rows":    clean_rows,
            })

    # Per-model pricing — Sonnet default, known via settings.
    cost = (total_in * 3.0 + total_out * 15.0) / 1_000_000

    status = "ok" if clean_tables else "error"
    reason = ""
    if not clean_tables:
        reason = "empty tables after parse + retry"

    logger.info(
        "native_pdf_fallback: %s | pages=%d tables=%d tokens_in=%d tokens_out=%d cost=$%.4f retried=%s",
        path.name, page_count, len(clean_tables), total_in, total_out, cost, retried,
    )

    return {
        "tables":         clean_tables,
        "pdf_bytes":      len(pdf_bytes),
        "page_count":     page_count,
        "input_tokens":   total_in,
        "output_tokens":  total_out,
        "cost_usd":       round(cost, 4),
        "status":         status,
        "reason":         reason,
        "retried":        retried,
    }


def _safe_parse_json(raw: str) -> dict | None:
    """Parse Claude response as JSON, tolerant of ```json fences and
    occasional preamble prose. Returns None on unrecoverable failure."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        # Strip fenced blocks — Claude occasionally opens with ```json
        lines = text.split("\n")
        if len(lines) > 2:
            text = "\n".join(lines[1:-1])
        else:
            text = text[3:]
    # Find the first { and last } — tolerates prose before/after the JSON
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("native_pdf_fallback JSON parse failed: %s", str(exc)[:150])
        return None


_CELL_WHITESPACE = re.compile(r"\s+")


def _cell_str(c: Any) -> str:
    """Coerce a table cell into a compact string. Claude occasionally
    emits a number for a number-looking cell; we want everything as str
    so the downstream statement segmenter has a consistent input."""
    if c is None:
        return ""
    s = str(c)
    return _CELL_WHITESPACE.sub(" ", s).strip()


def tables_to_baseline_shape(
    fallback_tables: list[dict],
) -> list[dict]:
    """Convert `[{page, caption, rows}, ...]` into the shape returned
    by `extract_tables_pdfplumber`: `[{page, tables: [[row, row, ...]]}, ...]`.

    pdfplumber groups multiple tables per page under one dict with
    `page` and `tables` = list of 2D arrays. We fold the caption into
    the rows by prepending it as a single-cell header — downstream
    segmenter matches on caption text anyway.
    """
    by_page: dict[int, list[list[list[str]]]] = {}
    for t in fallback_tables:
        page = int(t.get("page", 0) or 0)
        rows = list(t.get("rows", []))
        caption = t.get("caption", "")
        if caption:
            rows = [[caption]] + rows
        by_page.setdefault(page, []).append(rows)

    return [{"page": p, "tables": tbls} for p, tbls in sorted(by_page.items())]
