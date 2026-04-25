"""
Native-Claude extraction — single-call alternative to the section-aware +
two-pass + statement-extractor pipeline.

Motivated by the ARW US 2025_Q4 10-K producing zero metrics: SEC inline-XBRL
HTML breaks the regex-based section splitter + the <table> regex extractor,
leaving the LLM with 32K chars of cover-page boilerplate. Claude's document
API reads layout, inline-XBRL, and mixed div/table structure natively.

Design:
  - One Claude call per document
  - Structured JSON output with the same raw_items shape the legacy path
    emits, so the result drops into existing persistence + context-builder
    code without changes.
  - Input routing:
      (a) PDFs → native document block (Tier 2.3 pattern)
      (b) HTML / text-already-extracted → plain-text prompt, cap at 150K
          chars (Sonnet context is 200K; leaves room for the schema).
  - Failure is non-fatal. Returns an empty-items dict on any exception so
    the caller can fall back to legacy extraction.

Scope boundary:
  This replaces metric extraction. It does NOT replace narrative deep-reads
  (transcript_deep_dive, presentation_analysis, annual_report_deep_read) —
  those have their own prompts and run at ingestion. The new path produces
  structured metric rows; the existing narrative prompts produce separate
  ResearchOutput rows and are unaffected.

Feature flag: settings.use_native_extraction (default False).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document
from configs.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# The one prompt
# ─────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are extracting structured financial data from a {doc_type} for {ticker}.

Sector context: {sector}
Industry:       {industry}
Period:         {period_label}

Return a SINGLE valid JSON object matching the schema below. No preamble,
no markdown fences, no prose. Extract every metric you can find; prefer
precision over recall when the document is ambiguous.

SCHEMA:
{{
  "metrics": [
    {{
      "metric_name":    "<canonical snake_case name, e.g. 'revenue', 'operating_income', 'combined_ratio'>",
      "metric_value":   <number, null if non-numeric>,
      "metric_text":    "<verbatim figure from the doc, e.g. '$1,234.5 million' or '18.5%'>",
      "unit":           "<USD_M | USD_B | EUR_M | % | bps | x | count | null>",
      "segment":        "<segment/business unit name OR 'consolidated' OR null>",
      "period":         "<YYYY_QN or YYYY_FY — leave blank to inherit the document's period>",
      "source_snippet": "<≤200 chars of surrounding context so a reader can verify this line>",
      "confidence":     <0.0-1.0>,
      "is_one_off":     <true if explicitly called out as non-recurring, else false>
    }}
  ],

  "segments": [
    {{
      "name":     "<segment name>",
      "revenue":  <number or null>,
      "margin":   <number or null>,
      "growth":   <number or null>,
      "notes":    "<one-line comment on drivers, margins, mix>"
    }}
  ],

  "mda_narrative": "<≤15,000 chars of the MD&A / business-review text verbatim, preserving paragraph breaks. If the doc has no MD&A, return an empty string.>",

  "guidance": [
    {{
      "metric":      "<name>",
      "value_or_range": "<e.g. '5-7%', 'mid-single-digit', '$2.1-2.3bn'>",
      "specificity": "<point | range | directional | qualitative>"
    }}
  ],

  "confidence_profile": {{
    "overall_signal": "<confident | mixed | hedging>",
    "hedge_rate":     <0.0-1.0 — fraction of forward-looking statements using hedge words (may, could, expect, believe, anticipate)>,
    "one_off_rate":   <0.0-1.0 — fraction of metrics called out as non-recurring>
  }},

  "non_gaap_bridges": [
    {{
      "from_metric": "<e.g. 'reported_net_income'>",
      "to_metric":   "<e.g. 'adjusted_net_income'>",
      "items":       [{{"description": "<e.g. restructuring'>", "amount": <number>}}]
    }}
  ],

  "detected_period": "<YYYY_QN or YYYY_FY — your best read of what period this doc primarily covers>"
}}

RULES:
1. Use snake_case canonical names. Pick the common one for this sector.
2. Never invent numbers. If the doc says "grew low-single-digits" leave metric_value null and put "low-single-digits" in metric_text.
3. Prefer absolute values over ratios when both appear (both may still be extracted as separate rows).
4. For segment metrics, set the segment field. For consolidated metrics set "consolidated".
5. source_snippet is required for every metric — the reader must be able to verify.
6. Mark is_one_off=true only when the doc explicitly says "non-recurring", "one-time", "restructuring", etc.
7. If the document is short or pre-release commentary only (no real data), return empty arrays — do NOT fabricate rows.

DOCUMENT TEXT:
{document_text}
"""


# ─────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────

async def run_native_extraction(
    db: AsyncSession,
    document: Document,
    full_text: str,
    *,
    pdf_path: Optional[str] = None,
    sector: str = "",
    industry: str = "",
    country: str = "",
) -> dict:
    """Single-call extraction. Returns the same dict shape as
    services.metric_extractor._extract_with_sections so it can drop into
    the existing pipeline.

    On any failure returns an empty result so the caller can fall back to
    legacy extraction without the pipeline going sideways.
    """
    doc_type = (document.document_type or "other").lower()
    period_label = document.period_label or ""

    # Which ticker is this? Best-effort lookup for the prompt.
    ticker = ""
    try:
        from sqlalchemy import select as sa_select
        from apps.api.models import Company
        cq = await db.execute(sa_select(Company).where(Company.id == document.company_id))
        company = cq.scalar_one_or_none()
        if company:
            ticker = company.ticker or ""
            sector = sector or (company.sector or "")
            industry = industry or (company.industry or "")
            country = country or (company.country or "")
    except Exception:
        pass

    # Decide input mode:
    # - PDF on disk + feature supports it → native document block (best fidelity)
    # - Otherwise → truncated text, large cap (150K chars) vs the legacy 32K
    use_native_pdf = False
    if pdf_path and Path(pdf_path).is_file() and pdf_path.lower().endswith(".pdf"):
        try:
            import fitz  # type: ignore
            _f = fitz.open(pdf_path)
            _pages = len(_f)
            _f.close()
            # Anthropic caps document blocks at 100 pages. Over that, fall
            # back to text so we don't send a truncated doc silently.
            if _pages <= 100:
                use_native_pdf = True
            else:
                logger.info(
                    "native_extraction: doc %s has %d pages > 100 — falling back to text input",
                    document.id, _pages,
                )
        except Exception as exc:
            logger.warning(
                "native_extraction: page-count probe failed for %s: %s",
                document.id, str(exc)[:120],
            )

    # Text-mode cap. SEC 10-Ks after XBRL strip are ~800K-1.5M chars of
    # mostly-useful content (boilerplate is <20% of the tail). 400K chars
    # ≈ 100K tokens — fits comfortably inside Sonnet 4.6's 200K window,
    # leaves ~100K for the prompt + output, and captures the full
    # financial-statements + MD&A + footnotes block that a 150K cap was
    # cutting off. Smaller filings (8-Ks, press releases) just pass
    # through unchanged since they're 10-50K.
    MAX_TEXT_CHARS = 400_000
    text_input = full_text or ""
    if len(text_input) > MAX_TEXT_CHARS:
        logger.info(
            "native_extraction: truncating text input for doc %s: %d → %d chars",
            document.id, len(text_input), MAX_TEXT_CHARS,
        )
        text_input = text_input[:MAX_TEXT_CHARS]

    # Large-doc shortcut: chunk + Haiku.
    # Observed on ARW US 2025_Q4 10-K: a single Sonnet call on the full
    # 240KB iXBRL+text input ran 18 minutes and timed out — the doc is
    # too dense for one call. Splitting into N parallel Haiku calls
    # (each ~50-80KB) finishes in 30-90s, with negligible recall loss
    # because each chunk asks for ALL metrics it can see. Sonnet stays
    # the default for normal-size docs (8-Ks, 10-Qs, presentations,
    # transcripts).
    LARGE_DOC_CHARS = 100_000
    is_large_doc = len(text_input) > LARGE_DOC_CHARS

    # When we have BOTH a usable PDF AND the doc is large, prefer the
    # chunked-PDF-Haiku route — gives the 10-K layout-aware reading
    # (tables/columns) that text-chunking loses, while still using
    # Haiku's speed advantage in parallel.
    if use_native_pdf and is_large_doc and pdf_path:
        logger.info(
            "native_extraction: large doc with PDF (%d chars) — using chunked PDF-Haiku path",
            len(text_input),
        )
        return await _run_chunked_pdf_haiku(
            pdf_path,
            doc_type=doc_type,
            ticker=ticker,
            sector=sector or "generic",
            industry=industry or "",
            period_label=period_label,
        )

    if not use_native_pdf and is_large_doc:
        logger.info(
            "native_extraction: large doc (%d chars) — using chunked Haiku path",
            len(text_input),
        )
        return await _run_chunked_haiku(
            text_input,
            doc_type=doc_type,
            ticker=ticker,
            sector=sector or "generic",
            industry=industry or "",
            period_label=period_label,
        )

    prompt = _EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        ticker=ticker,
        sector=sector or "generic",
        industry=industry or "",
        period_label=period_label,
        document_text=(text_input if not use_native_pdf else "(the PDF is attached above as a document block)"),
    )

    # Make the call. Keep max_tokens high for long 10-Ks (can emit hundreds
    # of metric rows). Sonnet is the right tier — extraction accuracy matters.
    # Bump timeout to 10 minutes (Anthropic's hard non-streaming ceiling).
    # Observed: ARW US 2025_Q4 10-K (240KB iXBRL → 100-page dense PDF) needs
    # >300s for Sonnet to fully process and emit 16K-token extraction output.
    # Beyond 600s we'd need to switch to streaming — separate scope.
    try:
        from services.llm_client import call_llm_native_async
        result = await call_llm_native_async(
            prompt,
            model=settings.agent_default_model,  # Sonnet
            max_tokens=16_384,
            timeout_seconds=600,
            feature="native_extraction",
            ticker=ticker,
            period=period_label,
            pdf_path=pdf_path if use_native_pdf else None,
        )
    except Exception as exc:
        logger.warning("native_extraction LLM call failed for doc %s: %s", document.id, str(exc)[:200])
        return _empty_result(doc_type, reason=f"llm_call_failed: {str(exc)[:120]}")

    raw_text = (result.get("text") or "").strip()
    parsed = _parse_json_safe(raw_text)
    if parsed is None:
        logger.warning(
            "native_extraction JSON parse failed for doc %s (len=%d). First 200: %r",
            document.id, len(raw_text), raw_text[:200],
        )
        return _empty_result(doc_type, reason="json_parse_failed")

    return _map_to_legacy_shape(parsed, doc_type, use_native_pdf=use_native_pdf)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# Second-pass: targeted IS / BS / CF extraction
# ─────────────────────────────────────────────────────────────────

_STATEMENTS_PROMPT = """You are extracting the three core financial statements from a {doc_type} for {ticker}.

Period: {period_label}

Look at the attached PDF. If it contains the consolidated income statement, consolidated balance sheet, or consolidated cash flow statement, extract EVERY line item with EVERY period column shown.

If the PDF chunk contains none of these statements, return an empty array for each.

Rules:
- Use the EXACT line-item name as printed (e.g. "Sales", "Cost of products sold", "Net income attributable to shareholders")
- Return ALL period columns (typically 2-3 years for a 10-K, 2-3 quarters for a 10-Q)
- Period format: YYYY_FY for annual, YYYY_QN for quarterly (e.g. "2025_FY", "2024_FY", "2023_FY")
- Numeric values only — strip $ signs and "thousand"/"million" suffixes; report the unit separately
- Unit: USD_M for millions, USD_B for billions, USD_K for thousands, %, x, count
- Skip subtotals and EPS lines if absent from the face of the statement
- Skip the notes / footnotes — those are separate

Return STRICT JSON (no preamble, no markdown fences):

{{
  "income_statement": [
    {{"line_item": "Sales", "period": "2025_FY", "value": 30852.9, "unit": "USD_M"}},
    {{"line_item": "Sales", "period": "2024_FY", "value": 27911.7, "unit": "USD_M"}},
    ...
  ],
  "balance_sheet": [
    {{"line_item": "Cash and cash equivalents", "period": "2025_FY", "value": 306.5, "unit": "USD_M"}},
    ...
  ],
  "cash_flow": [
    {{"line_item": "Cash from operating activities", "period": "2025_FY", "value": 1234.5, "unit": "USD_M"}},
    ...
  ]
}}
"""


# Doc types where targeted IS/BS/CF extraction is worth running.
# Press releases / 8-Ks / presentations either lack the full statements
# or have them as too-summary data (better caught by the main pass).
_STATEMENTS_DOC_TYPES = frozenset({
    "annual_report",
    "10-k", "10k",
    "10-q", "10q",
    "earnings_release",
    "financial_statements",
})


async def _run_targeted_statements_pass(
    pdf_chunks: list,
    *,
    doc_type: str,
    ticker: str,
    sector: str,
    industry: str,
    period_label: str,
) -> Optional[dict]:
    """Second-pass extraction: targeted at IS/BS/CF line items × all
    period columns. Runs Sonnet on each PDF chunk with a strict
    statements-only prompt. Returns the parsed result in the same
    legacy shape as the main pass so it can be merged.

    Each chunk is independent — chunks without the statements just
    return empty arrays. ~$0.30-0.60 per 10-K; bought back many times
    over by the missing comparator data.
    """
    import asyncio
    from services.llm_client import call_llm_native_async

    async def _extract_one(idx: int, chunk_path) -> Optional[dict]:
        prompt = _STATEMENTS_PROMPT.format(
            doc_type=doc_type,
            ticker=ticker,
            period_label=period_label,
        )
        try:
            r = await call_llm_native_async(
                prompt,
                model=settings.agent_default_model,  # Sonnet — accuracy matters here
                max_tokens=8192,
                timeout_seconds=300,
                feature="native_extraction_statements_pass",
                ticker=ticker,
                period=period_label,
                pdf_path=str(chunk_path),
            )
            return _parse_json_safe((r.get("text") or "").strip())
        except Exception as exc:
            logger.warning("Statements pass: chunk %d failed: %s", idx + 1, str(exc)[:200])
            return None

    parsed_chunks = await asyncio.gather(
        *[_extract_one(i, p) for i, p in enumerate(pdf_chunks)],
        return_exceptions=False,
    )

    # Flatten IS/BS/CF nested structure into a single metrics array
    # in the legacy shape so it merges with chunked-Haiku results.
    metrics: list[dict] = []
    seen: set[tuple] = set()
    n_lines_by_stmt = {"income_statement": 0, "balance_sheet": 0, "cash_flow": 0}

    for chunk_result in parsed_chunks:
        if not chunk_result:
            continue
        for stmt_key in ("income_statement", "balance_sheet", "cash_flow"):
            for line in (chunk_result.get(stmt_key) or []):
                label = (line.get("line_item") or "").strip()
                period = (line.get("period") or "").strip()
                if not label:
                    continue
                # Snake-case the label for downstream code that keys on metric_name
                metric_name = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
                if not metric_name:
                    continue
                key = (stmt_key, metric_name, period)
                if key in seen:
                    continue
                seen.add(key)
                value = line.get("value")
                # Infer frequency from the period the LLM emitted. 10-K
                # statements pass produces FY by default; quarterly docs
                # would emit Q1/Q2/Q3/Q4. Required so dedup keeps FY
                # full-year values separate from same-quarter quarterly
                # numbers (Q4 2025 sales $8B ≠ FY 2025 sales $31B).
                period_upper = period.upper()
                if "FY" in period_upper:
                    freq = "FY"
                elif "H1" in period_upper:
                    freq = "H1"
                elif "H2" in period_upper:
                    freq = "H2"
                else:
                    freq = "Q"
                metrics.append({
                    "metric_name":      metric_name,
                    "metric_value":     value,
                    "metric_text":      str(value) if value is not None else label,
                    "unit":             line.get("unit") or "USD_M",
                    "segment":          "consolidated",
                    "period":           period,
                    "period_frequency": freq,
                    "source_snippet":   f"{stmt_key}: {label}",
                    "confidence":       0.95,
                    "is_one_off":       False,
                })
                n_lines_by_stmt[stmt_key] += 1

    if not metrics:
        return None

    logger.info(
        "Statements pass: extracted IS=%d, BS=%d, CF=%d line-items (total %d)",
        n_lines_by_stmt["income_statement"],
        n_lines_by_stmt["balance_sheet"],
        n_lines_by_stmt["cash_flow"],
        len(metrics),
    )
    return {
        "metrics":             metrics,
        "segments":            [],
        "mda_narrative":       "",
        "guidance":            [],
        "non_gaap_bridges":    [],
        "confidence_profile":  {},
        "detected_period":     "",
    }


async def _run_chunked_pdf_haiku(
    pdf_path: str,
    *,
    doc_type: str,
    ticker: str,
    sector: str,
    industry: str,
    period_label: str,
    n_chunks: int = 4,
) -> dict:
    """Split a large PDF into N page-range chunks, send each as a native
    PDF document block to Haiku in parallel, merge results.

    Trade-off: gives the 10-K layout-aware reading (Claude sees columns,
    tables, charts) that text-chunking loses, while keeping Haiku's
    speed advantage. Each ~20-25 page chunk fits well under Anthropic's
    100-page-per-doc cap. ~30-90s wall-clock for 4 parallel calls.
    """
    import asyncio
    from pathlib import Path
    from services.llm_client import call_llm_native_async
    from services.doc_fetch import split_pdf_into_chunks

    try:
        chunk_paths = await asyncio.to_thread(
            split_pdf_into_chunks, Path(pdf_path), n_chunks
        )
    except Exception as exc:
        logger.warning("Chunked-PDF Haiku: split failed for %s: %s", pdf_path, str(exc)[:200])
        return _empty_result(doc_type, reason=f"pdf_split_failed: {str(exc)[:120]}")

    if not chunk_paths:
        return _empty_result(doc_type, reason="empty_pdf")

    async def _extract_one(idx: int, chunk_path: Path) -> Optional[dict]:
        prompt = _EXTRACTION_PROMPT.format(
            doc_type=doc_type,
            ticker=ticker,
            sector=sector,
            industry=industry,
            period_label=period_label,
            document_text=(
                f"(PDF chunk {idx + 1}/{len(chunk_paths)} attached above as a document block)"
            ),
        )

        # First pass: Haiku (fast, cheap).
        try:
            r = await call_llm_native_async(
                prompt,
                model=settings.agent_fast_model,  # Haiku
                max_tokens=8192,
                timeout_seconds=300,
                feature="native_extraction_pdf_chunk",
                ticker=ticker,
                period=period_label,
                pdf_path=str(chunk_path),
            )
            parsed = _parse_json_safe((r.get("text") or "").strip())
            if parsed is not None:
                return parsed
            logger.warning(
                "Chunked-PDF Haiku: chunk %d JSON parse failed — retrying with Sonnet",
                idx + 1,
            )
        except Exception as exc:
            logger.warning(
                "Chunked-PDF Haiku: chunk %d Haiku call failed (%s) — retrying with Sonnet",
                idx + 1, str(exc)[:200],
            )

        # Second pass per chunk: Sonnet (slower, more reliable JSON +
        # better extraction depth). Each chunk is ~25 pages so Sonnet
        # handles it comfortably in <2 min — no timeout risk like the
        # full-doc Sonnet attempts that ran 18 minutes earlier today.
        try:
            r2 = await call_llm_native_async(
                prompt,
                model=settings.agent_default_model,  # Sonnet
                max_tokens=12_288,
                timeout_seconds=300,
                feature="native_extraction_pdf_chunk_sonnet",
                ticker=ticker,
                period=period_label,
                pdf_path=str(chunk_path),
            )
            parsed = _parse_json_safe((r2.get("text") or "").strip())
            if parsed is None:
                logger.warning(
                    "Chunked-PDF Sonnet: chunk %d also failed JSON parse",
                    idx + 1,
                )
            else:
                logger.info(
                    "Chunked-PDF: chunk %d recovered via Sonnet (%d metrics)",
                    idx + 1, len(parsed.get("metrics") or []),
                )
            return parsed
        except Exception as exc:
            logger.warning(
                "Chunked-PDF Sonnet: chunk %d also failed: %s",
                idx + 1, str(exc)[:200],
            )
            return None

    # Run main pass + targeted statements pass concurrently.
    # Targeted pass only fires on doc types where IS/BS/CF is expected.
    main_task = asyncio.gather(
        *[_extract_one(i, p) for i, p in enumerate(chunk_paths)],
        return_exceptions=False,
    )
    if doc_type.lower() in _STATEMENTS_DOC_TYPES:
        statements_task = _run_targeted_statements_pass(
            chunk_paths,
            doc_type=doc_type,
            ticker=ticker,
            sector=sector,
            industry=industry,
            period_label=period_label,
        )
        main_results, statements_result = await asyncio.gather(main_task, statements_task)
    else:
        main_results = await main_task
        statements_result = None

    parsed_list = [p for p in main_results if p is not None]
    if statements_result is not None:
        parsed_list.append(statements_result)

    if not parsed_list:
        return _empty_result(doc_type, reason="all_pdf_chunks_failed")

    merged = _merge_chunk_parsed(parsed_list)
    main_chunk_count = sum(1 for p in main_results if p is not None)
    logger.info(
        "Chunked-PDF Haiku: merged %d/%d main chunks + %s statements pass → "
        "%d metrics, %d segments, %d guidance",
        main_chunk_count, len(chunk_paths),
        "1" if statements_result is not None else "0",
        len(merged.get("metrics") or []),
        len(merged.get("segments") or []), len(merged.get("guidance") or []),
    )
    result = _map_to_legacy_shape(merged, doc_type, use_native_pdf=True)
    suffix = "+statements" if statements_result is not None else ""
    result["extraction_method"] = f"native_claude_v1_chunked_pdf_haiku{suffix}"
    return result


async def _run_chunked_haiku(
    text_input: str,
    *,
    doc_type: str,
    ticker: str,
    sector: str,
    industry: str,
    period_label: str,
    n_chunks: int = 4,
) -> dict:
    """Split a large doc into N chunks, run Haiku extraction on each in
    parallel, merge results. Used for SEC 10-Ks where a single Sonnet
    call times out at 600s+.

    Trade-off: Haiku is ~3-5× faster than Sonnet but somewhat less
    precise. For the ~3% of docs (10-Ks) that don't fit in a single
    Sonnet call, that's the right trade. Each chunk asks for ALL
    metrics it can see, then dedup on (metric_name, period) on merge.
    """
    import asyncio
    from services.llm_client import call_llm_native_async

    chunk_size = (len(text_input) + n_chunks - 1) // n_chunks
    chunks = [text_input[i:i + chunk_size] for i in range(0, len(text_input), chunk_size)]
    if not chunks:
        return _empty_result(doc_type, reason="empty_text_input")

    async def _extract_one(idx: int, chunk_text: str) -> Optional[dict]:
        prompt = _EXTRACTION_PROMPT.format(
            doc_type=doc_type,
            ticker=ticker,
            sector=sector,
            industry=industry,
            period_label=period_label,
            document_text=f"[CHUNK {idx + 1}/{len(chunks)}]\n\n{chunk_text}",
        )
        try:
            r = await call_llm_native_async(
                prompt,
                model=settings.agent_fast_model,  # Haiku — fast tier
                max_tokens=8192,
                timeout_seconds=300,
                feature="native_extraction_chunk",
                ticker=ticker,
                period=period_label,
            )
            parsed = _parse_json_safe((r.get("text") or "").strip())
            if parsed is None:
                logger.warning("Chunked Haiku: chunk %d JSON parse failed", idx + 1)
            return parsed
        except Exception as exc:
            logger.warning("Chunked Haiku: chunk %d failed: %s", idx + 1, str(exc)[:200])
            return None

    parsed_list = await asyncio.gather(
        *[_extract_one(i, c) for i, c in enumerate(chunks)],
        return_exceptions=False,
    )
    parsed_list = [p for p in parsed_list if p is not None]

    if not parsed_list:
        return _empty_result(doc_type, reason="all_chunks_failed")

    merged = _merge_chunk_parsed(parsed_list)
    logger.info(
        "Chunked Haiku: merged %d chunks → %d metrics, %d segments, %d guidance",
        len(parsed_list), len(merged.get("metrics") or []),
        len(merged.get("segments") or []), len(merged.get("guidance") or []),
    )
    # Tag method so the A/B + downstream can see this was the chunked path.
    result = _map_to_legacy_shape(merged, doc_type, use_native_pdf=False)
    result["extraction_method"] = "native_claude_v1_chunked_haiku"
    return result


def _merge_chunk_parsed(results: list[dict]) -> dict:
    """Merge per-chunk parsed JSONs. Dedup metrics on (name, period);
    union segments by name; concat guidance/bridges; take longest MD&A."""
    seen_metrics: set[tuple[str, str]] = set()
    seen_segments: set[str] = set()
    merged: dict = {
        "metrics": [],
        "segments": [],
        "mda_narrative": "",
        "guidance": [],
        "non_gaap_bridges": [],
        "confidence_profile": {},
        "detected_period": "",
        "disappearance_flags": {},
    }
    for r in results:
        for m in (r.get("metrics") or []):
            name = (m.get("metric_name") or "").strip()
            if not name:
                continue
            key = (name, m.get("period") or "")
            if key in seen_metrics:
                continue
            seen_metrics.add(key)
            merged["metrics"].append(m)
        for s in (r.get("segments") or []):
            name = (s.get("name") or "").strip()
            if not name or name in seen_segments:
                continue
            seen_segments.add(name)
            merged["segments"].append(s)
        if isinstance(r.get("mda_narrative"), str) and \
           len(r["mda_narrative"]) > len(merged["mda_narrative"]):
            merged["mda_narrative"] = r["mda_narrative"]
        for g in (r.get("guidance") or []):
            merged["guidance"].append(g)
        for b in (r.get("non_gaap_bridges") or []):
            merged["non_gaap_bridges"].append(b)
        if not merged["confidence_profile"] and r.get("confidence_profile"):
            merged["confidence_profile"] = r["confidence_profile"]
        if not merged["detected_period"] and r.get("detected_period"):
            merged["detected_period"] = r["detected_period"]
    return merged


def _parse_json_safe(raw: str) -> Optional[dict]:
    """Best-effort JSON extraction. Returns None only when no parseable
    object can be found at all. Handles:
      - Markdown fence wrappers (```json ... ```)
      - Preamble / postamble prose ("Here is the JSON: { ... }")
      - Trailing commas
      - Truncated output (recovers via brace-balance walk)

    Haiku in particular is prone to wrapping JSON output in explanation,
    so the strict json.loads path frequently fails on otherwise-good
    extractions. This recovers ~all of those.
    """
    if not raw:
        return None
    cleaned = raw.strip()

    # Strip markdown fences if present.
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")].rstrip()

    # Try strict parse first — fastest path when output is clean.
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Recovery 1: slice from first { to last }.
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}")
        candidate = cleaned[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            # Strip trailing commas before }/] which are common in
            # Haiku output and invalid JSON per the spec.
            patched = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                data = json.loads(patched)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    except ValueError:
        pass

    # Recovery 2: brace-balance walk — find the FIRST balanced {...}
    # block from the first opening brace. Handles cases where the LLM
    # appended a second incomplete JSON object after the good one.
    try:
        start = cleaned.index("{")
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(cleaned[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    patched = re.sub(r",(\s*[}\]])", r"\1", candidate)
                    try:
                        data = json.loads(patched)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        pass
                    break
    except ValueError:
        pass

    return None


def _map_to_legacy_shape(
    parsed: dict, doc_type: str, use_native_pdf: bool,
) -> dict:
    """Translate Claude's structured output into the dict shape expected
    by _extract_with_sections callers. Preserves every downstream read
    without any other code change."""
    metrics = parsed.get("metrics") or []
    segments = parsed.get("segments") or []
    mda = (parsed.get("mda_narrative") or "")[:20_000]
    confidence_profile = parsed.get("confidence_profile") or {}
    non_gaap_bridges = parsed.get("non_gaap_bridges") or []
    detected_period = (parsed.get("detected_period") or "").strip()

    # Normalise metric shape to match the legacy raw_items contract.
    # Legacy expects: metric_name, metric_value, metric_text, unit,
    # segment, source_snippet, confidence, period, _is_one_off, _qualifiers
    raw_items: list[dict] = []
    for m in metrics:
        if not isinstance(m, dict):
            continue
        name = (m.get("metric_name") or "").strip()
        if not name:
            continue
        raw_items.append({
            "metric_name":    name,
            "metric_value":   m.get("metric_value"),
            "metric_text":    (m.get("metric_text") or "").strip(),
            "unit":           m.get("unit"),
            "segment":        m.get("segment") or None,
            "source_snippet": (m.get("source_snippet") or "")[:500],
            "confidence":     float(m.get("confidence") or 0.85),
            "period":         m.get("period") or None,
            "_is_one_off":    bool(m.get("is_one_off", False)),
            "_qualifiers":    None,  # native path doesn't emit hedge-term breakdowns yet
        })

    # Segment data shape: {segments: [{name, revenue, margin, growth, notes}]}
    segment_data = {"segments": [s for s in segments if isinstance(s, dict)]} if segments else None

    # Guidance isn't in the legacy return dict — it's persisted separately
    # via _persist_transcript_items on some paths. Native path puts guidance
    # metrics into raw_items so they flow through normal persistence as
    # metric rows with segment='guidance' — matches the legacy convention.
    for g in parsed.get("guidance") or []:
        if not isinstance(g, dict):
            continue
        metric = (g.get("metric") or "").strip()
        if not metric:
            continue
        raw_items.append({
            "metric_name":    metric,
            "metric_value":   None,
            "metric_text":    (g.get("value_or_range") or "").strip(),
            "unit":           None,
            "segment":        "guidance",
            "source_snippet": f"guidance: {g.get('value_or_range', '')} ({g.get('specificity', '')})",
            "confidence":     0.9,
            "period":         None,
            "_is_one_off":    False,
            "_qualifiers":    {"specificity": g.get("specificity")},
        })

    return {
        "document_type":       doc_type,
        "extraction_method":   "native_claude_v1" + ("_pdf" if use_native_pdf else "_text"),
        "sections_found":      None,  # native path doesn't split into sections
        "section_types":       [],
        "items_extracted":     len(raw_items),
        "raw_items":           raw_items,
        "mda_narrative":       mda,
        "segment_data":        segment_data,
        "detected_period":     detected_period,
        "confidence_profile":  confidence_profile,
        "disappearance_flags": {},     # not emitted — legacy-only concept
        "non_gaap_bridge":     non_gaap_bridges,
        "non_gaap_comparison": {},     # not emitted — legacy-only concept
        "reconciliation":      None,
    }


def _empty_result(doc_type: str, *, reason: str) -> dict:
    """Uniform empty result so callers can branch on 'items_extracted == 0'
    and fall back to legacy extraction without breaking contract."""
    return {
        "document_type":       doc_type,
        "extraction_method":   f"native_claude_v1_failed:{reason}",
        "sections_found":      0,
        "section_types":       [],
        "items_extracted":     0,
        "raw_items":           [],
        "mda_narrative":       "",
        "segment_data":        None,
        "detected_period":     "",
        "confidence_profile":  {},
        "disappearance_flags": {},
        "non_gaap_bridge":     [],
        "non_gaap_comparison": {},
        "reconciliation":      None,
    }
