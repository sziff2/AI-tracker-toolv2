# Financial Extraction — Eval Programme & Process Improvements

## Context

The architecture in `_0_8_financial-extraction-architecture.md` reduces the
extraction error rate from ~15–25% to a target of <2% by pre-segmenting
documents before any LLM sees them. This document defines:

1. **The eval programme** — what to measure, how, and when
2. **Process improvements** beyond the architecture doc

---

## 1. Eval Programme

### 1.1 Eval Categories

| Category | What it measures | Pass threshold | File |
|---|---|---|---|
| **Accuracy** | Extracted values vs ground truth | ≥98% of values within 0.5% | `financial_extraction_evals.py::TestValueAccuracy` |
| **Hallucination** | Invented data, calibration, contamination | 0 fabricated values; calibration error <15% per bin | `financial_extraction_evals.py::TestHallucination` |
| **Reconciliation** | Cross-check sensitivity & specificity | ≥80% sensitivity, <5% false positives | `financial_extraction_evals.py::TestReconciliation` |
| **Efficiency** | Token cost and latency | ≤5,500 tokens; ≤13 seconds | `financial_extraction_evals.py::TestEfficiency` |
| **Regression** | New ≥ legacy on accuracy | New pipeline accuracy ≥ legacy accuracy | `financial_extraction_evals.py::TestRegression` |

### 1.2 Ground Truth Dataset

Build `evals/fixtures/ground_truth.json` (schema in `ground_truth_schema.json`).

**Minimum corpus:** 10 real documents. Suggested mix:

| Type | Count | Why |
|---|---|---|
| Earnings release (US GAAP, USD) | 3 | Baseline — dense, multi-period |
| Earnings release (IFRS, EUR/GBP) | 3 | Currency/unit detection stress test |
| Annual report / 10-K | 2 | Longer document, footnotes stress test |
| Document with restatement | 1 | Reconciler sensitivity |
| Non-standard format (bank, insurer) | 1 | Parser robustness |

For each document manually verify **at minimum**:
- Revenue (Q and FY)
- Net income (Q and FY)
- Free cash flow
- Total assets (most recent balance sheet date)
- One segment revenue (if applicable)

Spot-check these against the source PDF by hand — do not trust another LLM to
generate ground truth.

### 1.3 Running the Evals

```powershell
# Full suite
pytest evals\financial_extraction_evals.py -v

# By category (fast feedback loop during development)
pytest evals\financial_extraction_evals.py -v -m accuracy
pytest evals\financial_extraction_evals.py -v -m hallucination
pytest evals\financial_extraction_evals.py -v -m reconciliation
pytest evals\financial_extraction_evals.py -v -m efficiency
pytest evals\financial_extraction_evals.py -v -m regression

# Generate a report
pytest evals\financial_extraction_evals.py --tb=short --json-report --json-report-file=evals\reports\latest.json
```

### 1.4 Continuous Improvement Loop (the "AutoResearch" feedback cycle)

```
Production run
    │
    ▼
Log every extraction:
  • doc_id, ticker, period, label, extracted_value, confidence, tokens_used, latency_ms
    │
    ▼
Weekly: sample 20 production extractions
  • Human spot-check: is the value correct?
  • If wrong: add to ground_truth.json
    │
    ▼
Run eval suite on new ground truth
  • Did the failing case expose a systemic issue?
  • Update parser heuristics / prompts / reconciler accordingly
    │
    ▼
Run regression eval to confirm no regressions
    │
    └── repeat
```

This loop is the core of continuous improvement: each production error that is
discovered and verified becomes a permanent test case. The eval suite grows
alongside the pipeline's knowledge of edge cases.

### 1.5 Eval Metrics to Track Over Time

Log these to a simple `evals/reports/history.csv` after each eval run:

```
date, pipeline_version, accuracy_pct, hallucination_rate, reconciliation_sensitivity,
reconciliation_fpr, avg_tokens_per_doc, avg_latency_s, total_docs_tested
```

Plot the trend. The goal is a chart that goes up and to the right on accuracy /
sensitivity and down on tokens / latency.

---

## 2. Process Improvements

The following improvements go beyond the architecture doc and address the
four objectives: fewer hallucinations, higher accuracy, faster, cheaper.

### 2.1 Prompt-Level Improvements (Fewer Hallucinations)

**Problem:** LLMs fill in gaps when data is absent.

**Fix — Explicit "not found" instruction:**

In every statement extractor prompt, add:

```
IMPORTANT: If a line item is not present in the table above, DO NOT include it
in your response. Return ONLY line items explicitly shown in the data I gave you.
It is correct to return an empty array [] if the table has no extractable data.
```

**Fix — Grounded output only:**

After extraction, run a "grounding check": every value in the LLM output must
appear verbatim (or as a parseable variant) in the raw table text. Any value
that can't be traced back to source text gets flagged with `"grounded": false`
and a low confidence score (0.3).

This catches the most common hallucination pattern: the LLM calculates a
derived value (e.g. gross margin %) not shown in the table and presents it as
if it were extracted.

### 2.2 Structural Parser Improvements (Higher Accuracy)

**Problem:** The `classify_table()` heuristic fails on non-standard layouts
(banks with NII tables, insurance with combined ratios, etc.).

**Fix — Two-pass classification:**

1. First pass: pure heuristics (current approach) → fast, free
2. If `StatementType.UNKNOWN`, second pass: one-shot LLM classification prompt
   with the raw table (≤300 tokens). Cost: ~$0.0002 per unknown table.

This keeps 95%+ of tables in the free path while handling edge cases.

**Fix — Multi-row header support:**

Many earnings releases use two-row headers:

```
              Q4 2025        Q4 2024
              $M     %       $M     %
Revenue       8,234  —      7,891   —
```

The current `extract_periods_from_headers()` only reads the first row. Add
detection for "column group" headers by checking if row 1 contains only
numeric or blank values (it's a data row) vs. label values (it's a second
header row).

**Fix — Handle "restated" / "as reported" column disambiguation:**

Some tables have duplicate period columns: `Q4 2024 (Restated)` and
`Q4 2024 (As reported)`. Add logic to:
1. Detect these duplicates
2. Keep "Restated" as the primary value
3. Store "As reported" as a separate field: `value_as_reported`
4. Flag the metric with `"has_restatement": true`

### 2.3 Extraction Prompt Improvements (Higher Accuracy)

**Fix — Type-aware extraction prompts:**

The architecture doc has one prompt per statement type. Add **type-specific
field lists** so the model knows what to look for. Examples:

Income Statement prompt should include:
```
For an income statement, the primary metrics are:
Revenue / Net Sales, Gross Profit, Operating Profit / EBIT, EBITDA,
Net Income / Net Profit, EPS (basic), EPS (diluted).
Also extract any subtotals or adjusted/underlying variants present.
```

This dramatically reduces both missed extractions (underfitting) and invented
fields (hallucination from trying to "complete" a template).

**Fix — Confidence score guidance:**

Current prompts ask for `confidence: 0.0-1.0` but give no criteria. Add:

```
Assign confidence as follows:
  1.0 — value is explicit, unambiguous, no calculation required
  0.8 — value required minor inference (e.g. matching a footnote definition)
  0.6 — value was partially obscured or required calculation from subtotals
  0.4 — value was inferred from narrative text, not a table
  0.0 — value not found; do not include in output
```

### 2.4 Cost Reduction

**Estimated current cost breakdown (per document):**

| Step | Tokens | Cost @ $3/MTok (Sonnet) |
|---|---|---|
| 6 × statement extractors (parallel) | ~3,500 | ~$0.011 |
| Notes agent | ~500 | ~$0.0015 |
| Narrative agent | ~500 | ~$0.0015 |
| Reconciliation | 0 (no LLM) | $0 |
| **Total** | **~4,500** | **~$0.013** |

**Cost reduction levers:**

1. **Haiku for low-value tables:** Use `claude-haiku-4-5` for KPI tables and
   footnotes extraction. These are simpler extraction tasks. Save ~30% on those
   calls. Route P&L / BS / CF extractions to Sonnet for accuracy.

2. **Cache structural parser output:** The `FinancialDocumentStructure` is
   deterministic (no LLM). Cache it by `hash(pdf_bytes)`. If the same PDF is
   processed twice (e.g. during testing or retry), skip the parser entirely.

3. **Skip agents for missing statement types:** If the structural parser found
   no cash flow table, don't launch the Cash Flow Agent at all. Currently the
   architecture runs all 6 agents regardless.

4. **Batch small documents:** For companies that publish supplemental data
   packs (10-20 pages), batch 3-4 of them into a single LLM call using the
   multi-document schema from the batch processing API.

### 2.5 Monitoring & Observability (for the AutoResearch loop)

Add a `ExtractionTelemetry` object to every extraction run:

```python
@dataclass
class ExtractionTelemetry:
    doc_id: str
    ticker: str
    pipeline_version: str           # semver tag
    timestamp: datetime
    statement_types_found: list[str]
    tables_found: int
    tables_classified: int
    tables_unknown: int             # failed classification rate
    total_metrics_extracted: int
    low_confidence_count: int       # items with confidence < 0.7
    reconciliation_passed: bool
    reconciliation_issues: list[dict]
    total_tokens_used: int
    latency_ms: int
    model_used: str
```

Log this to your database on every production run. Build a simple weekly
summary query:

```sql
SELECT
    DATE_TRUNC('week', timestamp) AS week,
    AVG(total_tokens_used) AS avg_tokens,
    AVG(latency_ms) AS avg_latency_ms,
    AVG(low_confidence_count::float / NULLIF(total_metrics_extracted, 0)) AS low_conf_rate,
    SUM(CASE WHEN NOT reconciliation_passed THEN 1 ELSE 0 END)::float
      / COUNT(*) AS recon_failure_rate
FROM extraction_telemetry
GROUP BY 1
ORDER BY 1;
```

Rising `low_conf_rate` or `recon_failure_rate` is an early warning that the
pipeline is degrading (new document formats, model behaviour changes, etc.)
before users notice.

### 2.6 Handling Non-Standard Documents

The architecture falls back to the "one big prompt" approach for unknown
document types. Add a third path: **semi-structured extraction**.

For documents where the parser finds tables but can't classify them:

```
Document
    │
    ├─ Tables classified? ──YES──▶ Segmented pipeline (current Step 1-3)
    │
    ├─ Tables found, not classified? ──▶ NEW: semi-structured path
    │     • Pass raw tables to LLM with: "Classify and extract from these tables"
    │     • Use a single prompt with all unclassified tables (capped at 10)
    │     • Tag all outputs with "extraction_path": "semi_structured"
    │
    └─ No tables found? ──▶ Legacy one-big-prompt fallback
```

This three-path design means the legacy fallback is reserved only for truly
unstructured text (rare), while semi-structured handles the long tail of
unusual tables without paying the full parallel LLM cost.

---

## 3. Implementation Priority

| Improvement | Effort | Impact | Do when |
|---|---|---|---|
| Explicit "not found" instruction in prompts | 30 min | High (hallucination) | Phase 0.8 |
| Grounding check post-extraction | 2 hrs | High (hallucination) | Phase 0.8 |
| `ExtractionTelemetry` logging | 2 hrs | Medium (observability) | Phase 0.8 |
| Build ground_truth.json (10 docs) | 4 hrs | Critical (evals can't run without it) | Phase 0.8 |
| Two-pass table classification (LLM fallback) | 3 hrs | Medium (accuracy on edge cases) | Phase 1 |
| Multi-row header support | 2 hrs | Medium (accuracy) | Phase 1 |
| Confidence score criteria in prompts | 1 hr | Medium (calibration) | Phase 1 |
| Haiku routing for KPI/footnotes | 1 hr | Low-medium (cost) | Phase 2 |
| Three-path document routing | 4 hrs | Low (long tail) | Phase 2 |
| Restatement column handling | 3 hrs | Low (edge case) | Phase 2 |

---

## 4. Files in This Eval Programme

```
evals/
├── financial_extraction_evals.py   ← main pytest eval suite
├── fixtures/
│   ├── ground_truth_schema.json    ← schema + examples for ground truth dataset
│   ├── ground_truth.json           ← YOU BUILD THIS (10 real documents)
│   ├── pdfs/                       ← PDF test fixtures (git-ignored if large)
│   └── synthetic/                  ← Synthetic test PDFs for hallucination tests
└── reports/
    ├── latest.json                 ← most recent pytest-json-report output
    └── history.csv                 ← trend tracking over time

docs/
└── financial_extraction_eval_design.md  ← this file
```
