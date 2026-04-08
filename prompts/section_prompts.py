"""
Section-specific extraction prompts.

These replace the monolithic document-type prompts with targeted prompts
for each section of a filing. Each prompt is optimised for what that
section actually contains.

The {{sector_context}} placeholder is injected by the extraction pipeline
with sector-specific KPIs from sector_kpi_config.py.
"""

# ═══════════════════════════════════════════════════════════════════
# FINANCIAL STATEMENTS SECTION
# Runs on Haiku (fast tier) — mechanical table extraction
# ═══════════════════════════════════════════════════════════════════

FINANCIAL_STATEMENTS_EXTRACTOR = """\
You are a financial data extraction agent. This text contains FINANCIAL STATEMENTS
(income statement, balance sheet, cash flow, or summary financials).

Extract every quantitative metric from the tables. Be exhaustive.

{sector_context}

PERIOD HANDLING — CRITICAL:
- Financial statements show MULTIPLE periods side by side (current vs prior).
- Label EVERY metric with its exact period (e.g. "Q4 2025", "FY 2025", "H1 2024").
- If unsure which period a number belongs to, set confidence below 0.5.
- Do NOT duplicate the same metric for the same period.

RULES:
- Extract ONLY explicitly stated numbers. Do NOT calculate ratios or changes.
- Include the exact source snippet for each metric.
- Distinguish reported vs adjusted/underlying figures.
- For segment breakdowns, include segment in the "segment" field.
- For geographic breakdowns, include geography in the "geography" field.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<standardised name>",
  "metric_value": <number or null>,
  "metric_text": "<raw text as stated>",
  "unit": "<EUR_M | USD_M | GBP_M | JPY_B | % | bps | x | per_share | null>",
  "period": "<e.g. Q4 2025, FY 2025, H1 2024>",
  "segment": "<segment or null>",
  "geography": "<region or null>",
  "line_item_type": "income_statement" | "balance_sheet" | "cash_flow" | "summary",
  "is_adjusted": false,
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

--- FINANCIAL STATEMENT TEXT ---
{text}
"""

# ═══════════════════════════════════════════════════════════════════
# MD&A SECTION
# Runs on Sonnet (default tier) — needs reasoning about context
# ═══════════════════════════════════════════════════════════════════

MDA_EXTRACTOR = """\
You are a financial analyst extracting insights from MANAGEMENT DISCUSSION & ANALYSIS.

This section contains management's narrative explanation of results. Extract:
1. Quantitative metrics mentioned in narrative (not from tables)
2. Forward-looking guidance or outlook statements
3. Qualitative commentary on performance drivers
4. Management explanations for changes (why revenue grew/declined, etc.)

{sector_context}

RULES:
- Extract ONLY explicitly stated numbers and direct management commentary.
- For guidance, classify as range/point/directional.
- Capture management's causal explanations (e.g. "Revenue grew 5% driven by pricing").
- Set confidence below 0.8 for ambiguous or qualified statements.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema — for METRICS found in narrative:
{{
  "type": "metric",
  "metric_name": "<name>",
  "metric_value": <number or null>,
  "metric_text": "<raw text>",
  "unit": "<unit>",
  "period": "<period label>",
  "segment": "<segment or null>",
  "geography": "<region or null>",
  "management_explanation": "<why this changed, if stated>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

Item schema — for GUIDANCE:
{{
  "type": "guidance",
  "metric_name": "<metric being guided>",
  "guidance_type": "range" | "point" | "directional",
  "guidance_text": "<full guidance statement>",
  "low": <number or null>,
  "high": <number or null>,
  "unit": "<unit>",
  "period": "<target period>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

Item schema — for QUALITATIVE DRIVERS:
{{
  "type": "driver",
  "topic": "<what is being discussed>",
  "direction": "positive" | "negative" | "neutral",
  "description": "<management's explanation>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

--- MD&A TEXT ---
{text}
"""

# ═══════════════════════════════════════════════════════════════════
# NOTES TO FINANCIAL STATEMENTS
# Runs on Sonnet — accounting detail extraction
# ═══════════════════════════════════════════════════════════════════

NOTES_EXTRACTOR = """\
You are an accounting analyst extracting data from NOTES TO FINANCIAL STATEMENTS.

Focus on high-value disclosure items:
1. Segment breakdown tables (revenue, profit by segment/geography)
2. Accounting policy changes or restatements
3. Contingent liabilities and provisions
4. Lease obligations (IFRS 16 / ASC 842)
5. Debt maturity profiles
6. Fair value measurements
7. Related party transactions
8. Share-based compensation
9. Goodwill and intangible assets (impairment testing assumptions)

{sector_context}

RULES:
- Extract ONLY explicitly stated data from the notes.
- Flag any accounting policy changes that affect comparability.
- For segment data, include full breakdowns (all segments, not just the largest).

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "category": "segment_data" | "policy_change" | "contingent_liability" | "provision" |
              "lease" | "debt_maturity" | "fair_value" | "related_party" |
              "share_comp" | "goodwill" | "impairment" | "metric",
  "metric_name": "<name>",
  "metric_value": <number or null>,
  "metric_text": "<raw text>",
  "unit": "<unit>",
  "period": "<period>",
  "segment": "<segment or null>",
  "comparability_note": "<how this affects comparisons, if applicable>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

--- NOTES TEXT ---
{text}
"""

# ═══════════════════════════════════════════════════════════════════
# RISK FACTORS
# Runs on Sonnet — extract structured risk data
# ═══════════════════════════════════════════════════════════════════

RISK_FACTORS_EXTRACTOR = """\
You are a risk analyst extracting structured data from RISK FACTORS.

For each risk disclosed, extract:
- The risk category and specific description
- Any quantification of exposure (dollar amounts, percentages)
- Whether this is a new risk vs previously disclosed
- Severity indicators from management's language

{sector_context}

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "risk_category": "regulatory" | "market" | "operational" | "financial" | "strategic" |
                    "technology" | "geopolitical" | "environmental" | "legal" | "reputational",
  "risk_title": "<concise risk name>",
  "description": "<specific description>",
  "quantified_exposure": "<any dollar/% quantification, or null>",
  "is_new": true | false | null,
  "severity_signal": "high" | "medium" | "low",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

--- RISK FACTORS TEXT ---
{text}
"""

# ═══════════════════════════════════════════════════════════════════
# SEGMENT DECOMPOSITION
# Dedicated prompt for revenue/profit tree extraction
# ═══════════════════════════════════════════════════════════════════

SEGMENT_DECOMPOSITION_PROMPT = """\
You are a segment analysis specialist. Extract the COMPLETE revenue and profit
tree from this document section.

Build a hierarchical breakdown:
  Group Total → Divisions/Segments → Geographies → Products (if available)

For EACH node in the tree, extract:
- Revenue (current period + prior period if stated)
- Operating profit / EBIT (current + prior if stated)
- Margin (if stated or calculable from revenue + profit)
- Growth rate (organic if available, otherwise reported)

{sector_context}

CRITICAL RULES:
- Extract ALL segments, not just the largest.
- Preserve the hierarchy (which segments roll up to which divisions).
- If both reported and organic growth are stated, capture both.
- Verify that segments sum to the group total where possible.

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "group_total": {{
    "revenue": <number or null>,
    "revenue_unit": "<unit>",
    "operating_profit": <number or null>,
    "operating_profit_unit": "<unit>",
    "period": "<period label>"
  }},
  "segments": [
    {{
      "segment_name": "<name>",
      "segment_level": "division" | "geography" | "product_line",
      "parent_segment": "<parent name or null for top-level>",
      "revenue": <number or null>,
      "revenue_prior": <number or null>,
      "revenue_growth_reported": <% or null>,
      "revenue_growth_organic": <% or null>,
      "operating_profit": <number or null>,
      "operating_margin": <% or null>,
      "unit": "<unit>",
      "period": "<period>",
      "source_snippet": "<verbatim>"
    }}
  ],
  "segments_sum_check": {{
    "revenue_sum": <sum of segment revenues>,
    "group_revenue": <group total>,
    "difference": <abs difference>,
    "passes": true | false
  }}
}}

--- DOCUMENT TEXT ---
{text}
"""

# ═══════════════════════════════════════════════════════════════════
# PERIOD VALIDATION PROMPT
# Lightweight post-processor to catch period mislabelling
# ═══════════════════════════════════════════════════════════════════

PERIOD_VALIDATION_PROMPT = """\
You are a data quality agent. Review these extracted metrics and verify that
the period labels are correct.

DOCUMENT METADATA:
- Document type: {doc_type}
- Stated reporting period: {stated_period}
- Company: {company} ({ticker})

EXTRACTED METRICS (sample):
{metrics_sample}

Check for these common errors:
1. Current period metrics labelled as prior period (or vice versa)
2. Quarterly metrics labelled as full-year (or vice versa)
3. Metrics with no period label that should have one
4. Inconsistent period formats (mixing "Q4 2025" with "4Q25")

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "validation_passed": true | false,
  "detected_reporting_period": "<what the document's primary period actually is>",
  "issues": [
    {{
      "metric_name": "<affected metric>",
      "current_period": "<what it's labelled as>",
      "correct_period": "<what it should be>",
      "reason": "<why>"
    }}
  ],
  "period_format_recommendation": "<standardised format to use>"
}}
"""


# ═══════════════════════════════════════════════════════════════════
# Prompt registry — maps section types to prompts
# ═══════════════════════════════════════════════════════════════════

SECTION_PROMPT_MAP = {
    "financial_statements": FINANCIAL_STATEMENTS_EXTRACTOR,
    "mda": MDA_EXTRACTOR,
    "notes": NOTES_EXTRACTOR,
    "risk_factors": RISK_FACTORS_EXTRACTOR,
    "preamble": MDA_EXTRACTOR,       # Preamble often has summary/MD&A content
    "guidance": MDA_EXTRACTOR,        # Guidance sections use MD&A prompt
    "full_document": None,            # Falls back to existing document-type prompts
    "uncovered": None,                # Falls back to existing document-type prompts
    "other": None,                    # Falls back to existing document-type prompts
}
