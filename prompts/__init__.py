"""
Centralised prompt templates for every LLM-powered service.
Each prompt enforces strict JSON output as per S9 of the spec.
Includes document-type-specific prompts and multi-document synthesis.
"""

# ─────────────────────────────────────────────────────────────────
# Document Classifier
# ─────────────────────────────────────────────────────────────────
DOCUMENT_CLASSIFIER = """\
You are a document classification agent for an investment research system.
Given the first 2000 characters of a document, classify it and extract metadata.

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "document_type": "earnings_release" | "transcript" | "presentation" | "10-Q" | "10-K" | "annual_report" | "investor_letter" | "broker_note" | "other",
  "company_ticker": "<ticker or null>",
  "period_label": "<e.g. 2026_Q1 or null>",
  "title": "<best guess title>",
  "language": "<ISO 639-1 code>",
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT (first 2000 chars) ---
{text}
"""

# ═══════════════════════════════════════════════════════════════════
# DOCUMENT-TYPE-SPECIFIC EXTRACTION PROMPTS
# ═══════════════════════════════════════════════════════════════════

EARNINGS_RELEASE_EXTRACTOR = """\
You are a financial data extraction agent analysing an EARNINGS RELEASE.
This document contains official reported numbers. Extract with precision.

Focus on: revenue, operating profit, net profit, EPS, dividends (reported and organic),
margins and margin changes (bps), volume metrics, cash flow metrics, balance sheet
highlights, segment breakdowns by region/division, YoY and sequential comparisons.

RULES:
- Extract ONLY explicitly stated numbers. Do NOT infer or calculate.
- Every metric must include the exact source snippet.
- Distinguish between reported and organic/underlying figures.
- If a value is ambiguous, set confidence below 0.8.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<n>",
  "metric_value": <number or null>,
  "metric_text": "<raw text>",
  "unit": "<EUR_M | USD_M | % | bps | x | null>",
  "segment": "<segment or null>",
  "geography": "<region or null>",
  "reported_vs_organic": "reported" | "organic" | "underlying" | "unknown",
  "source_snippet": "<verbatim>",
  "page_number": <int or null>,
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

TRANSCRIPT_EXTRACTOR = """\
You are an earnings call transcript analysis agent. Extract qualitative AND quantitative signals.

Extract THREE categories:

1. MANAGEMENT GUIDANCE - forward-looking statements with numbers or direction
2. TONE & LANGUAGE SIGNALS - confidence level, hedging, defensiveness, optimism
3. KEY Q&A EXCHANGES - important analyst questions and how management responded

For GUIDANCE items:
{{
  "category": "guidance",
  "metric_name": "<what is being guided>",
  "guidance_type": "range" | "point" | "directional" | "qualitative",
  "guidance_text": "<full statement>",
  "low": <number or null>,
  "high": <number or null>,
  "unit": "<unit or null>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

For TONE items:
{{
  "category": "tone",
  "topic": "<what topic>",
  "tone_signal": "confident" | "cautious" | "defensive" | "evasive" | "optimistic" | "concerned",
  "description": "<what was said and how>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

For Q&A items:
{{
  "category": "qa_exchange",
  "analyst_firm": "<firm if identifiable, else null>",
  "topic": "<topic>",
  "question_summary": "<what was asked>",
  "answer_quality": "direct" | "partial" | "evasive" | "deferred",
  "key_insight": "<what the answer reveals>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

Respond ONLY with a JSON array mixing all three types. No preamble, no markdown fences.

--- DOCUMENT TEXT ---
{text}
"""

BROKER_NOTE_EXTRACTOR = """\
You are an analyst report extraction agent. This is a BROKER NOTE or SELL-SIDE REPORT.

Focus on: analyst rating and changes, price target and changes, consensus vs actuals,
key debates or variant perceptions, estimate revisions, sector comparisons, key risks,
catalysts to watch.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "category": "rating" | "price_target" | "estimate" | "consensus_comparison" | "key_debate" | "risk" | "catalyst" | "valuation",
  "broker_firm": "<firm or null>",
  "analyst_name": "<name or null>",
  "metric_or_topic": "<what is discussed>",
  "current_value": "<current value or view>",
  "prior_value": "<previous if changed, else null>",
  "direction": "upgrade" | "downgrade" | "maintain" | "initiate" | null,
  "description": "<full context>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

PRESENTATION_EXTRACTOR = """\
You are a strategic document analysis agent. This is an INVESTOR PRESENTATION.

Focus on: medium-term financial targets, strategic priorities, capital allocation,
market outlook, operational KPIs and targets, M&A strategy, ESG commitments.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "category": "financial_target" | "strategic_priority" | "capital_allocation" | "market_outlook" | "operational_kpi" | "esg_target" | "metric",
  "metric_or_topic": "<what is discussed>",
  "value": "<stated value or target>",
  "timeframe": "<e.g. 2025, medium-term, by 2030>",
  "description": "<full context>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

# ─────────────────────────────────────────────────────────────────
# COMBINED extraction — KPIs + guidance in one pass (faster)
# ─────────────────────────────────────────────────────────────────
COMBINED_EXTRACTOR = """\
You are a financial data extraction agent. Extract ALL quantitative metrics
AND forward-looking guidance from this text in a SINGLE pass.

RULES:
- Extract ONLY explicitly stated numbers. Do NOT infer or calculate.
- Every item must include the exact source snippet.
- Set confidence below 0.8 for ambiguous values.
- Classify each item as "metric" or "guidance".

Respond ONLY with a JSON array. No preamble, no markdown fences.

For METRICS:
{{
  "type": "metric",
  "metric_name": "<n>",
  "metric_value": <number or null>,
  "metric_text": "<raw text>",
  "unit": "<EUR_M | USD_M | % | bps | x | null>",
  "segment": "<segment or null>",
  "geography": "<region or null>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

For GUIDANCE:
{{
  "type": "guidance",
  "metric_name": "<metric being guided>",
  "guidance_type": "range" | "point" | "directional",
  "guidance_text": "<full statement>",
  "low": <number or null>,
  "high": <number or null>,
  "unit": "<unit or null>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

# ─────────────────────────────────────────────────────────────────
# Generic fallbacks
# ─────────────────────────────────────────────────────────────────
KPI_EXTRACTOR = """\
You are a KPI extraction agent for an investment research system.
Extract ONLY explicitly stated quantitative metrics from the document text.

RULES:
- Do NOT infer or calculate any values.
- Every metric must include the exact source snippet.
- If a value is ambiguous, set confidence below 0.8.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<n>",
  "metric_value": <number or null>,
  "metric_text": "<raw text>",
  "unit": "<EUR_M | USD_M | % | bps | x | null>",
  "segment": "<segment or null>",
  "geography": "<region or null>",
  "source_snippet": "<verbatim>",
  "page_number": <int or null>,
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

GUIDANCE_EXTRACTOR = """\
You are a guidance extraction agent. Identify forward-looking management
guidance statements from the text below.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<metric being guided>",
  "guidance_type": "range" | "point" | "directional",
  "guidance_text": "<full guidance statement>",
  "low": <number or null>,
  "high": <number or null>,
  "unit": "<unit or null>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

THESIS_COMPARATOR = """\
You are a thesis comparison agent. Compare the new quarterly data
against the existing investment thesis.

CURRENT THESIS:
{thesis}

NEW QUARTER DATA:
{quarter_data}

PRIOR QUARTER DATA:
{prior_data}

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "thesis_direction": "strengthened" | "weakened" | "unchanged",
  "supporting_signals": ["..."],
  "weakening_signals": ["..."],
  "new_risks": ["..."],
  "unresolved_questions": ["..."],
  "summary": "<2-3 sentence summary>"
}}
"""

SURPRISE_DETECTOR = """\
You are a surprise detection agent. Identify results that deviate
meaningfully from prior expectations or consensus.

PRIOR EXPECTATIONS / GUIDANCE:
{expectations}

ACTUAL RESULTS:
{actuals}

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_or_topic": "<what was surprising>",
  "direction": "positive" | "negative",
  "magnitude": "minor" | "major",
  "description": "<explanation>",
  "source_snippet": "<evidence>"
}}
"""

IR_QUESTION_GENERATOR = """\
You are an IR question generation agent. Produce sharp, specific
follow-up questions an analyst should ask management on the next IR call.

COMPANY: {company}
PERIOD: {period}
KEY FINDINGS:
{findings}

THESIS:
{thesis}

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "topic": "<broad topic>",
  "question": "<the actual question>",
  "rationale": "<why this matters for the thesis>"
}}

Generate 5-8 questions.
"""

ONE_PAGE_BRIEFING = """\
You are a research briefing agent. Produce a concise one-page internal
research update using the structure below.

COMPANY: {company} ({ticker})
PERIOD: {period}

EXTRACTED KPIs:
{kpis}

THESIS COMPARISON:
{thesis_comparison}

SURPRISES:
{surprises}

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "what_happened": "<summary of the quarter>",
  "what_changed": "<key changes vs prior>",
  "thesis_status": "<impact on thesis>",
  "risks": "<updated risk picture>",
  "follow_ups": "<open items>",
  "bottom_line": "<1-2 sentence conclusion>"
}}
"""

# ═══════════════════════════════════════════════════════════════════
# MULTI-DOCUMENT SYNTHESIS
# Runs AFTER all documents are processed individually.
# Combines earnings, transcript, broker, and presentation findings.
# ═══════════════════════════════════════════════════════════════════
SYNTHESIS_BRIEFING = """\
You are a senior buy-side research analyst producing an internal research update.
You have been given outputs from processing MULTIPLE documents for the same
company and period. Synthesise all sources into one cohesive, opinionated briefing.

COMPANY: {company} ({ticker})
PERIOD: {period}

INVESTMENT THESIS:
{thesis}

=== SOURCE 1: REPORTED NUMBERS (from earnings release) ===
{earnings_data}

=== SOURCE 2: MANAGEMENT COMMENTARY (from transcript) ===
{transcript_data}

=== SOURCE 3: SELL-SIDE VIEWS (from broker notes) ===
{broker_data}

=== SOURCE 4: STRATEGIC SIGNALS (from presentations) ===
{presentation_data}

=== THESIS COMPARISON ===
{thesis_comparison}

=== SURPRISES DETECTED ===
{surprises}

INSTRUCTIONS:
- Cross-reference reported numbers with management commentary. Flag discrepancies.
- Compare management guidance vs sell-side expectations. Who is more credible?
- Identify what management emphasised vs what they avoided discussing.
- Assess management tone - are they genuinely confident or talking their book?
- Note any sell-side disagreements or variant perceptions worth investigating.
- Be direct and opinionated. This is for internal consumption, not publication.

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "headline": "<one-line summary for the portfolio manager>",
  "what_happened": "<synthesis of the reported quarter - numbers first, then context>",
  "management_message": "<what management wants you to believe, and how credible it is>",
  "what_the_street_thinks": "<sell-side consensus view and where it may be wrong>",
  "thesis_impact": "<how this changes the investment thesis - be specific and opinionated>",
  "key_debates": "<the 2-3 most important unresolved questions for the position>",
  "risks_updated": "<what is new in the risk picture>",
  "action_items": "<specific follow-ups: calls to make, data to check, models to update>",
  "bottom_line": "<2-3 sentence conclusion - would you add, hold, or trim?>"
}}
"""
