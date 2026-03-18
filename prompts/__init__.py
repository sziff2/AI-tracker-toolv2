"""
Prompt library for the Research CoWork Agent.

Includes probabilistic / Bayesian output blocks in the briefing prompts.
All LLM responses must be pure JSON — no markdown fences, no preamble.
"""

# ═══════════════════════════════════════════════════════════════════
# DOCUMENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════
DOCUMENT_CLASSIFIER = """\
You are a document classification agent for a buy-side investment research firm.
Classify this financial document from the first 2000 characters.

Text:
{text}

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "document_type": "earnings_release" | "earnings_transcript" | "broker_note" | "investor_presentation" | "annual_report" | "regulatory_filing" | "press_release" | "other",
  "company_ticker": "<ticker or null>",
  "period_label": "<e.g. 2024_Q4, 2024_FY, or null>",
  "title": "<document title or null>",
  "language": "<ISO 639-1 code, default en>",
  "confidence": <0.0-1.0>
}}
"""

# ═══════════════════════════════════════════════════════════════════
# EXTRACTORS
# ═══════════════════════════════════════════════════════════════════

EARNINGS_RELEASE_EXTRACTOR = """\
You are a financial data extraction agent. This is an EARNINGS RELEASE.

Focus on: revenue, operating profit, EBIT, EBITDA, net income, EPS, margins,
free cash flow, net debt, working capital, organic growth, volume/price/mix breakdown.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<standardised metric name>",
  "metric_value": <numeric value or null>,
  "metric_text": "<raw text as stated>",
  "unit": "<€m, $m, %, x, etc. or null>",
  "segment": "<segment name or null>",
  "geography": "<geography or null>",
  "period": "<period this metric relates to>",
  "source_snippet": "<verbatim text>",
  "page_number": <int or null>,
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

TRANSCRIPT_EXTRACTOR = """\
You are a transcript analysis agent. This is an EARNINGS CALL TRANSCRIPT.

Focus on: management commentary on results, forward guidance (quantitative and
qualitative), tone, questions raised by analysts, topics management avoided,
strategic priorities, risks acknowledged or dismissed.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "category": "guidance" | "commentary" | "strategy" | "risk" | "analyst_question" | "management_tone",
  "speaker": "<name or role>",
  "metric_or_topic": "<what is discussed>",
  "metric_value": <numeric value or null>,
  "metric_text": "<raw text as stated>",
  "unit": "<unit or null>",
  "period": "<period this refers to>",
  "description": "<full context>",
  "source_snippet": "<verbatim>",
  "confidence": <0.0-1.0>
}}

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

COMBINED_EXTRACTOR = """\
You are a financial data extraction agent. Extract ALL quantitative metrics
AND forward-looking guidance from this text in a SINGLE pass.

RULES:
- Extract ONLY explicitly stated numbers. Do NOT infer or calculate.
- Every item must include the exact source snippet.
- Set confidence below 0.8 for ambiguous values.
- Classify each item as "metric" or "guidance".

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "type": "metric" | "guidance",
  "metric_name": "<standardised name>",
  "metric_value": <number or null>,
  "metric_text": "<raw text as stated>",
  "unit": "<unit or null>",
  "segment": "<segment or null>",
  "period": "<period or null>",
  "source_snippet": "<verbatim text>",
  "confidence": <0.0-1.0>
}}

--- DOCUMENT TEXT ---
{text}
"""

# ═══════════════════════════════════════════════════════════════════
# THESIS COMPARISON
# (Legacy — used in single-doc pipeline; see thesis_comparator.py for V2)
# ═══════════════════════════════════════════════════════════════════
THESIS_COMPARATOR = """\
Compare the new quarterly data against the existing investment thesis.

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

# ═══════════════════════════════════════════════════════════════════
# SURPRISE DETECTOR
# ═══════════════════════════════════════════════════════════════════
SURPRISE_DETECTOR = """\
You are a surprise detection agent. Identify results that deviate
meaningfully from prior expectations or consensus.

PRIOR EXPECTATIONS / GUIDANCE:
{expectations}

ACTUAL RESULTS:
{actuals}

CRITICAL RULES:
1. Only flag surprises where you have BOTH the prior expectation AND the actual result
   in the data above. Do NOT invent expectations or consensus figures.
2. If no prior guidance or expectations data is available, return an empty array [].
3. Compare ONLY against explicitly stated guidance, targets, or prior-period results
   from the data provided. Never fabricate market expectations.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_or_topic": "<what was surprising>",
  "direction": "positive" | "negative",
  "magnitude": "minor" | "major",
  "prior_value": "<the explicit prior expectation or guidance figure>",
  "actual_value": "<the actual reported figure>",
  "description": "<explanation using only data from above>",
  "source_snippet": "<evidence>"
}}
"""

# ═══════════════════════════════════════════════════════════════════
# IR QUESTION GENERATOR
# ═══════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════
# ONE-PAGE BRIEFING  (single-document pipeline)
# Now includes probabilistic scenario and Bayesian signal blocks.
# ═══════════════════════════════════════════════════════════════════
ONE_PAGE_BRIEFING = """\
You are a senior buy-side research analyst. Produce a concise one-page internal
research update AND a probabilistic assessment of the investment case.

COMPANY: {company} ({ticker})
PERIOD: {period}

EXTRACTED KPIs:
{kpis}

THESIS COMPARISON:
{thesis_comparison}

SURPRISES:
{surprises}

═══════════════════════════════════
CRITICAL RULES — READ CAREFULLY:
1. Use ONLY numbers from the data above. Do NOT invent figures or comparisons.
2. If consensus/expectations data is not provided, do NOT fabricate it.
3. Every number must be traceable to the source data above.
4. If information is missing, say so — do NOT fill gaps with assumptions.
5. For probabilities: assign your best estimate based on the evidence.
   Probabilities across scenarios for the same dimension must sum to 1.0.
6. Be direct and opinionated — this is for internal use.
═══════════════════════════════════

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "what_happened": "<summary of the quarter — ONLY using numbers from the data>",
  "what_changed": "<key changes vs prior>",
  "thesis_status": "<impact on thesis>",
  "risks": "<updated risk picture>",
  "follow_ups": "<open items>",
  "bottom_line": "<1-2 sentence conclusion>",
  "probabilistic": {{
    "scenarios": [
      {{
        "label": "Bull",
        "probability": <0.0-1.0>,
        "description": "<what this scenario looks like>",
        "key_trigger": "<what causes this outcome>",
        "thesis_impact": "strengthened" | "weakened" | "neutral",
        "implied_return": "<qualitative or % range>"
      }},
      {{
        "label": "Base",
        "probability": <0.0-1.0>,
        "description": "<what this scenario looks like>",
        "key_trigger": "<what causes this outcome>",
        "thesis_impact": "strengthened" | "weakened" | "neutral",
        "implied_return": "<qualitative or % range>"
      }},
      {{
        "label": "Bear",
        "probability": <0.0-1.0>,
        "description": "<what this scenario looks like>",
        "key_trigger": "<what causes this outcome>",
        "thesis_impact": "weakened" | "neutral",
        "implied_return": "<qualitative or % range>"
      }}
    ],
    "bayesian_signals": [
      {{
        "assumption": "<the prior belief being updated>",
        "prior_view": "<what was believed before this period>",
        "new_evidence": "<what the data or commentary showed>",
        "posterior_direction": "strengthened" | "weakened" | "unchanged" | "reversed",
        "update_magnitude": "large" | "moderate" | "small",
        "confidence": <0.0-1.0>,
        "source": "earnings" | "transcript" | "broker" | "guidance"
      }}
    ],
    "key_assumptions": [
      {{
        "assumption": "<key thesis assumption>",
        "probability": <0.0-1.0>,
        "direction": "positive" | "negative" | "neutral",
        "rationale": "<why this probability is assigned>",
        "key_watch": "<what to monitor>"
      }}
    ],
    "overall_conviction_direction": "buy" | "hold" | "sell" | "watch",
    "overall_conviction_score": <0.0-1.0>,
    "conviction_rationale": "<1-2 sentence explanation>"
  }}
}}
"""

# ═══════════════════════════════════════════════════════════════════
# MULTI-DOCUMENT SYNTHESIS  (batch pipeline)
# Now includes probabilistic scenario and Bayesian signal blocks.
# ═══════════════════════════════════════════════════════════════════
SYNTHESIS_BRIEFING = """\
You are a senior buy-side research analyst producing an internal research update.
You have been given outputs from processing MULTIPLE documents for the same
company and period. Synthesise all sources into one cohesive, opinionated briefing
AND a probabilistic assessment of the investment case.

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

═══════════════════════════════════
CRITICAL RULES — READ CAREFULLY:
1. Use ONLY numbers that appear in the source data above. Do NOT invent, estimate,
   or hallucinate any figures — no "expectations", "consensus", or "vs" comparisons
   unless the comparison figure is EXPLICITLY stated in the sources above.
2. If you do not have sell-side estimates or consensus numbers in the data, do NOT
   make up comparison figures. Simply describe the reported results on their own.
3. Every number you cite must be traceable to a specific source section above.
4. If data is missing or insufficient, say so explicitly. Do NOT fill gaps with
   assumptions or fabricated figures.
5. Cross-reference reported numbers with management commentary. Flag discrepancies.
6. Compare management guidance vs sell-side expectations ONLY if both are in the data.
7. Identify what management emphasised vs what they avoided discussing.
8. Assess management tone — are they genuinely confident or talking their book?
9. Be direct and opinionated. This is for internal consumption, not publication.
10. PROBABILISTIC BLOCK: Assign probabilities based on ALL available evidence above.
    Probabilities across all scenarios must sum to 1.0. Be explicit about what
    drives each scenario and what the Bayesian update to each key assumption is.
    A large positive earnings surprise should shift the bull probability upward.
    A guidance cut should shift bear probability upward. Be consistent.
═══════════════════════════════════

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "headline": "<one-sentence headline>",
  "what_happened": "<summary of results — ONLY using numbers from the data>",
  "what_changed": "<key changes vs prior period>",
  "management_message": "<what management wants the market to take away>",
  "thesis_impact": "<how results affect the thesis>",
  "risks": "<updated risk picture>",
  "follow_ups": "<open items for next IR call>",
  "bottom_line": "<1-2 sentence opinionated conclusion>",
  "probabilistic": {{
    "scenarios": [
      {{
        "label": "Bull",
        "probability": <0.0-1.0>,
        "description": "<what this scenario looks like over the next 12-18 months>",
        "key_trigger": "<what must occur for this to play out>",
        "thesis_impact": "strengthened",
        "implied_return": "<qualitative or % range>"
      }},
      {{
        "label": "Base",
        "probability": <0.0-1.0>,
        "description": "<what this scenario looks like over the next 12-18 months>",
        "key_trigger": "<the central path>",
        "thesis_impact": "neutral" | "strengthened" | "weakened",
        "implied_return": "<qualitative or % range>"
      }},
      {{
        "label": "Bear",
        "probability": <0.0-1.0>,
        "description": "<what this scenario looks like over the next 12-18 months>",
        "key_trigger": "<what causes this to go wrong>",
        "thesis_impact": "weakened",
        "implied_return": "<qualitative or % range>"
      }}
    ],
    "bayesian_signals": [
      {{
        "assumption": "<specific thesis assumption being updated>",
        "prior_view": "<what was believed entering this period>",
        "new_evidence": "<specific data point or commentary that updates the view>",
        "posterior_direction": "strengthened" | "weakened" | "unchanged" | "reversed",
        "update_magnitude": "large" | "moderate" | "small",
        "confidence": <0.0-1.0>,
        "source": "earnings" | "transcript" | "broker" | "guidance"
      }}
    ],
    "key_assumptions": [
      {{
        "assumption": "<key assumption underpinning the thesis>",
        "probability": <0.0-1.0 that this assumption holds>,
        "direction": "positive" | "negative" | "neutral",
        "rationale": "<concise reason for this probability>",
        "key_watch": "<leading indicator or event to monitor>"
      }}
    ],
    "overall_conviction_direction": "buy" | "add" | "hold" | "reduce" | "sell" | "watch",
    "overall_conviction_score": <0.0-1.0>,
    "conviction_rationale": "<1-2 sentence explanation of the overall stance>"
  }}
}}
"""
