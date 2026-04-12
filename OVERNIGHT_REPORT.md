# Overnight Test & Architecture Report — 2026-04-11

## Part 1: Full Pipeline Architecture

### Complete Workflow Schematic

```
═══════════════════════════════════════════════════════════════════
 PHASE A — Document Ingestion (runs once per document)
═══════════════════════════════════════════════════════════════════

  HARVESTER (weekly via Celery Beat + manual)
  ┌─────────────────────────────────────────────────┐
  │ services/harvester/__init__.py: run_harvest()   │
  │                                                 │
  │  ┌─ sec_edgar.py: fetch_sec_edgar()             │
  │  │  └─ EDGAR API → 10-K, 10-Q, 8-K filings     │
  │  │                                              │
  │  ├─ investegate.py: fetch_investegate()          │
  │  │  └─ UK RNS → earnings releases              │
  │  │                                              │
  │  ├─ ir_scraper.py: scrape_ir_page()             │
  │  │  └─ Regex scraper → PDF links                │
  │  │                                              │
  │  └─ ir_llm_scraper.py: scrape_ir_with_llm()    │
  │     └─ LLM scraper → complex IR sites           │
  │                                                 │
  │  dispatcher.py: dispatch_candidates()            │
  │  └─ Dedup, download, save to documents table     │
  └─────────────────────────────────────────────────┘
       │
       ▼
  COVERAGE MONITOR
  ┌─────────────────────────────────────────────────┐
  │ services/harvester/coverage.py                  │
  │  expected_period() → check_coverage()           │
  │  └─ Compares latest doc period vs expected       │
  │  └─ Appended to weekly Teams report              │
  └─────────────────────────────────────────────────┘
       │
       ▼
  DOCUMENT PROCESSING (triggered by "Analyse Period" button)
  ┌─────────────────────────────────────────────────┐
  │ services/background_processor.py                │
  │  run_batch_pipeline() → _process_one_doc()      │
  │                                                 │
  │  Step 1: PARSE                                  │
  │  ├─ document_parser.py: process_document()      │
  │  │  ├─ PDF → PyMuPDF + pdfplumber               │
  │  │  ├─ HTML → BeautifulSoup (SEC filings)        │
  │  │  └─ DOCX → python-docx                        │
  │  └─ Saves to document_sections table             │
  │                                                 │
  │  Step 2: EXTRACT                                │
  │  ├─ metric_extractor.py:                        │
  │  │  extract_by_document_type()                  │
  │  │  ├─ 10-Q/10-K/annual: section_splitter.py    │
  │  │  │  └─ split_into_sections() → FilingSection  │
  │  │  │  └─ Parallel LLM calls per section (Haiku) │
  │  │  └─ transcript/other: _extract_legacy()       │
  │  │     └─ Smart chunking → parallel extraction   │
  │  │                                              │
  │  ├─ Enrichment (parallel):                      │
  │  │  ├─ qualifier_extractor.py (hedge/one-off)    │
  │  │  ├─ segment_extractor.py (decomposition)      │
  │  │  └─ period_validator.py (disambiguation)      │
  │  │                                              │
  │  └─ Saves to:                                   │
  │     ├─ extracted_metrics table                    │
  │     ├─ extraction_profiles table                  │
  │     └─ research_outputs (extraction_context)      │
  │                                                 │
  │  Step 3: DOCUMENT ANALYSIS (NEW)                │
  │  ├─ If transcript:                              │
  │  │  └─ _analyse_document_with_llm()             │
  │  │     └─ prompts/agents/transcript_deep_dive.txt│
  │  │     └─ Saves to research_outputs              │
  │  │        (output_type='transcript_analysis')     │
  │  │                                              │
  │  ├─ If presentation:                            │
  │  │  └─ _analyse_document_with_llm()             │
  │  │     └─ prompts/agents/presentation_analysis.txt│
  │  │     └─ Saves to research_outputs              │
  │  │        (output_type='presentation_analysis')   │
  │  │                                              │
  │  Step 4: SYNTHESIS (legacy — still runs)        │
  │  ├─ thesis_comparator.py: compare_thesis()      │
  │  ├─ surprise_detector.py: detect_surprises()    │
  │  └─ output_generator.py: generate_briefing()    │
  │     └─ Saves to research_outputs                 │
  │                                                 │
  │  Processing job status: pending → processing →   │
  │  completed (tracked in processing_jobs table)     │
  └─────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
 CONTEXT BUILDER — Bridge between DB and Agents
═══════════════════════════════════════════════════════════════════

  services/context_builder.py: build_agent_context()
  ┌─────────────────────────────────────────────────┐
  │ Called ONCE per pipeline run. Queries DB and     │
  │ builds a single dict with everything agents need:│
  │                                                 │
  │ Identity:                                       │
  │  ├─ _build_company_meta() → ticker, name, sector│
  │  └─ period_label                                │
  │                                                 │
  │ Thesis & Metrics:                               │
  │  ├─ build_thesis_context() → core thesis text   │
  │  ├─ build_kpi_summary() → deduped key-value KPIs│
  │  ├─ build_guidance_summary() → guidance items   │
  │  ├─ build_prior_period_context() → prior summary│
  │  └─ build_tracked_kpi_context() → analyst KPIs  │
  │                                                 │
  │ Enriched Extraction:                            │
  │  └─ build_extraction_context()                  │
  │     └─ Queries research_outputs WHERE           │
  │        output_type='extraction_context'          │
  │     └─ Returns: mda_narrative, confidence_profile│
  │        disappearance_flags, non_gaap_bridge,     │
  │        segment_data, detected_period             │
  │                                                 │
  │ Document Analyses (from ingestion):             │
  │  ├─ _build_document_text('transcript')          │
  │  │  └─ Raw transcript from document_sections     │
  │  ├─ _build_document_text('presentation')        │
  │  │  └─ Raw presentation from document_sections   │
  │  ├─ _load_document_analysis('transcript_analysis')│
  │  │  └─ Pre-built LLM analysis from ingestion     │
  │  └─ _load_document_analysis('presentation_analysis')│
  │     └─ Pre-built LLM analysis from ingestion     │
  │                                                 │
  │ Macro:                                          │
  │  └─ build_context_contract()                    │
  │     └─ Queries context_contracts WHERE is_active │
  │     └─ Returns: rates, usd, credit, growth, etc.│
  └─────────────────────────────────────────────────┘
       │
       │  Returns single dict with ALL fields
       ▼

═══════════════════════════════════════════════════════════════════
 PHASE B — Agent Pipeline (triggered by "Analyse Period")
═══════════════════════════════════════════════════════════════════

  agents/orchestrator.py: AgentOrchestrator
  ┌─────────────────────────────────────────────────┐
  │ Entry: run_document_pipeline(company_id, period) │
  │                                                 │
  │ 1. Check Phase A complete (processing_job done)  │
  │ 2. Create pipeline_run record                    │
  │ 3. Call build_agent_context() → inputs dict      │
  │ 4. AgentRegistry.get_execution_order()           │
  │ 5. Group into layers by dependency depth         │
  │ 6. Execute layer by layer:                       │
  │                                                 │
  │ LAYER 0 ─────────────────────────────────────── │
  │ ┌─────────────────────────────────────────────┐ │
  │ │ Financial Analyst (agents/task/)             │ │
  │ │ Prompt: prompts/agents/financial_analyst.txt │ │
  │ │                                             │ │
  │ │ Receives: extracted_metrics, thesis,         │ │
  │ │   tracked_kpis, guidance, prior_period,      │ │
  │ │   transcript_deep_dive (pre-built),          │ │
  │ │   presentation_analysis (pre-built),         │ │
  │ │   confidence_profile, segment_data,          │ │
  │ │   context_contract                           │ │
  │ │                                             │ │
  │ │ Outputs: overall_grade (1-5),                │ │
  │ │   tracked_kpi_assessment [{kpi,score}],      │ │
  │ │   key_assumptions [{assumption,prob,prior}], │ │
  │ │   thesis_direction, key_surprises,           │ │
  │ │   revenue/margin/cashflow/balance assessments │ │
  │ └─────────────────────────────────────────────┘ │
  │       │                                         │
  │       │  FA output merged into inputs dict       │
  │       ▼                                         │
  │ LAYER 1 (parallel) ─────────────────────────── │
  │ ┌──────────────────────┐ ┌──────────────────┐  │
  │ │ Bear Case            │ │ Bull Case        │  │
  │ │ agents/task/         │ │ agents/task/     │  │
  │ │                      │ │                  │  │
  │ │ Receives: FA output  │ │ Receives: FA out │  │
  │ │ + thesis, segments,  │ │ + thesis, guid., │  │
  │ │ disappearance_flags, │ │ segments, macro  │  │
  │ │ non_gaap_bridge,     │ │ tailwinds        │  │
  │ │ confidence_profile,  │ │                  │  │
  │ │ macro headwinds      │ │ Outputs:         │  │
  │ │                      │ │ bull_thesis,     │  │
  │ │ Outputs:             │ │ upside_catalysts,│  │
  │ │ bear_thesis,         │ │ upside_scenario, │  │
  │ │ key_risks [{risk,    │ │ what_would_make_ │  │
  │ │   prob, impact}],    │ │ you_wrong        │  │
  │ │ downside_scenario,   │ │                  │  │
  │ │ early_warning_signals│ │                  │  │
  │ └──────────────────────┘ └──────────────────┘  │
  │       │                       │                 │
  │       │  Both merged into inputs                │
  │       ▼                                         │
  │ LAYER 2 ─────────────────────────────────────── │
  │ ┌─────────────────────────────────────────────┐ │
  │ │ Debate Agent (agents/meta/)                 │ │
  │ │                                             │ │
  │ │ Receives: FA + bear + bull outputs, thesis, │ │
  │ │   context_contract                           │ │
  │ │                                             │ │
  │ │ Outputs: debate_summary,                     │ │
  │ │   bear_probability + base_probability +      │ │
  │ │   bull_probability (sum to 100),             │ │
  │ │   verdict (buy|hold|watch|avoid),            │ │
  │ │   strongest_bear_arg, strongest_bull_arg,    │ │
  │ │   key_swing_factors                          │ │
  │ └─────────────────────────────────────────────┘ │
  │       │                                         │
  │       ▼                                         │
  │ LAYER 3 ─────────────────────────────────────── │
  │ ┌─────────────────────────────────────────────┐ │
  │ │ Quality Control (agents/meta/)              │ │
  │ │                                             │ │
  │ │ Receives: all_outputs (every agent's JSON), │ │
  │ │   context_contract, thesis                   │ │
  │ │                                             │ │
  │ │ Outputs: per_agent_scores (4-dim rubric),    │ │
  │ │   contract_violations, overall_score (0-1),  │ │
  │ │   recommendation (accept|review|rerun)       │ │
  │ └─────────────────────────────────────────────┘ │
  │                                                 │
  │ 7. Persist all outputs to agent_outputs table    │
  │ 8. Update pipeline_run with status, cost, QC     │
  │                                                 │
  │ Safety:                                         │
  │  ├─ BudgetGuard enforces $2 pipeline cap         │
  │  ├─ Circuit breaker stops on credit errors       │
  │  ├─ 300s timeout per pipeline                    │
  │  └─ Critical agent failure (FA) aborts pipeline  │
  └─────────────────────────────────────────────────┘
       │
       ▼
  UI RENDERING (apps/ui/index.html)
  ┌─────────────────────────────────────────────────┐
  │ Results tab queries:                            │
  │  GET /agent-outputs/{ticker}/{period}           │
  │  GET /pipeline-runs/latest/{ticker}/{period}    │
  │                                                 │
  │ Renders:                                        │
  │  ├─ Pipeline panel (status, cost, agent timeline)│
  │  ├─ FA card (grade badge, KPI scores, Bayesian  │
  │  │   assumption bars with prior→posterior)        │
  │  ├─ Transcript card (tone, guidance, evasion)    │
  │  ├─ Presentation card (priorities, omissions)    │
  │  ├─ Bear/Bull side-by-side (risks vs catalysts)  │
  │  ├─ Debate card (probability bar, verdict)       │
  │  └─ QC card (score, recommendation)              │
  └─────────────────────────────────────────────────┘
```

### Key Database Tables

| Table | Written by | Read by |
|-------|-----------|---------|
| `documents` | Harvester dispatcher | Background processor, context builder |
| `document_sections` | Document parser | Context builder (_build_document_text) |
| `extracted_metrics` | Metric extractor | Context builder (build_kpi_summary) |
| `extraction_profiles` | Background processor | Context builder (confidence, segments) |
| `research_outputs` | Background processor | Context builder (extraction_context, transcript/presentation analysis) |
| `context_contracts` | Macro View UI | Context builder (build_context_contract) |
| `processing_jobs` | Background processor | Phase A check, UI progress |
| `agent_outputs` | Orchestrator | UI rendering (agent cards) |
| `pipeline_runs` | Orchestrator | UI polling (pipeline panel) |
| `harvested_documents` | Harvester dispatcher | Dedup checks |
| `price_records` | Price feed (Yahoo) | Portfolio dashboard, scenario snapshots |
| `scenario_snapshots` | Price feed + manual save | Scenario history chart |

### Key Functions and Where They Live

| Function | File | Purpose |
|----------|------|---------|
| `run_harvest()` | services/harvester/__init__.py | Orchestrate all source scrapers |
| `dispatch_candidates()` | services/harvester/dispatcher.py | Dedup and ingest documents |
| `check_coverage()` | services/harvester/coverage.py | Compare latest vs expected period |
| `process_document()` | services/document_parser.py | Parse PDF/HTML/DOCX → sections |
| `extract_by_document_type()` | services/metric_extractor.py | Route to section-aware or legacy extraction |
| `split_into_sections()` | services/section_splitter.py | Split filing into FilingSections |
| `_analyse_document_with_llm()` | services/background_processor.py | Transcript/presentation LLM analysis |
| `_persist_extraction_profile()` | services/background_processor.py | Save enriched data to DB |
| `build_agent_context()` | services/context_builder.py | Build everything agents need |
| `build_context_contract()` | services/context_builder.py | Load active macro assumptions |
| `AgentOrchestrator.run_document_pipeline()` | agents/orchestrator.py | Execute agent pipeline |
| `AgentRegistry.autodiscover()` | agents/registry.py | Find and register all agents |
| `call_llm_native_async()` | services/llm_client.py | Async LLM call with retry + circuit breaker |
| `refresh_prices()` | services/price_feed.py | Yahoo Finance daily prices |

---

## Part 2: Integration Test Results

### 1. Infrastructure
- [x] Health endpoint: OK (uptime confirmed)
- [x] Auth: login works, blocks unauthenticated
- [x] Web service: online
- [x] Worker service: online (ANTHROPIC_API_KEY, PYTHONPATH=/app set)
- [x] Beat service: online (Monday 00:00 UTC schedule)

### 2. Unit Tests
- [x] 205 passed, 13 expected DB connection errors
- [x] Orchestrator tests: 5/5 pass (phase A block, pipeline, abort, merging, dependency resolution)

### 3. Agent Registry
- [x] 5 agents discovered: financial_analyst, bear_case, bull_case, debate_agent, quality_control
- [x] Execution order correct: FA → Bear+Bull → Debate → QC
- [x] 0 dependency warnings

### 4. Context Builder
- [x] build_agent_context returns all fields
- [x] transcript_text: 30,000 chars loaded from document_sections
- [x] presentation_text: 30,000 chars loaded from document_sections
- [x] context_contract loaded (version 1, 8 macro assumptions)
- [x] Prompt template variables resolve correctly (no {regime} errors)

### 5. Agent Pipeline
- [x] Pipeline completes: 4 agents completed, $0.05 total cost, ~46 seconds
- [x] FA: grade, thesis_direction, key_assumptions produced
- [x] Bear/Bull: one-sided cases with ranked risks/catalysts
- [x] Debate: probability split, verdict, swing factors
- [x] QC: skipped (wiring issue — runs in wrong layer when no debate output yet)
- [x] Outputs stored in agent_outputs table
- [x] Pipeline run stored in pipeline_runs table

---

## Part 3: Known Bugs

### Bug 1: 8-K documents cannot be removed
**Symptom:** Clicking "Remove" on an 8-K in the Documents tab fails silently.
**Cause:** FK constraint from `harvested_documents.document_id` → `documents.id`. The harvested record must be cleared first.
**Fix needed:** Delete cascade or clear harvested_documents reference before deleting document.
**Priority:** Medium

### Bug 2: QC agent skipped
**Symptom:** QC agent always shows "skipped" in pipeline results.
**Cause:** QC's `should_run()` checks `inputs.get("all_outputs")` — but `all_outputs` is only populated AFTER each layer. QC runs in the same layer as Debate (both META tier), so it doesn't see Debate's output.
**Fix needed:** The orchestrator already injects `all_outputs` after each layer. QC should run in a separate layer after Debate. Either change QC to depend on debate_agent, or add special handling in the orchestrator.
**Priority:** High (QC is the quality gate)

### Bug 3: Phase A race condition (FIXED)
**Symptom:** Agents ran before extraction completed, producing "DATA MISSING" output.
**Cause:** Phase A check looked for document sections (which existed from previous runs) instead of a completed processing job.
**Fix:** Now checks `processing_jobs.status = 'completed'`. Deployed in commit 88cfde5.
**Priority:** Fixed

### Bug 4: LLM credit exhaustion burned $9 (FIXED)
**Symptom:** 188 failed API calls when credits ran out, each still costing money.
**Cause:** No circuit breaker — parallel extraction calls all hit the API independently.
**Fix:** Circuit breaker trips on first credit/billing error, all subsequent calls fail instantly with no API call. Deployed in commit 88cfde5.
**Priority:** Fixed

### Bug 5: Deploys kill running extraction
**Symptom:** Extraction jobs stuck at 10% forever after a code push.
**Cause:** Background tasks run via `asyncio.create_task` on the web service. Deploys restart the web process, killing the task.
**Fix needed:** Move extraction to Celery worker (like the agent pipeline), or add job recovery that detects stuck jobs and re-queues them.
**Priority:** High (blocks testing)

### Bug 6: 10-Q extraction produces fewer metrics after DB rebuild
**Symptom:** Original extraction got 2,011 metrics. After DB rebuild + multiple failed runs, only got 68 (non-GAAP bridge items only).
**Cause:** Credit exhaustion during extraction caused most section LLM calls to fail. Only the non-GAAP bridge calls succeeded (they ran after credits were replenished).
**Fix:** Top up credits, clean the period, and re-extract. The extraction code itself is unchanged.
**Priority:** Resolved (not a code bug)

### Bug 7: Pipeline polling loops indefinitely
**Symptom:** UI polls /pipeline-runs/latest every 3 seconds even after pipeline completes. Floods logs with GET requests.
**Cause:** The polling function `pollPipelineStatus` checks for 'completed' status but finds old completed runs from previous attempts.
**Fix needed:** Poll should stop after finding a completed run, or track the specific pipeline_run_id it's waiting for.
**Priority:** Low (cosmetic, wastes bandwidth)

---

## Part 4: Recommended Next Steps

### Immediate (before next test)
1. Fix QC agent layer ordering (make it depend on debate_agent)
2. Top up Anthropic credits if low
3. Clean ALLY Q2 period and re-extract with full credits

### Short-term
4. Move document processing to Celery worker (prevents deploy kills)
5. Fix 8-K removal (cascade delete or clear FK)
6. Add job recovery for stuck processing jobs
7. Write remaining agent prompt refinements based on test output

### Medium-term
8. UI: render old synthesis output alongside agent cards (backward compat during transition)
9. Add per-document metric persistence (don't wait for batch to finish)
10. Add cost tracking dashboard (cumulative API spend per company/period)
