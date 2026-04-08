# Extraction Pipeline Upgrade — Claude Code Instructions

## Overview
These files implement 5 improvements to the document extraction pipeline:
1. **Section-level splitting** for 10-K/10-Q/earnings releases (MD&A, financials, notes, risk factors)
2. **Sector-specific KPI dictionaries** injected into prompts dynamically
3. **Tiered model routing** (Haiku for tables, Sonnet for extraction, Opus for synthesis)
4. **Period validation agent** — post-processor to catch period mislabelling
5. **Segment decomposition agent** — dedicated revenue/profit tree extraction

## Files to Add/Replace

### NEW FILES (copy to repo as-is):
```
services/section_splitter.py      — Section-level document splitting
services/sector_kpi_config.py     — Sector KPI dictionaries + context injection
services/period_validator.py      — Period disambiguation post-processor
services/segment_extractor.py     — Segment decomposition agent
services/qualifier_extractor.py   — Qualifier language detection (hedging, one-offs, confidence)
services/disappeared_detector.py  — Disappeared metrics & silent guidance drops
services/non_gaap_tracker.py      — Non-GAAP bridge extraction & cross-period comparison
prompts/section_prompts.py        — Section-specific extraction prompts
```

### MODIFIED FILES (replace existing):
```
configs/settings.py               — Added model tier settings
services/llm_client.py            — Added call_llm_with_model for tier routing
services/metric_extractor.py      — Full rewrite integrating all improvements
```

## Git Commands (PowerShell)

```powershell
# Navigate to your repo
cd C:\path\to\AI-tracker-tool

# Create a feature branch
git checkout -b feature/extraction-pipeline-v2

# Copy the new files into place (adjust source path as needed)
# If you downloaded the files to e.g. C:\Downloads\extraction-upgrade\:

# NEW files
Copy-Item "services\section_splitter.py"   -Destination "services\section_splitter.py"
Copy-Item "services\sector_kpi_config.py"  -Destination "services\sector_kpi_config.py"
Copy-Item "services\period_validator.py"   -Destination "services\period_validator.py"
Copy-Item "services\segment_extractor.py"  -Destination "services\segment_extractor.py"
Copy-Item "services\qualifier_extractor.py" -Destination "services\qualifier_extractor.py"
Copy-Item "services\disappeared_detector.py" -Destination "services\disappeared_detector.py"
Copy-Item "services\non_gaap_tracker.py"   -Destination "services\non_gaap_tracker.py"
Copy-Item "prompts\section_prompts.py"     -Destination "prompts\section_prompts.py"

# MODIFIED files (overwrite existing)
Copy-Item "configs\settings.py"            -Destination "configs\settings.py" -Force
Copy-Item "services\llm_client.py"         -Destination "services\llm_client.py" -Force
Copy-Item "services\metric_extractor.py"   -Destination "services\metric_extractor.py" -Force

# Stage all changes
git add -A

# Commit
git commit -m "feat: extraction pipeline v2 — section splitting, model tiers, sector KPIs, period validation, segment decomposition, qualifier analysis, disappearance detection, non-GAAP bridge tracking"

# Push
git push -u origin feature/extraction-pipeline-v2

# Then open a PR on GitHub, or merge directly:
# git checkout main
# git merge feature/extraction-pipeline-v2
# git push origin main
```

## Environment Variables to Add (.env)

```env
# Model tiers (add these to your .env)
LLM_MODEL_FAST=claude-haiku-4-5-20251001
LLM_MODEL=claude-sonnet-4-20250514
LLM_MODEL_ADVANCED=claude-sonnet-4-20250514
```

## What Each File Does

### `services/section_splitter.py`
Regex + heuristic splitter that identifies Financial Statements, MD&A, Notes,
Risk Factors, and Guidance sections in 10-K/10-Q/earnings releases. Tags each
section with a model tier (fast/default) and token budget.

### `services/sector_kpi_config.py`
Contains all subsector KPI dictionaries from your agent plan: Banks (NII, NIM,
CET1, NPL...), Insurance (combined ratio, solvency...), Beverages (organic growth,
price/mix...), Retail (LFL, basket size...), etc. Also has country-level GAAP/currency
context. Injected into prompts via `get_sector_context()`.

### `services/period_validator.py`
Detects the document's reporting period from text headers, then validates every
extracted metric's period label. Fixes missing periods, flags mismatches with
reduced confidence. Runs after extraction, before dedup.

### `services/segment_extractor.py`
Dedicated segment decomposition agent. Builds Group → Division → Geography
revenue/profit tree. Includes a sum-check (do segments add to group total?).
Converts output into standard metric items for the main pipeline.

### `prompts/section_prompts.py`
Section-specific extraction prompts (financial statements, MD&A, notes, risk
factors, segment decomposition). Each prompt is tuned for what that section
actually contains. Includes `{sector_context}` placeholder.

### `services/qualifier_extractor.py`
Regex-based post-processor that enriches every extracted item with qualifier
metadata: hedging words ("approximately", "broadly in line"), one-off
attributions ("restructuring", "transformation costs"), temporal qualifiers
("in the near term"), and management confidence signals ("confident" vs
"hope"). Produces a document-level confidence profile (hedge rate, one-off
rate, overall signal). Adjusts metric confidence scores based on hedging.

### `services/disappeared_detector.py`
Queries the database for prior period metrics and compares against current
extraction to detect: disappeared KPIs (reported before, absent now), silent
guidance drops (guided before, no update now), and guidance range narrowing/
widening. Flags tracked KPIs that disappear as high severity. Produces a
human-readable summary of all disappearance signals.

### `services/non_gaap_tracker.py`
Dedicated LLM pass that extracts the full GAAP-to-adjusted reconciliation
bridge (reported profit → adjustments → adjusted profit). Stores each
adjustment label and amount in the database for cross-period comparison.
Detects: new adjustments not in prior period, gap widening (GAAP vs adjusted
diverging), and recurring "one-offs" (same restructuring charge every quarter).
Particularly important for European industrials and consumer names.

### `services/llm_client.py` (modified)
Adds `tier` parameter to `call_llm()` and `call_llm_json_async()`. Routes
TIER_FAST → Haiku, TIER_DEFAULT → Sonnet, TIER_ADVANCED → Sonnet/Opus.

### `configs/settings.py` (modified)
Adds `llm_model_fast` and `llm_model_advanced` settings alongside existing
`llm_model`.

### `services/metric_extractor.py` (modified)
Complete rewrite of `extract_by_document_type()`. For structured filings:
section split → parallel section extraction → segment decomposition → period
validation → normalise/dedup → validate. For transcripts/broker notes: legacy
pipeline with sector context injection.

## Testing

After deploying, test with:
1. Upload a 10-K → verify sections are split (check logs for "Section splitter:" entries)
2. Upload an earnings release for a bank → verify sector KPIs (NIM, CET1) appear
3. Check API cost logs → Haiku calls should appear for table extraction
4. Check extracted metrics → period labels should match document's stated period
5. Upload a Heineken earnings release → verify beverage KPIs (organic growth, price/mix) are prioritised
6. Check segment decomposition output → verify segments sum to group total
7. Check extraction result for `confidence_profile` → should show hedge_rate, one_off_rate, overall_signal
8. Upload a second period for the same company → `disappearance_flags` should show any dropped KPIs/guidance
9. For companies with adjusted/underlying figures → `non_gaap_bridge` should contain the full reconciliation
10. Upload the same company's next quarter → `non_gaap_comparison` should flag new adjustments or gap widening

## Integration with Agent Plan

This extraction pipeline upgrade is a prerequisite for the Phase 3 subsector agents.
The sector KPI dictionaries in `sector_kpi_config.py` align exactly with the
subsector agent focus areas defined in your implementation plan:

- Banks Agent → uses `banks` KPI config
- Insurance Agent → uses `insurance` KPI config
- Consumer Agent → uses `beverages`, `food_staples`, `tobacco`, `retail` configs
- etc.

When you build the subsector agents in Phase 3, they'll consume the sector-enriched
extraction output from this pipeline.
