"""
Prompt Experimentation — A/B testing, variant management, and LLM-driven refinement.
"""

import json
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, PromptVariant, ABExperiment

router = APIRouter(tags=["experiments"])

AUTO_PROMOTE_THRESHOLD = 5


# ── Schemas ──────────────────────────────────────────────────
class VariantCreate(BaseModel):
    prompt_type: str
    variant_name: str
    prompt_text: str
    is_active: bool = False
    is_candidate: bool = True
    notes: Optional[str] = None


class VariantUpdate(BaseModel):
    prompt_text: Optional[str] = None
    is_candidate: Optional[bool] = None
    is_active: Optional[bool] = None
    notes: Optional[str] = None


class ExperimentCreate(BaseModel):
    prompt_type: str
    ticker: Optional[str] = None
    period_label: Optional[str] = None
    input_text: str


class ExperimentJudge(BaseModel):
    winner: str
    rating_a: Optional[int] = None
    rating_b: Optional[int] = None
    feedback: Optional[str] = None


class RefineRequest(BaseModel):
    prompt_type: str


# ── Smart prompt formatter ────────────────────────────────────
def _format_prompt(prompt_text: str, input_text: str, ticker: str = "", period: str = "") -> str:
    """
    Safely format any prompt template for A/B testing regardless of placeholders.
    Extraction prompts use {text}. Briefing/synthesis prompts use {company}, {kpis} etc.
    We fill every placeholder found with sensible test values.
    """
    placeholders = set(re.findall(r'\{(\w+)\}', prompt_text))

    # Double-braces used for JSON schema examples — leave them alone
    # We only act on single-brace placeholders
    fill = {}
    raw_input = input_text[:15000]

    for p in placeholders:
        if p in ('text', 'kpis', 'earnings_data', 'transcript_data', 'broker_data',
                 'presentation_data', 'quarter_data', 'prior_data', 'current_metrics',
                 'prior_metrics', 'document_summary', 'prior_analysis', 'findings',
                 'expectations', 'actuals', 'actual_results', 'statements', 'context',
                 'source_text'):
            fill[p] = raw_input
        elif p in ('thesis', 'thesis_comparison', 'thesis_risks', 'tracked_kpis'):
            fill[p] = '[Paste thesis text here for a better test — or leave as-is to test formatting only]'
        elif p == 'surprises':
            fill[p] = '[No surprises data — run surprise detection separately]'
        elif p == 'company':
            fill[p] = ticker or 'Test Company'
        elif p == 'ticker':
            fill[p] = ticker or 'TEST'
        elif p in ('period', 'current_period'):
            fill[p] = period or 'Test Period'
        elif p == 'prior_period':
            fill[p] = 'Prior Period'
        elif p == 'sector':
            fill[p] = 'Unknown sector'
        else:
            fill[p] = f'[{p}]'

    try:
        return prompt_text.format(**fill)
    except (KeyError, ValueError):
        # Last resort — strip all remaining {placeholders}
        return re.sub(r'\{(\w+)\}', lambda m: fill.get(m.group(1), f'[{m.group(1)}]'), prompt_text)


# ── Variant CRUD ─────────────────────────────────────────────
@router.get("/experiments/variants")
async def list_variants(prompt_type: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    q = select(PromptVariant).order_by(PromptVariant.prompt_type, PromptVariant.win_count.desc())
    if prompt_type:
        q = q.where(PromptVariant.prompt_type == prompt_type)
    result = await db.execute(q)
    return [{
        "id": str(v.id), "prompt_type": v.prompt_type, "variant_name": v.variant_name,
        "is_active": v.is_active, "is_candidate": v.is_candidate,
        "win_count": v.win_count, "loss_count": v.loss_count, "total_runs": v.total_runs,
        "avg_rating": float(v.avg_rating) if v.avg_rating else 0,
        "generation": v.generation, "notes": v.notes,
        "win_rate": round(v.win_count / max(v.win_count + v.loss_count, 1) * 100, 1),
        "prompt_preview": (v.prompt_text[:200] + "…") if len(v.prompt_text) > 200 else v.prompt_text,
        "created_at": v.created_at.isoformat() if v.created_at else None,
    } for v in result.scalars().all()]


@router.post("/experiments/variants", status_code=201)
async def create_variant(body: VariantCreate, db: AsyncSession = Depends(get_db)):
    if body.is_active:
        existing = await db.execute(
            select(PromptVariant).where(PromptVariant.prompt_type == body.prompt_type, PromptVariant.is_active == True)
        )
        for v in existing.scalars().all():
            v.is_active = False

    variant = PromptVariant(
        id=uuid.uuid4(), prompt_type=body.prompt_type, variant_name=body.variant_name,
        prompt_text=body.prompt_text, is_active=body.is_active,
        is_candidate=body.is_candidate, notes=body.notes,
    )
    db.add(variant)
    await db.commit()
    return {"id": str(variant.id), "variant_name": variant.variant_name}


@router.get("/experiments/variants/{variant_id}")
async def get_variant(variant_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(PromptVariant).where(PromptVariant.id == variant_id))
    v = result.scalar_one_or_none()
    if not v:
        raise HTTPException(404, "Variant not found")
    return {
        "id": str(v.id), "prompt_type": v.prompt_type, "variant_name": v.variant_name,
        "prompt_text": v.prompt_text, "is_active": v.is_active, "is_candidate": v.is_candidate,
        "win_count": v.win_count, "loss_count": v.loss_count,
        "total_runs": v.total_runs, "avg_rating": float(v.avg_rating) if v.avg_rating else 0,
        "generation": v.generation, "notes": v.notes,
    }


@router.patch("/experiments/variants/{variant_id}")
async def update_variant(variant_id: uuid.UUID, body: VariantUpdate, db: AsyncSession = Depends(get_db)):
    """Update prompt text, candidate flag, or notes on an existing variant."""
    result = await db.execute(select(PromptVariant).where(PromptVariant.id == variant_id))
    v = result.scalar_one_or_none()
    if not v:
        raise HTTPException(404, "Variant not found")
    if body.prompt_text is not None:
        v.prompt_text = body.prompt_text
    if body.is_candidate is not None:
        v.is_candidate = body.is_candidate
    if body.notes is not None:
        v.notes = body.notes
    if body.is_active is not None:
        if body.is_active:
            others = await db.execute(
                select(PromptVariant).where(PromptVariant.prompt_type == v.prompt_type, PromptVariant.is_active == True)
            )
            for other in others.scalars().all():
                other.is_active = False
        v.is_active = body.is_active
    await db.commit()
    return {"status": "updated", "id": str(v.id), "variant_name": v.variant_name}


@router.post("/experiments/variants/{variant_id}/activate")
async def activate_variant(variant_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(PromptVariant).where(PromptVariant.id == variant_id))
    v = result.scalar_one_or_none()
    if not v:
        raise HTTPException(404, "Variant not found")
    others = await db.execute(
        select(PromptVariant).where(PromptVariant.prompt_type == v.prompt_type, PromptVariant.is_active == True)
    )
    for other in others.scalars().all():
        other.is_active = False
    v.is_active = True
    await db.commit()
    return {"status": "activated", "variant_name": v.variant_name}


# ── Run Experiment ───────────────────────────────────────────
@router.post("/experiments/run")
async def run_experiment(body: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    from services.llm_client import call_llm_json_async
    import asyncio

    active_q = await db.execute(
        select(PromptVariant).where(
            PromptVariant.prompt_type == body.prompt_type, PromptVariant.is_active == True
        ).limit(1)
    )
    active = active_q.scalar_one_or_none()

    candidates_q = await db.execute(
        select(PromptVariant).where(
            PromptVariant.prompt_type == body.prompt_type,
            PromptVariant.is_candidate == True,
            PromptVariant.is_active == False,
        ).order_by(func.random()).limit(1)
    )
    candidate = candidates_q.scalar_one_or_none()

    if not active:
        raise HTTPException(400, f"No active variant for '{body.prompt_type}'. Click 'Seed Default Variants' first.")
    if not candidate:
        raise HTTPException(400, f"No candidate variants for '{body.prompt_type}'. The 3 new variants should have is_candidate=true — check Prompt Lab.")

    company_id = None
    ticker = body.ticker or ""
    period = body.period_label or ""
    if body.ticker:
        co_q = await db.execute(select(Company).where(Company.ticker == body.ticker.upper()))
        co = co_q.scalar_one_or_none()
        if co:
            company_id = co.id
            ticker = co.ticker

    # Smart format — works for both extraction ({text}) and briefing ({company}, {kpis} etc)
    prompt_a = _format_prompt(active.prompt_text, body.input_text, ticker, period)
    prompt_b = _format_prompt(candidate.prompt_text, body.input_text, ticker, period)

    try:
        result_a, result_b = await asyncio.gather(
            call_llm_json_async(prompt_a, max_tokens=8192, feature="experiment", ticker=ticker),
            call_llm_json_async(prompt_b, max_tokens=8192, feature="experiment", ticker=ticker),
            return_exceptions=True,
        )
    except Exception as e:
        raise HTTPException(500, f"LLM calls failed: {str(e)[:200]}")

    output_a = json.dumps(result_a, default=str) if not isinstance(result_a, Exception) else json.dumps({"error": str(result_a)[:200]})
    output_b = json.dumps(result_b, default=str) if not isinstance(result_b, Exception) else json.dumps({"error": str(result_b)[:200]})

    active.total_runs += 1
    candidate.total_runs += 1

    experiment = ABExperiment(
        id=uuid.uuid4(), company_id=company_id, prompt_type=body.prompt_type,
        period_label=body.period_label, variant_a_id=active.id, variant_b_id=candidate.id,
        output_a=output_a, output_b=output_b, status="pending",
    )
    db.add(experiment)
    await db.commit()

    return {
        "experiment_id": str(experiment.id),
        "variant_a": {"id": str(active.id), "name": active.variant_name, "generation": active.generation, "is_active": True},
        "variant_b": {"id": str(candidate.id), "name": candidate.variant_name, "generation": candidate.generation, "is_active": False},
        "output_a": result_a if not isinstance(result_a, Exception) else {"error": str(result_a)[:200]},
        "output_b": result_b if not isinstance(result_b, Exception) else {"error": str(result_b)[:200]},
    }


# ── Judge Experiment ─────────────────────────────────────────
@router.post("/experiments/{experiment_id}/judge")
async def judge_experiment(experiment_id: uuid.UUID, body: ExperimentJudge, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ABExperiment).where(ABExperiment.id == experiment_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    exp.winner = body.winner
    exp.rating_a = body.rating_a
    exp.rating_b = body.rating_b
    exp.analyst_feedback = body.feedback
    exp.status = "completed"

    # Update variant stats
    va_q = await db.execute(select(PromptVariant).where(PromptVariant.id == exp.variant_a_id))
    va = va_q.scalar_one_or_none()
    vb_q = await db.execute(select(PromptVariant).where(PromptVariant.id == exp.variant_b_id))
    vb = vb_q.scalar_one_or_none()

    if va and vb:
        if body.winner == "a":
            va.win_count += 1; vb.loss_count += 1
        elif body.winner == "b":
            vb.win_count += 1; va.loss_count += 1

        if body.rating_a:
            total_a = float(va.avg_rating or 0) * max(va.total_runs - 1, 0)
            va.avg_rating = (total_a + body.rating_a) / va.total_runs if va.total_runs else body.rating_a
        if body.rating_b:
            total_b = float(vb.avg_rating or 0) * max(vb.total_runs - 1, 0)
            vb.avg_rating = (total_b + body.rating_b) / vb.total_runs if vb.total_runs else body.rating_b

    await db.commit()

    # Auto-promote check
    auto_promoted = False
    message = "Judged."
    winner_variant = va if body.winner == "a" else (vb if body.winner == "b" else None)
    if winner_variant and not winner_variant.is_active and winner_variant.win_count >= AUTO_PROMOTE_THRESHOLD:
        others = await db.execute(
            select(PromptVariant).where(
                PromptVariant.prompt_type == winner_variant.prompt_type, PromptVariant.is_active == True
            )
        )
        for other in others.scalars().all():
            other.is_active = False
        winner_variant.is_active = True
        await db.commit()
        auto_promoted = True
        message = f"Auto-promoted {winner_variant.variant_name} as new active variant!"

    return {"status": "judged", "winner": body.winner, "auto_promoted": auto_promoted, "message": message}


# ── List Experiments ─────────────────────────────────────────
@router.get("/experiments")
async def list_experiments(prompt_type: Optional[str] = None, status: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    q = select(ABExperiment).order_by(ABExperiment.created_at.desc()).limit(50)
    if prompt_type:
        q = q.where(ABExperiment.prompt_type == prompt_type)
    if status:
        q = q.where(ABExperiment.status == status)
    result = await db.execute(q)
    return [{
        "id": str(e.id), "prompt_type": e.prompt_type, "period_label": e.period_label,
        "winner": e.winner, "rating_a": e.rating_a, "rating_b": e.rating_b,
        "feedback": e.analyst_feedback, "status": e.status,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    } for e in result.scalars().all()]


# ── LLM-Driven Refinement ───────────────────────────────────
@router.post("/experiments/refine")
async def refine_prompts(body: RefineRequest, db: AsyncSession = Depends(get_db)):
    from services.llm_client import call_llm

    active_q = await db.execute(
        select(PromptVariant).where(
            PromptVariant.prompt_type == body.prompt_type, PromptVariant.is_active == True
        ).limit(1)
    )
    active = active_q.scalar_one_or_none()
    if not active:
        raise HTTPException(400, "No active variant for this type")

    exps_q = await db.execute(
        select(ABExperiment).where(
            ABExperiment.prompt_type == body.prompt_type,
            ABExperiment.status == "completed",
            ABExperiment.analyst_feedback != None,
        ).order_by(ABExperiment.created_at.desc()).limit(20)
    )
    experiments = exps_q.scalars().all()

    if len(experiments) < 1:
        raise HTTPException(400, f"No judged experiments with feedback found for '{body.prompt_type}'. Run at least one experiment and submit a judgement first.")

    best_q = await db.execute(
        select(PromptVariant).where(PromptVariant.prompt_type == body.prompt_type)
        .order_by(PromptVariant.avg_rating.desc()).limit(1)
    )
    best = best_q.scalar_one_or_none()

    feedback_lines = []
    for exp in experiments:
        winner_label = "A (active)" if exp.winner == "a" else "B (challenger)" if exp.winner == "b" else "Tie"
        feedback_lines.append(
            f"Winner: {winner_label} | Rating A: {exp.rating_a}/5 | Rating B: {exp.rating_b}/5 | Feedback: {exp.analyst_feedback}"
        )

    refinement_prompt = f"""You are a prompt engineering expert for a buy-side investment research platform.
Your task is to improve a prompt based on analyst feedback.

CURRENT ACTIVE PROMPT ({active.variant_name}, generation {active.generation}):
---
{active.prompt_text}
---

BEST-PERFORMING VARIANT ({best.variant_name if best else 'N/A'}):
---
{best.prompt_text if best and best.id != active.id else 'Same as active.'}
---

ANALYST FEEDBACK FROM {len(experiments)} EXPERIMENTS:
{chr(10).join(feedback_lines)}

INSTRUCTIONS:
1. Analyse the feedback — what do analysts consistently prefer?
2. Identify specific weaknesses in the current prompt
3. Generate an IMPROVED prompt that addresses the feedback
4. Keep the same placeholder variables (e.g. {{company}}, {{kpis}}, {{text}})
5. Keep the same output JSON schema — only improve instructions/framing

Return ONLY the improved prompt text. No explanation, no markdown fences."""

    try:
        new_prompt_text = call_llm(refinement_prompt, max_tokens=8192, feature="prompt_refine")

        new_variant = PromptVariant(
            id=uuid.uuid4(),
            prompt_type=body.prompt_type,
            variant_name=f"v{active.generation + 1}_llm_refined",
            prompt_text=new_prompt_text,
            is_active=False,
            is_candidate=True,
            parent_variant_id=active.id,
            generation=active.generation + 1,
            notes=f"LLM-refined from {active.variant_name} based on {len(experiments)} experiments.",
        )
        db.add(new_variant)
        await db.commit()

        return {
            "status": "refined",
            "new_variant_id": str(new_variant.id),
            "new_variant_name": new_variant.variant_name,
            "generation": new_variant.generation,
            "based_on_experiments": len(experiments),
            "prompt_preview": new_prompt_text[:500] + "…" if len(new_prompt_text) > 500 else new_prompt_text,
        }
    except Exception as e:
        raise HTTPException(500, f"Refinement failed: {str(e)[:200]}")


# ── Seed initial variants ─────────────────────────────────────
@router.post("/experiments/seed")
async def seed_variants(db: AsyncSession = Depends(get_db)):
    from prompts import (
        COMBINED_EXTRACTOR, EARNINGS_RELEASE_EXTRACTOR, TRANSCRIPT_EXTRACTOR,
        BROKER_NOTE_EXTRACTOR, PRESENTATION_EXTRACTOR,
        ONE_PAGE_BRIEFING, SYNTHESIS_BRIEFING,
        SURPRISE_DETECTOR, IR_QUESTION_GENERATOR,
    )
    from services.thesis_comparator import THESIS_COMPARATOR_V2

    seeds = [
        ("extraction_combined", "v1_default", COMBINED_EXTRACTOR),
        ("extraction_earnings", "v1_default", EARNINGS_RELEASE_EXTRACTOR),
        ("extraction_transcript", "v1_default", TRANSCRIPT_EXTRACTOR),
        ("extraction_broker", "v1_default", BROKER_NOTE_EXTRACTOR),
        ("extraction_presentation", "v1_default", PRESENTATION_EXTRACTOR),
        ("synthesis", "v1_default", SYNTHESIS_BRIEFING),
        ("briefing", "v1_default", ONE_PAGE_BRIEFING),
        ("thesis_comparison", "v1_default", THESIS_COMPARATOR_V2),
        ("surprise", "v1_default", SURPRISE_DETECTOR),
        ("ir_questions", "v1_default", IR_QUESTION_GENERATOR),
    ]

    created = []
    for ptype, name, text in seeds:
        existing = await db.execute(
            select(PromptVariant).where(PromptVariant.prompt_type == ptype, PromptVariant.variant_name == name)
        )
        if existing.scalar_one_or_none():
            continue
        v = PromptVariant(
            id=uuid.uuid4(), prompt_type=ptype, variant_name=name,
            prompt_text=text, is_active=True, is_candidate=False,
            notes="Seeded from codebase defaults",
        )
        db.add(v)
        created.append(ptype)

    await db.commit()
    return {"seeded": created, "message": f"Created {len(created)} default variants"}


# ── Stats ─────────────────────────────────────────────────────
@router.get("/experiments/stats")
async def experiment_stats(db: AsyncSession = Depends(get_db)):
    variants_q = await db.execute(
        select(PromptVariant.prompt_type, func.count(PromptVariant.id))
        .group_by(PromptVariant.prompt_type)
    )
    variant_counts = dict(variants_q.all())

    exps_q = await db.execute(
        select(ABExperiment.prompt_type, ABExperiment.status, func.count(ABExperiment.id))
        .group_by(ABExperiment.prompt_type, ABExperiment.status)
    )
    exp_counts = {}
    for row in exps_q.all():
        if row[0] not in exp_counts:
            exp_counts[row[0]] = {}
        exp_counts[row[0]][row[1]] = row[2]

    return {
        "variants_by_type": variant_counts,
        "experiments_by_type": exp_counts,
        "auto_promote_threshold": AUTO_PROMOTE_THRESHOLD,
    }
