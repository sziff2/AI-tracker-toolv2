"""
Prompt Experimentation — A/B testing, variant management, and LLM-driven refinement.

Workflow:
  1. Seed initial prompt variants (or import from prompts/__init__.py)
  2. Run experiments: same input → two variants → side-by-side output
  3. Analyst picks winner → stats updated
  4. After N experiments, auto-promote winner as default
  5. Periodically, LLM reviews feedback and generates improved variants
"""

import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, PromptVariant, ABExperiment

router = APIRouter(tags=["experiments"])

AUTO_PROMOTE_THRESHOLD = 5  # wins needed to auto-promote


# ── Schemas ──────────────────────────────────────────────────
class VariantCreate(BaseModel):
    prompt_type: str
    variant_name: str
    prompt_text: str
    is_active: bool = False
    notes: Optional[str] = None


class ExperimentCreate(BaseModel):
    prompt_type: str
    ticker: Optional[str] = None
    period_label: Optional[str] = None
    input_text: str  # the document text or context to test against


class ExperimentJudge(BaseModel):
    winner: str          # "a" | "b" | "tie"
    rating_a: Optional[int] = None  # 1-5
    rating_b: Optional[int] = None  # 1-5
    feedback: Optional[str] = None


class RefineRequest(BaseModel):
    prompt_type: str


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
    # If setting as active, deactivate others of same type
    if body.is_active:
        existing = await db.execute(
            select(PromptVariant).where(PromptVariant.prompt_type == body.prompt_type, PromptVariant.is_active == True)
        )
        for v in existing.scalars().all():
            v.is_active = False

    variant = PromptVariant(
        id=uuid.uuid4(), prompt_type=body.prompt_type, variant_name=body.variant_name,
        prompt_text=body.prompt_text, is_active=body.is_active, notes=body.notes,
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
        "prompt_text": v.prompt_text, "is_active": v.is_active,
        "win_count": v.win_count, "loss_count": v.loss_count,
        "total_runs": v.total_runs, "avg_rating": float(v.avg_rating) if v.avg_rating else 0,
        "generation": v.generation, "notes": v.notes,
    }


@router.post("/experiments/variants/{variant_id}/activate")
async def activate_variant(variant_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(PromptVariant).where(PromptVariant.id == variant_id))
    v = result.scalar_one_or_none()
    if not v:
        raise HTTPException(404, "Variant not found")
    # Deactivate others of same type
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
    """
    Run the same input through two prompt variants and return both outputs
    for side-by-side comparison.
    """
    from services.llm_client import call_llm_json_async

    # Find two variants: the active one + a random candidate
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
        raise HTTPException(400, f"No active variant for prompt type '{body.prompt_type}'. Seed variants first.")
    if not candidate:
        raise HTTPException(400, f"No candidate variants to test against for '{body.prompt_type}'. Create a variant first.")

    # Look up company if provided
    company_id = None
    if body.ticker:
        co_q = await db.execute(select(Company).where(Company.ticker == body.ticker.upper()))
        co = co_q.scalar_one_or_none()
        if co:
            company_id = co.id

    # Run both prompts in parallel
    import asyncio
    prompt_a = active.prompt_text.format(text=body.input_text[:15000])
    prompt_b = candidate.prompt_text.format(text=body.input_text[:15000])

    try:
        result_a, result_b = await asyncio.gather(
            call_llm_json_async(prompt_a, max_tokens=8192),
            call_llm_json_async(prompt_b, max_tokens=8192),
            return_exceptions=True,
        )
    except Exception as e:
        raise HTTPException(500, f"LLM calls failed: {str(e)[:200]}")

    output_a = json.dumps(result_a, default=str) if not isinstance(result_a, Exception) else json.dumps({"error": str(result_a)[:200]})
    output_b = json.dumps(result_b, default=str) if not isinstance(result_b, Exception) else json.dumps({"error": str(result_b)[:200]})

    # Update run counts
    active.total_runs += 1
    candidate.total_runs += 1

    # Create experiment record
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
    """
    Record which variant the analyst preferred.
    Auto-promotes winner if it crosses the threshold.
    """
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
    vb_q = await db.execute(select(PromptVariant).where(PromptVariant.id == exp.variant_b_id))
    va = va_q.scalar_one_or_none()
    vb = vb_q.scalar_one_or_none()

    if va and vb:
        if body.winner == "a":
            va.win_count += 1
            vb.loss_count += 1
        elif body.winner == "b":
            vb.win_count += 1
            va.loss_count += 1

        # Update average ratings
        if body.rating_a and va:
            total = va.win_count + va.loss_count
            va.avg_rating = ((float(va.avg_rating or 0) * (total - 1)) + body.rating_a) / total if total > 0 else body.rating_a
        if body.rating_b and vb:
            total = vb.win_count + vb.loss_count
            vb.avg_rating = ((float(vb.avg_rating or 0) * (total - 1)) + body.rating_b) / total if total > 0 else body.rating_b

        # Auto-promote: if challenger wins enough, it becomes the default
        auto_promoted = None
        if body.winner == "b" and vb.win_count >= AUTO_PROMOTE_THRESHOLD and not vb.is_active:
            # Check win rate is > 60%
            total_b = vb.win_count + vb.loss_count
            if total_b > 0 and (vb.win_count / total_b) > 0.6:
                va.is_active = False
                vb.is_active = True
                auto_promoted = vb.variant_name

    await db.commit()

    return {
        "status": "judged",
        "winner": body.winner,
        "auto_promoted": auto_promoted,
        "message": f"Variant '{auto_promoted}' auto-promoted as new default!" if auto_promoted else None,
    }


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
    """
    Review all experiment feedback for a prompt type and use the LLM
    to generate an improved variant based on what analysts preferred.
    """
    from services.llm_client import call_llm

    # Get the current active variant
    active_q = await db.execute(
        select(PromptVariant).where(
            PromptVariant.prompt_type == body.prompt_type, PromptVariant.is_active == True
        ).limit(1)
    )
    active = active_q.scalar_one_or_none()
    if not active:
        raise HTTPException(400, "No active variant for this type")

    # Get all completed experiments with feedback
    exps_q = await db.execute(
        select(ABExperiment).where(
            ABExperiment.prompt_type == body.prompt_type,
            ABExperiment.status == "completed",
            ABExperiment.analyst_feedback != None,
        ).order_by(ABExperiment.created_at.desc()).limit(20)
    )
    experiments = exps_q.scalars().all()

    if len(experiments) < 3:
        raise HTTPException(400, f"Need at least 3 judged experiments with feedback to refine. Currently have {len(experiments)}.")

    # Build feedback summary
    feedback_lines = []
    for exp in experiments:
        winner_label = "A (active)" if exp.winner == "a" else "B (challenger)" if exp.winner == "b" else "Tie"
        feedback_lines.append(
            f"Winner: {winner_label} | Rating A: {exp.rating_a}/5 | Rating B: {exp.rating_b}/5 | Feedback: {exp.analyst_feedback}"
        )

    # Get top-performing variant for reference
    best_q = await db.execute(
        select(PromptVariant).where(PromptVariant.prompt_type == body.prompt_type)
        .order_by(PromptVariant.avg_rating.desc()).limit(1)
    )
    best = best_q.scalar_one_or_none()

    refinement_prompt = f"""You are a prompt engineering expert. Your task is to improve an LLM prompt
based on analyst feedback from A/B testing.

CURRENT ACTIVE PROMPT ({active.variant_name}, generation {active.generation}):
---
{active.prompt_text}
---

BEST-PERFORMING VARIANT ({best.variant_name if best else 'N/A'}, avg rating {float(best.avg_rating) if best else 0:.1f}/5):
---
{best.prompt_text if best and best.id != active.id else 'Same as active.'}
---

ANALYST FEEDBACK FROM {len(experiments)} EXPERIMENTS:
{chr(10).join(feedback_lines)}

INSTRUCTIONS:
1. Analyse the feedback patterns — what do analysts consistently prefer?
2. Identify specific weaknesses in the current prompt
3. Generate an IMPROVED prompt that addresses the feedback
4. Keep the same output JSON schema — only change the instructions/framing
5. The prompt must contain {{text}} as a placeholder for the input document

Return ONLY the improved prompt text. No explanation, no markdown fences."""

    try:
        new_prompt_text = call_llm(refinement_prompt, max_tokens=8192)

        # Validate it still has the {text} placeholder
        if "{text}" not in new_prompt_text:
            new_prompt_text = new_prompt_text + "\n\n--- DOCUMENT TEXT ---\n{text}"

        # Create new variant
        new_variant = PromptVariant(
            id=uuid.uuid4(),
            prompt_type=body.prompt_type,
            variant_name=f"v{active.generation + 1}_llm_refined",
            prompt_text=new_prompt_text,
            is_active=False,
            is_candidate=True,
            parent_variant_id=active.id,
            generation=active.generation + 1,
            notes=f"LLM-refined from {active.variant_name} based on {len(experiments)} experiments. Key feedback: {experiments[0].analyst_feedback[:200] if experiments[0].analyst_feedback else 'N/A'}",
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


# ── Seed initial variants from current prompts ───────────────
@router.post("/experiments/seed")
async def seed_variants(db: AsyncSession = Depends(get_db)):
    """Seed prompt variants from the current prompts/__init__.py for A/B testing."""
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


# ── Stats dashboard ──────────────────────────────────────────
@router.get("/experiments/stats")
async def experiment_stats(db: AsyncSession = Depends(get_db)):
    """Overview stats for the experimentation system."""
    # Count by type
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
