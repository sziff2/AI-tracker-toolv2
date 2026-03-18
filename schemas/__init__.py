"""
Pydantic schemas for the Research CoWork Agent.
Includes probabilistic scenario and Bayesian signal structures.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# LLM Response Schemas  (used by prompt services)
# ═══════════════════════════════════════════════════════════════════

class ClassifiedDocument(BaseModel):
    document_type: str
    company_ticker: Optional[str] = None
    period_label: Optional[str] = None
    title: Optional[str] = None
    language: str = "en"
    confidence: float = 1.0


class ExtractedKPI(BaseModel):
    metric_name: str
    metric_value: Optional[float] = None
    metric_text: str
    unit: Optional[str] = None
    segment: Optional[str] = None
    geography: Optional[str] = None
    source_snippet: str
    page_number: Optional[int] = None
    confidence: float = 1.0


class GuidanceItem(BaseModel):
    metric_name: str
    guidance_type: str          # range | point | directional
    guidance_text: str
    low: Optional[float] = None
    high: Optional[float] = None
    unit: Optional[str] = None
    source_snippet: str
    confidence: float = 1.0


class ThesisComparison(BaseModel):
    thesis_direction: str       # strengthened | weakened | unchanged
    confidence: float = 0.85
    supporting_signals: list[str] = Field(default_factory=list)
    weakening_signals: list[str] = Field(default_factory=list)
    new_risks: list[str] = Field(default_factory=list)
    risks_receding: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    summary: str


class SurpriseItem(BaseModel):
    metric_or_topic: str
    direction: str              # positive | negative
    magnitude: str              # minor | major
    description: str
    source_snippet: str


class IRQuestion(BaseModel):
    topic: str
    question: str
    rationale: str


# ─────────────────────────────────────────────────────────────────
# Probabilistic / Bayesian structures
# ─────────────────────────────────────────────────────────────────

class ScenarioOutcome(BaseModel):
    """
    A single weighted scenario for a key assumption or forward outlook.
    Probabilities across all scenarios for the same dimension should sum to ~1.0.
    """
    label: str                          # e.g. "Bull", "Base", "Bear"
    probability: float                  # 0.0–1.0
    description: str                    # What this scenario implies
    key_trigger: str                    # What would cause this outcome
    thesis_impact: str                  # strengthened | weakened | neutral
    implied_return: Optional[str] = None  # e.g. "+30%", "-15%", qualitative


class BayesianSignal(BaseModel):
    """
    A single piece of evidence from the period that should update prior beliefs.
    Captures what the analyst believed before, what the evidence suggests,
    and the direction + magnitude of the update.
    """
    assumption: str                     # The prior belief being updated
    prior_view: str                     # What was believed before this period
    new_evidence: str                   # What the data/commentary showed
    posterior_direction: str            # strengthened | weakened | unchanged | reversed
    update_magnitude: str               # large | moderate | small
    confidence: float                   # 0.0–1.0, how certain is this signal
    source: str                         # earnings | transcript | broker | guidance


class AssumptionProbability(BaseModel):
    """
    A key thesis assumption with explicit probability assignment.
    Used to build a probability tree for the investment case.
    """
    assumption: str                     # e.g. "CBAM fully implemented by 2026"
    probability: float                  # analyst's current probability: 0.0–1.0
    direction: str                      # positive | negative | neutral for thesis
    rationale: str                      # why this probability was assigned
    key_watch: str                      # what to monitor to update this


class ProbabilisticBriefing(BaseModel):
    """
    Full probabilistic output embedded in the briefing.
    Scenario tree + Bayesian signal updates + key assumption probabilities.
    """
    scenarios: list[ScenarioOutcome] = Field(default_factory=list)
    bayesian_signals: list[BayesianSignal] = Field(default_factory=list)
    key_assumptions: list[AssumptionProbability] = Field(default_factory=list)
    overall_conviction_direction: str   # buy | hold | sell | watch
    overall_conviction_score: float     # 0.0–1.0
    conviction_rationale: str


# ─────────────────────────────────────────────────────────────────
# Enriched BriefingSection (backwards-compatible)
# ─────────────────────────────────────────────────────────────────

class BriefingSection(BaseModel):
    what_happened: str
    what_changed: str
    thesis_status: str
    risks: str
    follow_ups: str
    bottom_line: str
    # Optional probabilistic enrichment — present only when prompts request it
    probabilistic: Optional[ProbabilisticBriefing] = None


# ═══════════════════════════════════════════════════════════════════
# API output schemas (ORM → response)
# ═══════════════════════════════════════════════════════════════════

class CompanyOut(BaseModel):
    id: uuid.UUID
    ticker: str
    name: str
    sector: Optional[str]
    country: Optional[str]
    coverage_status: Optional[str]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    document_type: Optional[str]
    period_label: Optional[str]
    original_filename: Optional[str]
    parsing_status: Optional[str]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class DocumentOutFull(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    document_type: Optional[str]
    period_label: Optional[str]
    original_filename: Optional[str]
    file_size: Optional[int]
    page_count: Optional[int]
    upload_timestamp: Optional[datetime]
    file_path: Optional[str]
    checksum: Optional[str]
    parsing_status: Optional[str]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ExtractedMetricOut(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    document_id: uuid.UUID
    period_label: Optional[str]
    metric_name: str
    metric_value: Optional[float]
    metric_text: Optional[str]
    unit: Optional[str]
    segment: Optional[str]
    geography: Optional[str]
    source_snippet: Optional[str]
    page_number: Optional[int]
    confidence: Optional[float]
    needs_review: bool
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ThesisCreate(BaseModel):
    thesis_date: date
    core_thesis: str
    variant_perception: Optional[str] = None
    key_risks: Optional[str] = None
    debate_points: Optional[str] = None
    capital_allocation_view: Optional[str] = None
    valuation_framework: Optional[str] = None


class ThesisOut(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    thesis_date: date
    core_thesis: Optional[str]
    variant_perception: Optional[str]
    key_risks: Optional[str]
    debate_points: Optional[str]
    capital_allocation_view: Optional[str]
    valuation_framework: Optional[str]
    active: bool
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class EventAssessmentOut(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    document_id: uuid.UUID
    event_type: Optional[str]
    thesis_direction: Optional[str]
    surprise_level: Optional[str]
    summary: Optional[str]
    confidence: Optional[float]
    needs_review: bool
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ResearchOutputOut(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    period_label: Optional[str]
    output_type: Optional[str]
    content_path: Optional[str]
    review_status: Optional[str]
    approved_by: Optional[str]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ReviewAction(BaseModel):
    comment: Optional[str] = None


class ReviewQueueOut(BaseModel):
    id: uuid.UUID
    entity_type: str
    entity_id: uuid.UUID
    queue_reason: Optional[str]
    priority: Optional[str]
    assigned_to: Optional[str]
    status: str
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}
