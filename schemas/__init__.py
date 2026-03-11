"""
Pydantic schemas for API request / response validation.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# Company
# ═══════════════════════════════════════════════════════════════════
class CompanyCreate(BaseModel):
    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    coverage_status: str = "active"
    primary_analyst: Optional[str] = None


class CompanyUpdate(BaseModel):
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    coverage_status: Optional[str] = None
    primary_analyst: Optional[str] = None


class CompanyOut(BaseModel):
    id: uuid.UUID
    ticker: str
    name: str
    sector: Optional[str]
    industry: Optional[str]
    country: Optional[str]
    coverage_status: Optional[str]
    primary_analyst: Optional[str]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════════════════
# Document
# ═══════════════════════════════════════════════════════════════════
class DocumentCreate(BaseModel):
    document_type: str
    title: str
    period_label: str
    source: str = "manual"
    source_url: Optional[str] = None
    published_at: Optional[datetime] = None


class DocumentOut(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    document_type: Optional[str]
    title: Optional[str]
    period_label: Optional[str]
    source: Optional[str]
    source_url: Optional[str]
    published_at: Optional[datetime]
    file_path: Optional[str]
    checksum: Optional[str]
    parsing_status: Optional[str]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════════════════
# Extracted Metric
# ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════
# Thesis
# ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════
# Event Assessment
# ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════
# Research Output
# ═══════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════
# Review Queue
# ═══════════════════════════════════════════════════════════════════
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
    supporting_signals: list[str] = Field(default_factory=list)
    weakening_signals: list[str] = Field(default_factory=list)
    new_risks: list[str] = Field(default_factory=list)
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


class BriefingSection(BaseModel):
    what_happened: str
    what_changed: str
    thesis_status: str
    risks: str
    follow_ups: str
    bottom_line: str
