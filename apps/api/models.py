"""
SQLAlchemy ORM models – mirrors §6 of the technical specification.
"""

from sqlalchemy import (
    Boolean, Column, Date, ForeignKey, Integer, LargeBinary, Numeric, Text, DateTime,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from apps.api.database import Base, TimestampMixin, new_uuid


# ─────────────────────────────────────────────────────────────────
# Companies
# ─────────────────────────────────────────────────────────────────
class Company(Base, TimestampMixin):
    __tablename__ = "companies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    ticker = Column(Text, unique=True, nullable=False, index=True)
    name = Column(Text, nullable=False)
    sector = Column(Text)
    industry = Column(Text)
    country = Column(Text)
    coverage_status = Column(Text, default="active")       # active | dropped | watchlist
    primary_analyst = Column(Text)

    # relationships
    documents = relationship("Document", back_populates="company", lazy="selectin")
    thesis_versions = relationship("ThesisVersion", back_populates="company", lazy="selectin")
    extracted_metrics = relationship("ExtractedMetric", back_populates="company", lazy="selectin")
    event_assessments = relationship("EventAssessment", back_populates="company", lazy="selectin")
    research_outputs = relationship("ResearchOutput", back_populates="company", lazy="selectin")


# ─────────────────────────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────────────────────────
class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    document_type = Column(Text)           # earnings_release | transcript | presentation | 10-Q | 10-K
    title = Column(Text)
    period_label = Column(Text)            # e.g. "2026_Q1"
    source = Column(Text)                  # email | ir_website | sec_filing | manual
    source_url = Column(Text)
    published_at = Column(DateTime(timezone=True))
    file_path = Column(Text)
    file_content = Column(LargeBinary)     # store raw file bytes in DB
    checksum = Column(Text)
    parsing_status = Column(Text, default="pending")  # pending | processing | completed | failed

    company = relationship("Company", back_populates="documents")
    sections = relationship("DocumentSection", back_populates="document", lazy="selectin")
    extracted_metrics = relationship("ExtractedMetric", back_populates="document", lazy="selectin")
    event_assessments = relationship("EventAssessment", back_populates="document", lazy="selectin")


# ─────────────────────────────────────────────────────────────────
# Document Sections
# ─────────────────────────────────────────────────────────────────
class DocumentSection(Base, TimestampMixin):
    __tablename__ = "document_sections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    section_type = Column(Text)            # header | body | table | footnote
    section_title = Column(Text)
    page_number = Column(Integer)
    text_content = Column(Text)
    # embedding = Column(VECTOR)  ← requires pgvector; add when needed

    document = relationship("Document", back_populates="sections")


# ─────────────────────────────────────────────────────────────────
# Extracted Metrics
# ─────────────────────────────────────────────────────────────────
class ExtractedMetric(Base, TimestampMixin):
    __tablename__ = "extracted_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    period_label = Column(Text)
    metric_name = Column(Text, nullable=False)
    metric_value = Column(Numeric)
    metric_text = Column(Text)             # raw text representation
    unit = Column(Text)                    # EUR_M | % | bps | x
    segment = Column(Text)
    geography = Column(Text)
    source_snippet = Column(Text)
    page_number = Column(Integer)
    confidence = Column(Numeric, default=1.0)
    needs_review = Column(Boolean, default=False)

    company = relationship("Company", back_populates="extracted_metrics")
    document = relationship("Document", back_populates="extracted_metrics")


# ─────────────────────────────────────────────────────────────────
# Thesis Versions
# ─────────────────────────────────────────────────────────────────
class ThesisVersion(Base, TimestampMixin):
    __tablename__ = "thesis_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    thesis_date = Column(Date, nullable=False)
    core_thesis = Column(Text)
    variant_perception = Column(Text)
    key_risks = Column(Text)
    debate_points = Column(Text)
    capital_allocation_view = Column(Text)
    valuation_framework = Column(Text)
    active = Column(Boolean, default=True)

    company = relationship("Company", back_populates="thesis_versions")


# ─────────────────────────────────────────────────────────────────
# Event Assessments
# ─────────────────────────────────────────────────────────────────
class EventAssessment(Base, TimestampMixin):
    __tablename__ = "event_assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    event_type = Column(Text)              # earnings | guidance_update | management_change
    thesis_direction = Column(Text)        # strengthened | weakened | unchanged
    surprise_level = Column(Text)          # none | minor | major
    summary = Column(Text)
    confidence = Column(Numeric, default=1.0)
    needs_review = Column(Boolean, default=True)

    company = relationship("Company", back_populates="event_assessments")
    document = relationship("Document", back_populates="event_assessments")


# ─────────────────────────────────────────────────────────────────
# Research Outputs
# ─────────────────────────────────────────────────────────────────
class ResearchOutput(Base, TimestampMixin):
    __tablename__ = "research_outputs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    period_label = Column(Text)
    output_type = Column(Text)             # briefing | ir_questions | thesis_drift | synthesis | full_analysis
    content_path = Column(Text)
    content_json = Column(Text)            # store full JSON output
    review_status = Column(Text, default="draft")  # draft | reviewed | approved
    approved_by = Column(Text)

    company = relationship("Company", back_populates="research_outputs")


# ─────────────────────────────────────────────────────────────────
# Review Queue
# ─────────────────────────────────────────────────────────────────
class ReviewQueueItem(Base, TimestampMixin):
    __tablename__ = "review_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    entity_type = Column(Text, nullable=False)  # metric | assessment | output
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    queue_reason = Column(Text)
    priority = Column(Text, default="normal")  # low | normal | high | critical
    assigned_to = Column(Text)
    status = Column(Text, default="open")      # open | approved | rejected | edited


# ─────────────────────────────────────────────────────────────────
# Tracked KPIs — the metrics an analyst wants to monitor per company
# ─────────────────────────────────────────────────────────────────
class TrackedKPI(Base, TimestampMixin):
    __tablename__ = "tracked_kpis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    kpi_name = Column(Text, nullable=False)          # e.g. "Revenue organic growth"
    unit = Column(Text)                               # e.g. "%" or "EUR_M"
    display_order = Column(Integer, default=0)

    company = relationship("Company")
    scores = relationship("KPIScore", back_populates="tracked_kpi", lazy="selectin")


# ─────────────────────────────────────────────────────────────────
# KPI Scores — actual values and analyst scores per period
# ─────────────────────────────────────────────────────────────────
class KPIScore(Base, TimestampMixin):
    __tablename__ = "kpi_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    tracked_kpi_id = Column(UUID(as_uuid=True), ForeignKey("tracked_kpis.id"), nullable=False)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    period_label = Column(Text, nullable=False)       # e.g. "2025_Q2"
    value = Column(Numeric)                            # extracted or manual value
    value_text = Column(Text)                          # text representation
    score = Column(Integer)                            # analyst score 1-5
    comment = Column(Text)                             # analyst comment

    tracked_kpi = relationship("TrackedKPI", back_populates="scores")
    company = relationship("Company")
