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
    # IC Summary fields
    recommendation = Column(Text)             # buy | hold | trim | exit | under_review
    catalyst = Column(Text)
    conviction = Column(Text)                 # high | medium | low
    what_would_make_us_wrong = Column(Text)
    disconfirming_evidence = Column(Text)
    positive_surprises = Column(Text)
    negative_surprises = Column(Text)

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


# ─────────────────────────────────────────────────────────────────
# Processing Jobs — track background pipeline status
# ─────────────────────────────────────────────────────────────────
class ProcessingJob(Base, TimestampMixin):
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    period_label = Column(Text, nullable=False)
    job_type = Column(Text, default="single")       # single | batch
    status = Column(Text, default="queued")          # queued | processing | completed | failed
    current_step = Column(Text)                      # upload | parse | extract | compare | surprises | briefing | ir_questions | synthesis
    steps_completed = Column(Text, default="[]")     # JSON array of completed step names
    progress_pct = Column(Integer, default=0)
    result_json = Column(Text)                       # full output JSON when done
    error_message = Column(Text)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Decision Log — immutable record of analyst actions
# ─────────────────────────────────────────────────────────────────
class DecisionLog(Base, TimestampMixin):
    __tablename__ = "decision_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    action = Column(Text, nullable=False)            # hold | add | trim | exit | initiate | watchlist
    rationale = Column(Text, nullable=False)
    old_weight = Column(Numeric)
    new_weight = Column(Numeric)
    conviction = Column(Integer)                     # 1-5
    author = Column(Text)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Analyst Notes — freeform research notes
# ─────────────────────────────────────────────────────────────────
class AnalystNote(Base, TimestampMixin):
    __tablename__ = "analyst_notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    note_type = Column(Text, default="general")      # general | call_note | meeting | thesis_update
    title = Column(Text)
    content = Column(Text, nullable=False)
    author = Column(Text)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Prompt Variants — versioned prompts for A/B testing
# ─────────────────────────────────────────────────────────────────
class PromptVariant(Base, TimestampMixin):
    __tablename__ = "prompt_variants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    prompt_type = Column(Text, nullable=False)        # extraction | synthesis | thesis_comparison | surprise | ir_questions | briefing
    variant_name = Column(Text, nullable=False)       # e.g. "v1_default", "v2_concise", "v3_llm_refined"
    prompt_text = Column(Text, nullable=False)
    is_active = Column(Boolean, default=False)         # the current default for this type
    is_candidate = Column(Boolean, default=True)       # eligible for A/B testing
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    total_runs = Column(Integer, default=0)
    avg_rating = Column(Numeric, default=0)
    parent_variant_id = Column(UUID(as_uuid=True), ForeignKey("prompt_variants.id"), nullable=True)
    generation = Column(Integer, default=1)            # tracks refinement generations
    notes = Column(Text)                               # why this variant was created


# ─────────────────────────────────────────────────────────────────
# A/B Experiments — records of side-by-side comparisons
# ─────────────────────────────────────────────────────────────────
class ABExperiment(Base, TimestampMixin):
    __tablename__ = "ab_experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    prompt_type = Column(Text, nullable=False)
    period_label = Column(Text)
    variant_a_id = Column(UUID(as_uuid=True), ForeignKey("prompt_variants.id"), nullable=False)
    variant_b_id = Column(UUID(as_uuid=True), ForeignKey("prompt_variants.id"), nullable=False)
    output_a = Column(Text)                            # JSON output from variant A
    output_b = Column(Text)                            # JSON output from variant B
    winner = Column(Text)                              # "a" | "b" | "tie" | null (pending)
    rating_a = Column(Integer)                         # 1-5 analyst rating
    rating_b = Column(Integer)                         # 1-5 analyst rating
    analyst_feedback = Column(Text)                    # freeform feedback on why
    status = Column(Text, default="pending")           # pending | completed

    variant_a = relationship("PromptVariant", foreign_keys=[variant_a_id])
    variant_b = relationship("PromptVariant", foreign_keys=[variant_b_id])
    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# ESG Data — one row per company, JSON blob for all PAI/ESG fields
# ─────────────────────────────────────────────────────────────────
class ESGData(Base, TimestampMixin):
    __tablename__ = "esg_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False, unique=True)
    data = Column(Text, default="{}")              # JSON blob of all ESG fields
    ai_summary = Column(Text, nullable=True)
    ai_summary_date = Column(DateTime(timezone=True), nullable=True)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Portfolios — multi-portfolio support
# ─────────────────────────────────────────────────────────────────
class Portfolio(Base, TimestampMixin):
    __tablename__ = "portfolios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    name = Column(Text, nullable=False)                # e.g. "OverGlob", "OverGac"
    description = Column(Text)
    benchmark = Column(Text)                           # e.g. "MSCI World"
    currency = Column(Text, default="USD")
    is_active = Column(Boolean, default=True)

    holdings = relationship("PortfolioHolding", back_populates="portfolio", lazy="selectin")


# ─────────────────────────────────────────────────────────────────
# Portfolio Holdings — links companies to portfolios with weights
# ─────────────────────────────────────────────────────────────────
class PortfolioHolding(Base, TimestampMixin):
    __tablename__ = "portfolio_holdings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    weight = Column(Numeric, default=0)                # current weight %
    cost_basis = Column(Numeric)                       # average cost
    shares = Column(Numeric)
    date_added = Column(DateTime(timezone=True))
    status = Column(Text, default="active")            # active | watchlist | exited

    portfolio = relationship("Portfolio", back_populates="holdings")
    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Price History — track current and historical prices
# ─────────────────────────────────────────────────────────────────
class PriceRecord(Base, TimestampMixin):
    __tablename__ = "price_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    price = Column(Numeric, nullable=False)
    currency = Column(Text, default="USD")
    price_date = Column(DateTime(timezone=True))
    source = Column(Text, default="manual")            # manual | api | import

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Valuation Scenarios — bear / base / bull cases per company
# ─────────────────────────────────────────────────────────────────
class ValuationScenario(Base, TimestampMixin):
    __tablename__ = "valuation_scenarios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    scenario_type = Column(Text, nullable=False)       # bear | base | bull
    probability = Column(Numeric)                      # 0-100%
    target_price = Column(Numeric)
    currency = Column(Text, default="USD")
    methodology = Column(Text)                         # EV/EBITDA | P/E | DCF | SOTP | P/TBV | FCF yield
    methodology_detail = Column(Text)                  # e.g. "7x 2026E EBITDA"
    key_assumptions = Column(Text)                     # JSON or freetext
    time_horizon = Column(Text, default="12m")         # 12m | 18m | 3y
    last_reviewed = Column(DateTime(timezone=True))
    author = Column(Text)

    company = relationship("Company")
