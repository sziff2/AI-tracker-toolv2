"""
SQLAlchemy ORM models – mirrors §6 of the technical specification.

Changes vs previous version:
  - Removed Document.file_content (LargeBinary) — CLAUDE.md explicitly prohibits
    storing raw PDFs in PostgreSQL (Railway storage quota risk).
  - Added JSONB import for agent tables (Text retained on existing columns).
  - Added 6 agent infrastructure tables: AgentOutput, AgentCalibration,
    ContextContract, SectorThesis, ThesisMacroDependency, PipelineRun.
  - Added HarvestReport (weekly harvest summary, referenced in CLAUDE.md).
  - Added agent columns to ThesisVersion: pillars, macro_dependencies,
    sector_dependencies, generated_by, contract_version.
  - Added agent columns to ProcessingJob: agent_results, agents_completed,
    agents_failed.
"""

from sqlalchemy import (
    Boolean, Column, Date, Float, ForeignKey, Index, Integer,
    LargeBinary, Numeric, Text, DateTime,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from apps.api.database import Base, TimestampMixin, new_uuid

# Tier 3.4 — pgvector embedding column type. Fall back to Text when the
# package isn't installed locally yet so `pip install -r requirements.txt`
# works cleanly from a stale environment; Railway prod always has it.
try:
    from pgvector.sqlalchemy import Vector as _Vector
except ImportError:  # pragma: no cover — only hit pre-install
    _Vector = lambda _dim: Text  # type: ignore[assignment]


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
    cik = Column(Text)                                     # SEC CIK number for EDGAR lookups
    peer_tickers = Column(JSONB, default=list)             # analyst-curated peer set for Tier 5.1 comparison

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
    __table_args__ = (
        Index("ix_documents_company_period", "company_id", "period_label"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    document_type = Column(Text)           # earnings_release | transcript | presentation | 10-Q | 10-K
    title = Column(Text)
    period_label = Column(Text)            # e.g. "2026_Q1"
    source = Column(Text)                  # email | ir_website | sec_filing | manual
    source_url = Column(Text)
    published_at = Column(DateTime(timezone=True))
    file_path = Column(Text)
    # NOTE: file_content (LargeBinary) removed — CLAUDE.md prohibits storing raw
    # PDFs in PostgreSQL. Parsed text lives in DocumentSection. Raw files on disk
    # at file_path only.
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
    # Tier 3.4 — 384-dim embedding matching BAAI/bge-small-en-v1.5
    # (local sentence-transformers, no third-party API). Nullable —
    # rows without an embedding are invisible to vector search but
    # still searchable via keyword. See services/vector_search.py.
    embedding = Column(_Vector(384), nullable=True)

    document = relationship("Document", back_populates="sections")


# ─────────────────────────────────────────────────────────────────
# Extracted Metrics
# ─────────────────────────────────────────────────────────────────
class ExtractedMetric(Base, TimestampMixin):
    __tablename__ = "extracted_metrics"
    __table_args__ = (
        Index("ix_metrics_company_period", "company_id", "period_label"),
        Index("ix_metrics_document", "document_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    period_label = Column(Text)
    # Distinguishes the SHAPE of the period: a single quarter ("Q") vs
    # a full fiscal year ("FY") vs interim halves ("H1"/"H2"). Same
    # year+quarter labels mean very different things — Q4 2025 sales
    # ($8B for ARW) vs FY 2025 sales ($31B). Without this the dedup
    # silently collapses 12-month and 3-month figures into one row
    # whichever was extracted last. Default "Q" preserves existing
    # quarterly behaviour; the targeted IS/BS/CF native-extraction
    # pass writes "FY" for 10-K full-year line items.
    period_frequency = Column(Text, default="Q", nullable=True)
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
    # Qualifier enrichment from v2 extraction pipeline
    is_one_off = Column(Boolean, default=False)     # one-off / non-recurring item
    qualifier_json = Column(JSONB, nullable=True)   # {hedge_terms, attribution, temporal, ...}

    company = relationship("Company", back_populates="extracted_metrics")
    document = relationship("Document", back_populates="extracted_metrics")


# ─────────────────────────────────────────────────────────────────
# Thesis Versions
# ─────────────────────────────────────────────────────────────────
class ThesisVersion(Base, TimestampMixin):
    __tablename__ = "thesis_versions"
    __table_args__ = (
        Index("ix_thesis_company_active", "company_id", "active"),
    )

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
    # Agent architecture fields (migration 0.6)
    # Structured thesis as pillars — JSON array of {pillar, evidence, risk} objects
    pillars = Column(JSONB, nullable=True)
    # IDs of macro assumptions this thesis depends on (from context_contracts)
    macro_dependencies = Column(JSONB, nullable=True)
    # Sector-level views this thesis assumes (from sector_theses)
    sector_dependencies = Column(JSONB, nullable=True)
    # "agent" | "analyst" | "hybrid" — who/what produced this version
    generated_by = Column(Text, nullable=True)
    # FK to the context_contracts.version that was active when this was generated
    contract_version = Column(Integer, nullable=True)

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
    __table_args__ = (
        Index("ix_outputs_company_period", "company_id", "period_label"),
    )

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
# Tracked KPIs
# ─────────────────────────────────────────────────────────────────
class TrackedKPI(Base, TimestampMixin):
    __tablename__ = "tracked_kpis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    kpi_name = Column(Text, nullable=False)
    unit = Column(Text)
    display_order = Column(Integer, default=0)

    company = relationship("Company")
    scores = relationship("KPIScore", back_populates="tracked_kpi", lazy="selectin")


# ─────────────────────────────────────────────────────────────────
# KPI Scores
# ─────────────────────────────────────────────────────────────────
class KPIScore(Base, TimestampMixin):
    __tablename__ = "kpi_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    tracked_kpi_id = Column(UUID(as_uuid=True), ForeignKey("tracked_kpis.id"), nullable=False)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    period_label = Column(Text, nullable=False)
    value = Column(Numeric)
    value_text = Column(Text)
    score = Column(Integer)
    comment = Column(Text)

    tracked_kpi = relationship("TrackedKPI", back_populates="scores")
    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Processing Jobs
# ─────────────────────────────────────────────────────────────────
class ProcessingJob(Base, TimestampMixin):
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    period_label = Column(Text, nullable=False)
    job_type = Column(Text, default="single")       # single | batch
    status = Column(Text, default="queued")          # queued | processing | completed | failed
    current_step = Column(Text)
    steps_completed = Column(Text, default="[]")     # JSON array
    progress_pct = Column(Integer, default=0)
    log_entries = Column(Text, default="[]")         # JSON array of {ts, level, message}
    result_json = Column(Text)
    error_message = Column(Text)
    model = Column(Text, default="standard")         # fast | standard | deep
    # Agent architecture fields (migration 0.6)
    # Full JSON blob of per-agent results keyed by agent_id
    agent_results = Column(Text, nullable=True)
    # Comma-separated list of agent_ids that completed successfully
    agents_completed = Column(Text, nullable=True)
    # Comma-separated list of agent_ids that failed
    agents_failed = Column(Text, nullable=True)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Decision Log
# ─────────────────────────────────────────────────────────────────
class DecisionLog(Base, TimestampMixin):
    __tablename__ = "decision_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    action = Column(Text, nullable=False)
    rationale = Column(Text, nullable=False)
    old_weight = Column(Numeric)
    new_weight = Column(Numeric)
    conviction = Column(Integer)
    author = Column(Text)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Analyst Notes
# ─────────────────────────────────────────────────────────────────
class AnalystNote(Base, TimestampMixin):
    __tablename__ = "analyst_notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    note_type = Column(Text, default="general")
    title = Column(Text)
    content = Column(Text, nullable=False)
    author = Column(Text)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Prompt Variants
# ─────────────────────────────────────────────────────────────────
class PromptVariant(Base, TimestampMixin):
    __tablename__ = "prompt_variants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    prompt_type = Column(Text, nullable=False)
    variant_name = Column(Text, nullable=False)
    prompt_text = Column(Text, nullable=False)
    is_active = Column(Boolean, default=False)
    is_candidate = Column(Boolean, default=True)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    total_runs = Column(Integer, default=0)
    avg_rating = Column(Numeric, default=0)
    parent_variant_id = Column(UUID(as_uuid=True), ForeignKey("prompt_variants.id"), nullable=True)
    generation = Column(Integer, default=1)
    notes = Column(Text)


# ─────────────────────────────────────────────────────────────────
# A/B Experiments
# ─────────────────────────────────────────────────────────────────
class ABExperiment(Base, TimestampMixin):
    __tablename__ = "ab_experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    prompt_type = Column(Text, nullable=False)
    period_label = Column(Text)
    variant_a_id = Column(UUID(as_uuid=True), ForeignKey("prompt_variants.id"), nullable=False)
    variant_b_id = Column(UUID(as_uuid=True), ForeignKey("prompt_variants.id"), nullable=False)
    output_a = Column(Text)
    output_b = Column(Text)
    winner = Column(Text)                              # "a" | "b" | "tie" | null
    rating_a = Column(Integer)
    rating_b = Column(Integer)
    analyst_feedback = Column(Text)
    status = Column(Text, default="pending")

    variant_a = relationship("PromptVariant", foreign_keys=[variant_a_id])
    variant_b = relationship("PromptVariant", foreign_keys=[variant_b_id])
    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# ESG Data
# ─────────────────────────────────────────────────────────────────
class ESGData(Base, TimestampMixin):
    __tablename__ = "esg_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False, unique=True)
    data = Column(Text, default="{}")
    ai_summary = Column(Text, nullable=True)
    ai_summary_date = Column(DateTime(timezone=True), nullable=True)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Portfolios
# ─────────────────────────────────────────────────────────────────
class Portfolio(Base, TimestampMixin):
    __tablename__ = "portfolios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    name = Column(Text, nullable=False)
    description = Column(Text)
    benchmark = Column(Text)
    currency = Column(Text, default="USD")
    is_active = Column(Boolean, default=True)

    holdings = relationship("PortfolioHolding", back_populates="portfolio", lazy="selectin")


# ─────────────────────────────────────────────────────────────────
# Portfolio Holdings
# ─────────────────────────────────────────────────────────────────
class PortfolioHolding(Base, TimestampMixin):
    __tablename__ = "portfolio_holdings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    weight = Column(Numeric, default=0)
    cost_basis = Column(Numeric)
    shares = Column(Numeric)
    date_added = Column(DateTime(timezone=True))
    status = Column(Text, default="active")

    portfolio = relationship("Portfolio", back_populates="holdings")
    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Scenario Snapshots
# ─────────────────────────────────────────────────────────────────
class ScenarioSnapshot(Base, TimestampMixin):
    __tablename__ = "scenario_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    snapshot_date = Column(DateTime(timezone=True), nullable=False)
    scenario_type = Column(Text, nullable=False)
    target_price = Column(Numeric)
    probability = Column(Numeric)
    current_price = Column(Numeric)
    currency = Column(Text, default="USD")
    source = Column(Text, default="manual")

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Price Records
# ─────────────────────────────────────────────────────────────────
class PriceRecord(Base, TimestampMixin):
    __tablename__ = "price_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    price = Column(Numeric, nullable=False)
    currency = Column(Text, default="USD")
    price_date = Column(DateTime(timezone=True))
    source = Column(Text, default="manual")

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# FX Rates (daily / monthly-EOM, for cross-currency correlation/analytics)
# ─────────────────────────────────────────────────────────────────
class FXRate(Base, TimestampMixin):
    __tablename__ = "fx_rates"
    __table_args__ = (
        Index("ix_fx_ccy_date", "currency", "rate_date", unique=True),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    currency = Column(Text, nullable=False)          # 3-letter ISO, e.g. "GBP", "EUR"
    rate_date = Column(Date, nullable=False)         # EOM or trade date
    rate_to_usd = Column(Numeric(18, 8), nullable=False)   # 1 unit of <currency> in USD
    source = Column(Text, default="yahoo")


# ─────────────────────────────────────────────────────────────────
# Valuation Scenarios
# ─────────────────────────────────────────────────────────────────
class ValuationScenario(Base, TimestampMixin):
    __tablename__ = "valuation_scenarios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    scenario_type = Column(Text, nullable=False)
    probability = Column(Numeric)
    target_price = Column(Numeric)
    currency = Column(Text, default="USD")
    methodology = Column(Text)
    methodology_detail = Column(Text)
    key_assumptions = Column(Text)
    time_horizon = Column(Text, default="12m")
    last_reviewed = Column(DateTime(timezone=True))
    author = Column(Text)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Management Statements
# ─────────────────────────────────────────────────────────────────
class ManagementStatement(Base, TimestampMixin):
    __tablename__ = "management_statements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    statement_date = Column(Text)
    speaker = Column(Text)
    category = Column(Text, nullable=False)
    statement_text = Column(Text, nullable=False)
    target_metric = Column(Text)
    target_value = Column(Text)
    target_direction = Column(Text)
    target_timeframe = Column(Text)
    target_deadline = Column(Text)
    confidence_type = Column(Text)
    source_snippet = Column(Text)
    status = Column(Text, default="open")
    score = Column(Integer)
    outcome_value = Column(Text)
    outcome_date = Column(Text)
    outcome_evidence = Column(Text)
    assessed_by = Column(Text)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Consensus Expectations — analyst-curated street estimates
# Used to render Actual-vs-Consensus beat/miss in the Results tab
# alongside extracted_metrics and to feed agent prompts the right
# benchmark for "what did the market expect" framing.
# ─────────────────────────────────────────────────────────────────
class ConsensusExpectation(Base, TimestampMixin):
    __tablename__ = "consensus_expectations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False, index=True)
    period_label = Column(Text, nullable=False, index=True)
    metric_name = Column(Text, nullable=False)   # canonical name, e.g. "Revenue", "EPS", "NII"
    consensus_value = Column(Numeric)            # numeric estimate
    unit = Column(Text)                          # e.g. "SEK_M", "USD_M", "%", "x"
    source = Column(Text)                        # e.g. "VARA", "Bloomberg", "Visible Alpha", "Analyst inputs"
    notes = Column(Text)                         # free-form: "median of 12 analysts", etc.
    uploaded_by = Column(Text)                   # analyst name from localStorage

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Management Execution Scorecard
# ─────────────────────────────────────────────────────────────────
class ExecutionScorecard(Base, TimestampMixin):
    __tablename__ = "execution_scorecards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    period = Column(Text)
    overall_score = Column(Numeric)
    guidance_bias = Column(Text)
    execution_reliability = Column(Text)
    strategic_consistency = Column(Text)
    category_scores = Column(Text)
    total_statements = Column(Integer, default=0)
    delivered_count = Column(Integer, default=0)
    missed_count = Column(Integer, default=0)
    open_count = Column(Integer, default=0)
    ai_assessment = Column(Text)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# LLM Usage Log
# ─────────────────────────────────────────────────────────────────
class LLMUsageLog(Base):
    __tablename__ = "llm_usage_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    model = Column(Text, nullable=False)
    feature = Column(Text, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    cost_usd = Column(Numeric)
    ticker = Column(Text)
    period_label = Column(Text)
    duration_ms = Column(Integer)


# ─────────────────────────────────────────────────────────────────
# Extraction Feedback
# ─────────────────────────────────────────────────────────────────
class ExtractionFeedback(Base, TimestampMixin):
    __tablename__ = "extraction_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    period_label = Column(Text)
    section = Column(Text, nullable=False)
    tag = Column(Text, nullable=False)
    comment = Column(Text)
    source_snippet = Column(Text)
    metric_id = Column(UUID(as_uuid=True), ForeignKey("extracted_metrics.id"), nullable=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    prompt_type = Column(Text)
    author = Column(Text)
    promoted = Column(Boolean, default=False)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Harvester Sources
# ─────────────────────────────────────────────────────────────────
class HarvesterSource(Base, TimestampMixin):
    __tablename__ = "harvester_sources"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id       = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False, unique=True)
    ir_docs_url      = Column(Text, nullable=True)
    ir_url           = Column(Text, nullable=True)
    rss_url          = Column(Text, nullable=True)
    ir_reachable     = Column(Boolean, default=False)
    discovery_method = Column(Text, nullable=True)
    last_checked_at  = Column(DateTime(timezone=True), nullable=True)
    override         = Column(Boolean, default=False)
    notes            = Column(Text, nullable=True)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Harvested Documents
# ─────────────────────────────────────────────────────────────────
class HarvestedDocument(Base, TimestampMixin):
    __tablename__ = "harvested_documents"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id    = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    source        = Column(Text)
    source_url    = Column(Text, unique=True, nullable=False)
    headline      = Column(Text)
    period_label  = Column(Text, nullable=True)
    discovered_at = Column(DateTime(timezone=True))
    ingested      = Column(Boolean, default=False)
    document_id   = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    error         = Column(Text, nullable=True)

    company  = relationship("Company")
    document = relationship("Document")


# ─────────────────────────────────────────────────────────────────
# Harvest Reports — weekly harvest summary (CLAUDE.md §Weekly Auto-Harvest)
# ─────────────────────────────────────────────────────────────────
class HarvestReport(Base, TimestampMixin):
    __tablename__ = "harvest_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    # "weekly_auto" | "manual" | "single_company"
    trigger = Column(Text, nullable=False, default="weekly_auto")
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    companies_attempted = Column(Integer, default=0)
    companies_succeeded = Column(Integer, default=0)
    companies_failed = Column(Integer, default=0)
    documents_found = Column(Integer, default=0)
    documents_new = Column(Integer, default=0)
    # Per-company breakdown — JSONB for queryability
    per_company = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)


# ═════════════════════════════════════════════════════════════════
# AGENT INFRASTRUCTURE TABLES (migration 0.6)
# ═════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# Context Contracts — shared macro assumptions injected into all agents
# ─────────────────────────────────────────────────────────────────
class ContextContract(Base, TimestampMixin):
    __tablename__ = "context_contracts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    # Monotonically increasing version — agents record which version they ran under
    version = Column(Integer, nullable=False, unique=True)
    is_active = Column(Boolean, default=False)   # only one row is_active=True at a time
    # Macro assumptions blob — regime, rates, credit, growth, FX, commodities, geopolitical
    # JSONB so individual keys are queryable: e.g. WHERE macro_assumptions->>'regime' = 'risk_off'
    macro_assumptions = Column(JSONB, nullable=False, default=dict)
    # Analyst overrides applied on top of agent-generated assumptions
    analyst_overrides = Column(JSONB, nullable=True)
    authored_by = Column(Text, nullable=True)    # "macro_regime_agent" | analyst name

    sector_theses = relationship("SectorThesis", back_populates="contract", lazy="selectin")


# ─────────────────────────────────────────────────────────────────
# Sector Theses — sector-level views linked to a context contract
# ─────────────────────────────────────────────────────────────────
class SectorThesis(Base, TimestampMixin):
    __tablename__ = "sector_theses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    sector = Column(Text, nullable=False, index=True)
    contract_id = Column(UUID(as_uuid=True), ForeignKey("context_contracts.id"), nullable=False)
    # Full sector thesis — outlook, risks, key themes, relative value
    thesis_json = Column(JSONB, nullable=False, default=dict)
    generated_by = Column(Text, nullable=True)    # "sector_thesis_agent" | analyst name

    contract = relationship("ContextContract", back_populates="sector_theses")
    macro_dependencies = relationship(
        "ThesisMacroDependency", back_populates="sector_thesis", lazy="selectin"
    )


# ─────────────────────────────────────────────────────────────────
# Thesis Macro Dependencies — links a thesis to specific macro assumptions
# ─────────────────────────────────────────────────────────────────
class ThesisMacroDependency(Base, TimestampMixin):
    __tablename__ = "thesis_macro_dependencies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    sector_thesis_id = Column(UUID(as_uuid=True), ForeignKey("sector_theses.id"), nullable=False)
    # Which assumption key this thesis depends on (e.g. "rates", "regime", "usd_strength")
    assumption_key = Column(Text, nullable=False)
    # Direction sensitivity: "positive" | "negative" | "neutral"
    sensitivity = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    sector_thesis = relationship("SectorThesis", back_populates="macro_dependencies")


# ─────────────────────────────────────────────────────────────────
# Agent Outputs — one row per agent × pipeline run
# ─────────────────────────────────────────────────────────────────
class AgentOutput(Base, TimestampMixin):
    __tablename__ = "agent_outputs"
    __table_args__ = (
        Index("ix_agent_outputs_company_period", "company_id", "period_label"),
        Index("ix_agent_outputs_agent_company", "agent_id", "company_id"),
        Index("ix_agent_outputs_pipeline_run", "pipeline_run_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    # Which agent produced this output
    agent_id = Column(Text, nullable=False)
    # Context
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    period_label = Column(Text, nullable=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=True)
    # Execution
    pipeline_run_id = Column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=True)
    status = Column(Text, nullable=False, default="completed")  # completed | failed | skipped | degraded
    # Results — JSONB for queryability (e.g. filter by output->>'thesis_direction')
    output_json = Column(JSONB, nullable=True)
    confidence = Column(Float, nullable=True)
    qc_score = Column(Float, nullable=True)          # set post-hoc by QC agent
    # Performance
    duration_ms = Column(Integer, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    # Prompt lineage
    prompt_variant_id = Column(Text, nullable=True)  # A/B variant used
    # Predictions for calibration tracking
    predictions_json = Column(JSONB, nullable=True)  # list of {metric, direction, horizon, value}
    predictions_resolved = Column(Boolean, default=False)
    # Error details when status = "failed"
    error_message = Column(Text, nullable=True)
    # Record of what inputs were passed to this agent
    inputs_used = Column(JSONB, nullable=True)
    # TTL — when this cached output expires (cache_ttl_hours from agent class)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    company = relationship("Company")
    pipeline_run = relationship("PipelineRun", back_populates="agent_outputs")


# ─────────────────────────────────────────────────────────────────
# Agent Calibration — per-agent accuracy tracking over time
# ─────────────────────────────────────────────────────────────────
class AgentCalibration(Base, TimestampMixin):
    __tablename__ = "agent_calibration"

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    # One row per agent_id — upserted on each resolution
    agent_id = Column(Text, nullable=False, unique=True)
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy = Column(Float, nullable=True)           # correct / total
    # Per-direction breakdown — JSONB: {"up": {"total": 10, "correct": 7}, ...}
    direction_accuracy = Column(JSONB, nullable=True)
    # Confidence calibration — how well confidence scores correlate with accuracy
    calibration_score = Column(Float, nullable=True)
    last_resolved_at = Column(DateTime(timezone=True), nullable=True)


# ─────────────────────────────────────────────────────────────────
# Pipeline Runs — audit trail: one row per "Run Analysis" click
# ─────────────────────────────────────────────────────────────────
class PipelineRun(Base, TimestampMixin):
    __tablename__ = "pipeline_runs"
    __table_args__ = (
        Index("ix_pipeline_runs_company_period", "company_id", "period_label"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    # Context
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    period_label = Column(Text, nullable=True)
    # What triggered this run
    trigger = Column(Text, nullable=False, default="manual")  # manual | auto | scheduled
    # Execution state
    status = Column(Text, nullable=False, default="running")  # running | completed | failed | cancelled | halted_incomplete | phase_a_incomplete | budget_exceeded | aborted
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    # Cost accounting — aggregated from all AgentOutput rows in this run
    total_cost_usd = Column(Float, nullable=True)
    total_input_tokens = Column(Integer, nullable=True)
    total_output_tokens = Column(Integer, nullable=True)
    total_llm_calls = Column(Integer, nullable=True)
    # Agent progress
    agents_planned = Column(Integer, nullable=True)
    agents_completed = Column(Integer, nullable=True)
    agents_failed = Column(Integer, nullable=True)
    agents_skipped = Column(Integer, nullable=True)
    # Quality
    overall_qc_score = Column(Float, nullable=True)
    # Error details if status = "failed"
    error_message = Column(Text, nullable=True)
    # Full per-agent execution log — JSONB array of
    # {agent_id, status, duration_ms, cost_usd, error} for the UI timeline view
    agent_execution_log = Column(JSONB, nullable=True)
    # Structured warnings from pre-flight gates (completeness, source coverage)
    # and any other non-fatal signals attached during the run.
    # Shape: {"completeness": CompletenessReport, "source_coverage": SourceCoverageReport, ...}
    warnings = Column(JSONB, nullable=True)
    # Which context contract version was active for this run
    contract_version = Column(Integer, nullable=True)

    company = relationship("Company")
    agent_outputs = relationship("AgentOutput", back_populates="pipeline_run", lazy="selectin")


# ─────────────────────────────────────────────────────────────────
# Extraction Profiles — per-document enrichment data from v2 pipeline
# ─────────────────────────────────────────────────────────────────
class ExtractionProfile(Base, TimestampMixin):
    """Stores enriched extraction metadata produced by the v2 pipeline.

    One row per document extraction. Contains confidence profiles, segment
    decomposition, disappearance flags, non-GAAP bridges, and reconciliation
    data. Agents read this via context_builder to understand extraction quality.
    """
    __tablename__ = "extraction_profiles"
    __table_args__ = (
        Index("ix_extraction_profiles_company_period", "company_id", "period_label"),
        Index("ix_extraction_profiles_document", "document_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    period_label = Column(Text)
    extraction_method = Column(Text)            # section_aware_v2 | legacy
    sections_found = Column(Integer)
    section_types = Column(JSONB)               # ["financial_statements", "mda", ...]
    items_extracted = Column(Integer)
    # Document-level confidence profile from qualifier analysis
    confidence_profile = Column(JSONB)          # {overall_signal, hedge_rate, one_off_rate, ...}
    # Segment decomposition summary
    segment_data = Column(JSONB)                # {segments: [{name, revenue, ...}], sum_check}
    # Disappeared metrics from prior period
    disappearance_flags = Column(JSONB)         # {disappeared: [...], new: [...]}
    # Non-GAAP bridge data
    non_gaap_bridges = Column(JSONB)            # [{gaap_metric, adjusted_metric, adjustments}]
    non_gaap_comparison = Column(JSONB)         # {total_flags, gap_trend}
    # MD&A narrative text (capped, for synthesis context)
    mda_narrative = Column(Text)
    detected_period = Column(Text)
    # Reconciliation report from extraction_reconciler — Q-sum vs FY,
    # segment vs consolidated, BS equation, P&L vs CF cross-checks.
    # Shape: {passed: bool, checks_run: int, checks_passed: int, issues: [...]}
    reconciliation = Column(JSONB)

    company = relationship("Company")
    document = relationship("Document")


# ─────────────────────────────────────────────────────────────────
# Ingestion Triage — per-candidate classification and priority decision
# ─────────────────────────────────────────────────────────────────
class IngestionTriage(Base, TimestampMixin):
    """Audit row for every harvester candidate that Document Triage Agent
    classified. Used to track whether the agent got the period/type right,
    whether the analyst overrode the auto-ingest decision, and whether the
    ingested document ended up being useful. Feeds the source-quality
    agent's reliability scoring over time."""
    __tablename__ = "ingestion_triage"
    __table_args__ = (
        Index("ix_ingestion_triage_company", "company_id"),
        Index("ix_ingestion_triage_source_url", "source_url"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True)
    # Candidate identity — same source_url the dispatcher dedupes on
    source_url = Column(Text, nullable=False)
    candidate_title = Column(Text, nullable=True)
    source_type = Column(Text, nullable=True)      # sec_edgar | ir_regex | ir_llm | investegate | rss
    # Agent classification
    document_type = Column(Text, nullable=True)    # annual_report | 10-Q | transcript | ...
    period_label = Column(Text, nullable=True)
    priority = Column(Text, nullable=True)         # immediate | normal | low | skip
    relevance_score = Column(Integer, nullable=True)   # 0-100
    auto_ingest = Column(Boolean, default=True)
    needs_review = Column(Boolean, default=False)
    rationale = Column(Text, nullable=True)
    # Outcome tracking
    was_ingested = Column(Boolean, default=False)
    analyst_override = Column(Text, nullable=True) # null | upgraded | downgraded | reclassified | skipped
    was_useful = Column(Boolean, nullable=True)    # set later from analyst feedback
    # Link to the Document row if ingestion went ahead
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)

    company = relationship("Company")


# ─────────────────────────────────────────────────────────────────
# Coverage Rescan Log — one row per auto/manual rescan attempt
# ─────────────────────────────────────────────────────────────────
class CoverageRescanLog(Base, TimestampMixin):
    """Records every Coverage-Monitor-triggered rescan attempt so we
    don't hammer a broken source. The Coverage Monitor checks this table
    before triggering a new scan — if the same (company, doc_type,
    expected_period) was rescanned in the last 24h, skip."""
    __tablename__ = "coverage_rescan_log"
    __table_args__ = (
        Index("ix_coverage_rescan_company", "company_id"),
        Index("ix_coverage_rescan_triggered_at", "triggered_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id"), nullable=False)
    ticker = Column(Text, nullable=True)
    doc_type = Column(Text, nullable=True)
    expected_period = Column(Text, nullable=True)
    # "auto" = triggered by CoverageMonitor. "manual" = analyst clicked Rescan.
    triggered_by = Column(Text, nullable=False, default="auto")
    triggered_at = Column(DateTime(timezone=True), nullable=False)
    sources_tried = Column(JSONB, nullable=True)     # list[str]
    candidates_found = Column(Integer, nullable=True)
    result = Column(Text, nullable=True)             # success | no_new_candidates | error
    error_message = Column(Text, nullable=True)

    company = relationship("Company")
