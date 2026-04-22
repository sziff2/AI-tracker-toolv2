"""Tests for the briefing PDF renderer (Tier 3.3).

Verifies the renderer produces a valid PDF byte stream and handles
the sparse-data cases gracefully. We do not assert layout — that's a
visual review concern — but we do check that optional sections noop
silently and the output is a well-formed PDF."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

reportlab = pytest.importorskip("reportlab")

from services.briefing_pdf import (
    render_briefing_pdf,
    _fmt, _fmt_pct, _escape,
)


# ─────────────────────────────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────────────────────────────

def test_fmt_handles_none_and_empty():
    assert _fmt(None) == "—"
    assert _fmt("") == "—"


def test_fmt_precision_bucket():
    # <10 → 2dp
    assert _fmt(3.456) == "$3.46"
    # 10-999 → 1dp
    assert _fmt(42.7) == "$42.7"
    # ≥1000 → commas, no dp
    assert _fmt(12345.6) == "$12,346"


def test_fmt_pct_interprets_probabilities_vs_percent():
    # 0-1 treated as probability
    assert _fmt_pct(0.3) == "30%"
    # >1 treated as already-percent
    assert _fmt_pct(15.3) == "+15.3%"
    assert _fmt_pct(-8.0) == "-8.0%"


def test_escape_html_special_chars():
    assert _escape("A & B < C") == "A &amp; B &lt; C"


# ─────────────────────────────────────────────────────────────────
# Renderer — happy path
# ─────────────────────────────────────────────────────────────────

def test_renderer_returns_valid_pdf_bytes():
    pdf = render_briefing_pdf(
        company_name="Ally Financial Inc",
        ticker="ALLY US",
        period="2025_Q3",
        briefing={
            "headline":           "Core ROTCE restated; Q3 broadly in line",
            "bottom_line":        "Neutral; watch credit migration.",
            "what_happened":      "Q3 delivered $1.03 diluted EPS vs consensus.",
            "management_message": "Management emphasised deposit growth.",
            "thesis_impact":      "Thesis intact; pricing power confirmed.",
        },
        thesis={
            "core_thesis": "Deposit-franchise leader at cycle trough.",
            "key_risks":   "Credit migration, used-car price risk.",
        },
        scenarios={
            "bull": {"target_price": 55.0, "probability": 0.25, "methodology": "12x 2026 EPS"},
            "base": {"target_price": 42.0, "probability": 0.50, "methodology": "9x 2026 EPS"},
            "bear": {"target_price": 28.0, "probability": 0.25, "methodology": "6x 2026 EPS"},
        },
        kpi_rows=[
            {"name": "NCO rate", "periods": {
                "2025_Q1": {"value": 1.91}, "2025_Q2": {"value": 1.82},
                "2025_Q3": {"value": 1.75},
            }},
            {"name": "CET1", "periods": {
                "2025_Q3": {"value": 9.9},
            }},
        ],
        decisions=[
            {"created_at": "2025-10-20T10:00:00Z", "action": "HOLD",
             "rationale": "Credit migration flat; no catalyst yet."},
        ],
    )
    assert isinstance(pdf, bytes)
    # PDF magic bytes
    assert pdf.startswith(b"%PDF-")
    # EOF marker
    assert b"%%EOF" in pdf[-64:]
    # Reasonable size — should be several KB minimum for the above content
    assert len(pdf) > 2000


def test_renderer_handles_sparse_briefing():
    """Only the headline — everything else empty. Should still produce a valid PDF."""
    pdf = render_briefing_pdf(
        company_name="TestCo",
        ticker="TST US",
        period="2025_Q1",
        briefing={"headline": "Just a header."},
    )
    assert pdf.startswith(b"%PDF-")
    assert b"%%EOF" in pdf[-64:]


def test_renderer_handles_all_empty_sections():
    """Extreme edge — caller passes only identity. Still renders cover + footer."""
    pdf = render_briefing_pdf(
        company_name="TestCo",
        ticker="TST US",
        period="2025_Q1",
    )
    assert pdf.startswith(b"%PDF-")
    assert b"%%EOF" in pdf[-64:]


def test_renderer_escapes_dangerous_html_in_content():
    """Values containing <, >, & must not break reportlab's mini-HTML parser."""
    pdf = render_briefing_pdf(
        company_name="TestCo",
        ticker="TST US",
        period="2025_Q1",
        briefing={
            "headline":    "Revenue <5% vs consensus; R&D up",
            "bottom_line": "Margin <20%; still investing",
        },
    )
    assert pdf.startswith(b"%PDF-")


def test_renderer_scenarios_missing_probability_does_not_crash():
    pdf = render_briefing_pdf(
        company_name="TestCo",
        ticker="TST US",
        period="2025_Q1",
        scenarios={"base": {"target_price": 10.0}},  # no probability
    )
    assert pdf.startswith(b"%PDF-")


def test_renderer_deterministic_when_generated_at_pinned():
    """Same inputs + same generated_at → byte-identical output (modulo
    reportlab's internal timestamps, which may shift — so we check size
    is reasonably close, not exact equality)."""
    ts = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)
    args = dict(
        company_name="TestCo", ticker="TST US", period="2025_Q1",
        briefing={"headline": "Deterministic test"},
        generated_at=ts,
    )
    pdf1 = render_briefing_pdf(**args)
    pdf2 = render_briefing_pdf(**args)
    # Reportlab embeds creation time — lengths should match within ±64 bytes
    assert abs(len(pdf1) - len(pdf2)) < 64
