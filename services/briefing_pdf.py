"""
Briefing PDF renderer (Tier 3.3) — produces a one-click-downloadable
investment-memo style PDF of a company's latest analysis.

Uses reportlab rather than WeasyPrint:
  - Pure Python, no Cairo / Pango / GDK-PixBuf / fontconfig native deps
  - Ships in the Railway web image with zero Dockerfile changes
  - Fonts bundled (Helvetica / Times) so the output is stable across
    build environments

Layout is deliberately clean-clinical — think PM-ready investment memo
rather than marketing piece. Section order mirrors the analyst's
mental model: headline first, then thesis + decision log for context,
then scenarios (bull / base / bear), then KPIs + raw agent outputs
tucked at the bottom for reference.

The renderer is pure — takes a dict of content, returns PDF bytes.
DB access lives in the API route (apps/api/routes/briefing.py) so the
renderer is trivially testable.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    """ParagraphStyles — one palette for the whole document."""
    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontSize=20, leading=24, spaceAfter=6,
            textColor=colors.HexColor("#0f1117"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontSize=13, leading=17, spaceBefore=18, spaceAfter=6,
            textColor=colors.HexColor("#a88932"),  # accent gold
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontSize=11, leading=15, spaceBefore=10, spaceAfter=4,
            textColor=colors.HexColor("#2a2d37"),
        ),
        "meta": ParagraphStyle(
            "meta", parent=base["Normal"],
            fontSize=9, leading=12,
            textColor=colors.HexColor("#5a5d68"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10, leading=14, spaceAfter=6,
        ),
        "body_bold": ParagraphStyle(
            "body_bold", parent=base["Normal"],
            fontSize=10, leading=14, spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"],
            fontSize=8, leading=11,
            textColor=colors.HexColor("#5a5d68"),
        ),
        "accent": ParagraphStyle(
            "accent", parent=base["Normal"],
            fontSize=11, leading=15, spaceAfter=6,
            textColor=colors.HexColor("#c9a960"),
            fontName="Helvetica-Bold",
        ),
    }
    return styles


def _escape(text: Any) -> str:
    """Escape text for reportlab Paragraph (it uses mini-HTML)."""
    if text is None:
        return ""
    s = str(text)
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


# ─────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────

def _header_section(story: list, styles: dict, *, company_name: str,
                    ticker: str, period: str, generated_at: datetime) -> None:
    story.append(Paragraph(_escape(company_name), styles["h1"]))
    story.append(Paragraph(
        f"{_escape(ticker)} &nbsp;&middot;&nbsp; {_escape(period)}",
        styles["meta"],
    ))
    story.append(Paragraph(
        f"Investment briefing generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        styles["small"],
    ))
    story.append(Spacer(1, 10))


def _thesis_section(story: list, styles: dict, thesis: dict | None) -> None:
    if not thesis:
        return
    story.append(Paragraph("Investment Thesis", styles["h2"]))
    core = thesis.get("core_thesis") or thesis.get("text") or ""
    if core:
        story.append(Paragraph(_escape(core), styles["body"]))
    risks = thesis.get("key_risks")
    if risks:
        story.append(Paragraph("Key risks", styles["h3"]))
        story.append(Paragraph(_escape(risks), styles["body"]))
    val = thesis.get("valuation_framework")
    if val:
        story.append(Paragraph("Valuation framework", styles["h3"]))
        story.append(Paragraph(_escape(val), styles["body"]))


def _briefing_section(story: list, styles: dict, briefing: dict) -> None:
    """The synthesis output — headline + what_happened + management_message
    + thesis_impact + bottom_line. Keys are optional; skip when absent."""
    if not briefing:
        return
    story.append(Paragraph("Bottom line", styles["h2"]))
    if briefing.get("headline"):
        story.append(Paragraph(_escape(briefing["headline"]), styles["accent"]))
    for label, key in [
        ("Bottom line",        "bottom_line"),
        ("What happened",      "what_happened"),
        ("Management message", "management_message"),
        ("Thesis impact",      "thesis_impact"),
        ("What changed",       "what_changed"),
        ("Thesis status",      "thesis_status"),
    ]:
        val = briefing.get(key)
        if val:
            story.append(Paragraph(label, styles["h3"]))
            story.append(Paragraph(_escape(val), styles["body"]))


def _scenarios_section(story: list, styles: dict, scenarios: dict | None) -> None:
    """Bull / base / bear prices + probabilities as a compact table."""
    if not scenarios:
        return
    story.append(Paragraph("Scenarios", styles["h2"]))

    rows = [["Scenario", "Target price", "Probability", "Implied return"]]
    for key in ("bull", "base", "bear"):
        s = scenarios.get(key)
        if not isinstance(s, dict):
            continue
        target = s.get("target_price") or s.get("price")
        prob = s.get("probability")
        ret = s.get("implied_return")
        rows.append([
            key.capitalize(),
            _fmt(target, prefix=""),
            _fmt_pct(prob) if isinstance(prob, (int, float)) else str(prob or ""),
            _fmt_pct(ret) if isinstance(ret, (int, float)) else str(ret or ""),
        ])
    if len(rows) < 2:
        return
    t = Table(rows, colWidths=[30*mm, 35*mm, 30*mm, 35*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12141a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.HexColor("#e2e4ea")),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#2a2d37")),
        ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # Add methodology strings if present — underneath the table so
    # analysts reading the scenario see the reasoning directly.
    for key in ("bull", "base", "bear"):
        s = scenarios.get(key)
        if isinstance(s, dict) and s.get("methodology"):
            story.append(Paragraph(
                f"<b>{key.capitalize()}:</b> {_escape(s['methodology'])}",
                styles["small"],
            ))


def _kpi_section(story: list, styles: dict, kpi_rows: list[dict]) -> None:
    if not kpi_rows:
        return
    story.append(Paragraph("Tracked KPIs", styles["h2"]))

    # Collect periods (columns) from the supplied rows. Limit to the
    # 4 most recent periods so the table fits comfortably on A4.
    all_periods: set[str] = set()
    for row in kpi_rows:
        all_periods.update((row.get("periods") or {}).keys())
    periods_sorted = sorted(all_periods, reverse=True)[:4]
    periods_sorted.reverse()  # chronological left-to-right

    if not periods_sorted:
        return

    header = ["KPI"] + periods_sorted
    table_rows = [header]
    for row in kpi_rows[:20]:  # cap at 20 KPIs per page
        name = row.get("name") or row.get("kpi") or ""
        cells = [_escape(name)]
        for p in periods_sorted:
            pval = (row.get("periods") or {}).get(p) or {}
            v = pval.get("value") or pval.get("value_text") or ""
            cells.append(_fmt(v, prefix=""))
        table_rows.append(cells)
    t = Table(table_rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12141a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.HexColor("#e2e4ea")),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#2a2d37")),
        ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
    ]))
    story.append(t)


def _decisions_section(story: list, styles: dict, decisions: list[dict]) -> None:
    if not decisions:
        return
    story.append(Paragraph("Recent decision log", styles["h2"]))
    # Show the 5 most recent decisions
    for d in decisions[:5]:
        ts = d.get("created_at") or d.get("date") or ""
        if isinstance(ts, str) and "T" in ts:
            ts = ts.split("T")[0]
        action = d.get("action") or d.get("decision") or ""
        rationale = d.get("rationale") or d.get("note") or ""
        line = f"<b>{_escape(ts)}</b> — {_escape(action)}"
        if rationale:
            line += f": {_escape(rationale)}"
        story.append(Paragraph(line, styles["body"]))


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def render_briefing_pdf(
    *,
    company_name: str,
    ticker: str,
    period: str,
    briefing: dict | None = None,
    thesis: dict | None = None,
    scenarios: dict | None = None,
    kpi_rows: list[dict] | None = None,
    decisions: list[dict] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Render a briefing PDF and return the bytes.

    `briefing` is the synthesis JSON (headline / bottom_line / etc.).
    `thesis` is the active ThesisVersion serialised to dict.
    `scenarios` is a dict with optional bull/base/bear sub-dicts.
    `kpi_rows` is a list of {name, periods: {period_label: {value, ...}}}.
    `decisions` is the recent decision log (list of dicts).

    All sections are optional — missing data is skipped silently.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=15*mm, bottomMargin=15*mm,
        leftMargin=18*mm, rightMargin=18*mm,
        title=f"{ticker} {period} Briefing",
        author="Oldfield Partners",
    )

    styles = _build_styles()
    story: list = []
    now = generated_at or datetime.now(timezone.utc)

    _header_section(story, styles, company_name=company_name,
                    ticker=ticker, period=period, generated_at=now)
    _briefing_section(story, styles, briefing or {})
    _scenarios_section(story, styles, scenarios)
    _thesis_section(story, styles, thesis)
    _kpi_section(story, styles, kpi_rows or [])
    _decisions_section(story, styles, decisions or [])

    # Footer on every page
    def _footer(canvas, _doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#5a5d68"))
        canvas.drawString(18*mm, 8*mm,
                          f"{ticker} — {period} — Oldfield Partners research")
        canvas.drawRightString(A4[0] - 18*mm, 8*mm,
                               f"Page {_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _fmt(v: Any, *, prefix: str = "$") -> str:
    if v is None or v == "":
        return "—"
    try:
        num = float(v)
        # Pick a sensible precision
        if abs(num) >= 1000:
            return f"{prefix}{num:,.0f}"
        if abs(num) >= 10:
            return f"{prefix}{num:,.1f}"
        return f"{prefix}{num:.2f}"
    except (TypeError, ValueError):
        return _escape(str(v))


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        n = float(v)
        # Heuristic: probabilities (0-1) vs already-percent (e.g. 15)
        if 0 <= n <= 1:
            return f"{n*100:.0f}%"
        return f"{n:+.1f}%"
    except (TypeError, ValueError):
        return str(v)
