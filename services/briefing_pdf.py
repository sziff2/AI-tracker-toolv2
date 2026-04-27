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
# Per-agent sections — mirror the UI cards in the Results tab so the
# downloaded PDF and the on-screen analysis surface the same content.
# ─────────────────────────────────────────────────────────────────

# Order + match list mirrors apps/ui/index.html FIN_TEMPLATE so the
# PDF and the Financial Summary panel render identical rows.
_FIN_TEMPLATE = [
    ("Income Statement", [
        {"label": "Revenue",          "match": ["revenue", "total revenue", "net sales", "total income", "turnover", "group revenue", "net interest income"], "type": "currency"},
        {"label": "  Change",         "derived": "yoy", "from": "Revenue",                                     "type": "pct"},
        {"label": "Gross profit",     "match": ["gross profit"],                                                "type": "currency"},
        {"label": "  Margin",         "derived": "ratio", "num": "Gross profit", "den": "Revenue",              "type": "pct"},
        {"label": "EBITDA",           "match": ["ebitda", "adjusted ebitda"],                                   "type": "currency"},
        {"label": "  Margin",         "derived": "ratio", "num": "EBITDA", "den": "Revenue",                    "type": "pct"},
        {"label": "Operating profit", "match": ["operating profit", "operating income", "ebit"],                "type": "currency"},
        {"label": "  Margin",         "derived": "ratio", "num": "Operating profit", "den": "Revenue",          "type": "pct"},
        {"label": "Net interest",     "match": ["net interest", "net interest expense"],                        "type": "currency"},
        {"label": "Net income",       "match": ["net income", "net profit", "profit for the period", "net income (attributable)"], "type": "currency"},
        {"label": "  Margin",         "derived": "ratio", "num": "Net income", "den": "Revenue",                "type": "pct"},
    ]),
    ("Cash Flow", [
        {"label": "OCF",              "match": ["operating cash flow", "cash from operations"],                 "type": "currency"},
        {"label": "Capex + leases",   "match": ["capex", "capital expenditure", "capital expenditures"],        "type": "currency"},
        {"label": "  % Sales",        "derived": "ratio", "num": "Capex + leases", "den": "Revenue",            "type": "pct"},
        {"label": "FCF",              "match": ["free cash flow", "fcf"],                                       "type": "currency"},
        {"label": "Dividend",         "match": ["dividends paid", "total dividends", "dividend"],               "type": "currency"},
        {"label": "Buyback (net)",    "match": ["share buyback", "buyback", "share repurchases"],               "type": "currency"},
    ]),
    ("Balance Sheet", [
        {"label": "Net Debt",         "match": ["net debt"],                                                     "type": "currency"},
        {"label": "  Net Debt/EBITDA","derived": "ratio", "num": "Net Debt", "den": "EBITDA",                    "type": "x"},
        {"label": "Equity",           "match": ["total equity", "shareholders equity", "stockholders equity"],   "type": "currency"},
    ]),
    ("Returns", [
        {"label": "ROE",              "match": ["roe", "return on equity"],                                      "type": "pct"},
        {"label": "ROIC",             "match": ["roic", "return on invested capital"],                           "type": "pct"},
    ]),
    ("Per Share", [
        {"label": "EPS",              "match": ["eps", "earnings per share", "eps (diluted)", "diluted eps"],   "type": "per_share"},
        {"label": "DPS",              "match": ["dps", "dividend per share", "dividends per share"],             "type": "per_share"},
        {"label": "Book value/sh",    "match": ["book value per share"],                                          "type": "per_share"},
    ]),
]


def _match_key(name: str) -> str:
    """Mirror the UI _matchKey: lowercase + collapse underscores/hyphens
    so snake_case, TitleCase, hyphenated all collapse to one form."""
    if not name:
        return ""
    s = str(name).lower().replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def _resolve_template_row(tpl_row: dict, index: dict, matched: dict) -> dict | None:
    """Resolve a template row to a /results-summary row, computing
    derived rows (yoy, ratio) from prior matches in `matched`."""
    if "derived" in tpl_row:
        kind = tpl_row["derived"]
        if kind == "yoy":
            num = matched.get(tpl_row.get("from"))
            if num and num.get("actual") is not None and num.get("prior_value"):
                v = num["prior_value"]
                if v != 0:
                    return {"actual": (num["actual"] - v) / abs(v) * 100, "unit": "%"}
        elif kind == "ratio":
            num = matched.get(tpl_row.get("num"))
            den = matched.get(tpl_row.get("den"))
            if num and den and num.get("actual") is not None and den.get("actual"):
                ratio = num["actual"] / den["actual"]
                if tpl_row.get("type") == "pct":
                    ratio *= 100
                return {"actual": ratio, "unit": "%" if tpl_row.get("type") == "pct" else ("x" if tpl_row.get("type") == "x" else None)}
        return None
    # Direct match
    for term in tpl_row.get("match", []):
        k = _match_key(term)
        if k in index:
            return index[k]
    # Substring fallback (both directions)
    for term in tpl_row.get("match", []):
        k = _match_key(term)
        for key in index.keys():
            if key == k or k in key or key in k:
                return index[key]
    return None


def _fmt_value(v, unit: str | None) -> str:
    if v is None:
        return "—"
    u = (unit or "").upper()
    try:
        n = float(v)
    except (TypeError, ValueError):
        return _escape(str(v))
    if u == "%":
        return f"{n:+.1f}%" if abs(n) < 1000 else f"{n:.0f}%"
    if u == "X":
        return f"{n:.2f}x"
    if u == "BPS":
        return f"{n:.0f}bps"
    if abs(n) < 1:
        return f"{n:.3f}"
    if abs(n) < 10:
        return f"{n:.2f}"
    return f"{n:,.0f}"


def _financial_summary_section(story: list, styles: dict, fs: dict) -> None:
    """Analyst-template financial table (IS / CF / BS / Returns / Per Share)
    mirroring the UI's Financial Summary panel. Renders only rows where the
    extractor matched a metric so the output stays compact."""
    rows = (fs or {}).get("rows") or []
    if not rows:
        return
    # Index by canonical match key
    index: dict[str, dict] = {}
    for r in rows:
        k = _match_key(r.get("metric_name"))
        if k and k not in index:
            index[k] = r

    matched: dict[str, dict] = {}
    sections: list[tuple[str, list[tuple[dict, dict]]]] = []
    for sec_title, sec_rows in _FIN_TEMPLATE:
        lines: list[tuple[dict, dict]] = []
        for tpl in sec_rows:
            data = _resolve_template_row(tpl, index, matched)
            if not data:
                continue
            if "derived" not in tpl:
                matched[tpl["label"]] = data
            lines.append((tpl, data))
        if lines:
            sections.append((sec_title, lines))

    if not sections:
        return

    # Detect dominant currency unit for the FX header
    unit_votes: dict[str, int] = {}
    for r in matched.values():
        u = (r.get("unit") or "").upper()
        if not u or u in ("%", "X", "BPS"):
            continue
        unit_votes[u] = unit_votes.get(u, 0) + 1
    fx = max(unit_votes, key=unit_votes.get) if unit_votes else ""
    fx_display = fx.replace("_M", "m").replace("_B", "b").replace("_", "") if fx else ""

    title = "Financial Summary"
    if fx_display:
        title += f"  ·  FX: {fx_display}"
    if fs.get("prior_period"):
        title += f"  ·  YoY vs {fs['prior_period']}"
    story.append(Paragraph(title, styles["h2"]))

    table_rows: list[list] = []
    for sec_title, lines in sections:
        table_rows.append([Paragraph(f"<b>{_escape(sec_title)}</b>", styles["small"]), ""])
        for tpl, data in lines:
            label = tpl["label"]
            indent = label.startswith("  ")
            label_clean = label.lstrip()
            actual = data.get("actual")
            unit = data.get("unit") or ""
            # Inline vs-cons badge if present
            bm = data.get("beat_miss_pct")
            vs = ""
            if data.get("has_consensus") and bm is not None:
                sign = "+" if bm > 0 else ""
                vs = f"  ({sign}{bm:.1f}% vs cons)"
            label_para = Paragraph(
                f"{'&nbsp;&nbsp;' if indent else ''}{_escape(label_clean)}",
                styles["small"] if indent else styles["body"],
            )
            value_str = _fmt_value(actual, unit)
            value_para = Paragraph(
                f"<font face='Helvetica-Bold'>{_escape(value_str)}</font>{_escape(vs)}",
                styles["small"] if indent else styles["body"],
            )
            table_rows.append([label_para, value_para])

    t = Table(table_rows, colWidths=[110*mm, 60*mm])
    t.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("ALIGN",         (1, 0), (1, -1), "RIGHT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW",     (0, 0), (-1, 0), 0.4, colors.HexColor("#a88932")),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def _bullets(story: list, styles: dict, items, prefix: str = "•") -> None:
    if not items:
        return
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return
    for it in items:
        if isinstance(it, dict):
            line = " · ".join(
                f"<b>{_escape(k)}:</b> {_escape(v)}"
                for k, v in it.items() if v not in (None, "", [], {})
            )
        else:
            line = _escape(str(it))
        if line.strip():
            story.append(Paragraph(f"{prefix} {line}", styles["body"]))


def _financial_analyst_section(story: list, styles: dict, fa: dict) -> None:
    if not fa:
        return
    story.append(Paragraph("Financial Analyst Assessment", styles["h2"]))

    grade = fa.get("overall_grade")
    direction = fa.get("thesis_direction") or ""
    benchmark = fa.get("expectations_benchmark") or ""
    grade_label = ["", "Significant miss", "Miss", "In line", "Beat", "Significant beat"]
    badge_bits: list[str] = []
    if grade:
        try:
            badge_bits.append(f"Grade {int(grade)}/5 — {grade_label[int(grade)]}")
        except (ValueError, IndexError, TypeError):
            badge_bits.append(f"Grade: {grade}")
    if benchmark:
        badge_bits.append(f"vs {benchmark}")
    if direction:
        badge_bits.append(f"Thesis: {direction}")
    if badge_bits:
        story.append(Paragraph(_escape(" · ".join(badge_bits)), styles["accent"]))

    sections = [
        ("Revenue",            "revenue_assessment"),
        ("Margins",            "margin_analysis"),
        ("Cash Flow",          "cash_flow_quality"),
        ("Balance Sheet",      "balance_sheet_flags"),
        ("Segments",           "segment_commentary"),
        ("Guidance",           "guidance_quality"),
        ("Management Signals", "management_signals"),
    ]
    for label, key in sections:
        v = fa.get(key)
        if v:
            story.append(Paragraph(label, styles["h3"]))
            story.append(Paragraph(_escape(_compact(v)), styles["body"]))

    if fa.get("tracked_kpi_assessment"):
        story.append(Paragraph("Tracked KPI Scores", styles["h3"]))
        for kpi in fa["tracked_kpi_assessment"][:15]:
            if not isinstance(kpi, dict):
                continue
            name = kpi.get("kpi_name", "")
            score = kpi.get("score", "")
            value = kpi.get("value", "")
            comment = kpi.get("comment", "") or kpi.get("rationale", "")
            line = f"<b>{_escape(name)}:</b> {_escape(score)}/5"
            if value:
                line += f" — {_escape(value)}"
            if comment:
                line += f" — {_escape(comment)}"
            story.append(Paragraph(line, styles["body"]))

    if fa.get("key_surprises"):
        story.append(Paragraph("Key Surprises", styles["h3"]))
        _bullets(story, styles, fa["key_surprises"])

    if fa.get("key_assumptions"):
        story.append(Paragraph("Key Assumptions", styles["h3"]))
        for a in fa["key_assumptions"][:10]:
            if isinstance(a, dict):
                ass = a.get("assumption", "")
                prob = a.get("probability")
                prior = a.get("prior")
                arrow = a.get("direction", "")
                bits = [_escape(ass)]
                if prob is not None:
                    try:
                        bits.append(f"{int(float(prob)*100)}%")
                    except (TypeError, ValueError):
                        pass
                if prior is not None:
                    try:
                        bits.append(f"(prior {int(float(prior)*100)}%)")
                    except (TypeError, ValueError):
                        pass
                if arrow:
                    bits.append(f"[{arrow}]")
                story.append(Paragraph("• " + " — ".join(bits), styles["body"]))


def _bull_bear_section(story: list, styles: dict, bull: dict, bear: dict) -> None:
    if not bull and not bear:
        return
    if bull:
        story.append(Paragraph("Bull Case", styles["h2"]))
        if bull.get("bull_thesis"):
            story.append(Paragraph(_escape(_compact(bull["bull_thesis"])), styles["body"]))
        if bull.get("upside_catalysts"):
            story.append(Paragraph("Upside catalysts", styles["h3"]))
            _bullets(story, styles, bull["upside_catalysts"])
        up = bull.get("upside_scenario")
        if isinstance(up, dict):
            line = []
            if up.get("implied_return"):
                line.append(f"<b>Implied return:</b> {_escape(up['implied_return'])}")
            if up.get("description"):
                line.append(_escape(up["description"]))
            if up.get("timeline"):
                line.append(f"<i>{_escape(up['timeline'])}</i>")
            if up.get("key_trigger"):
                line.append(f"Trigger: {_escape(up['key_trigger'])}")
            if line:
                story.append(Paragraph("Upside scenario", styles["h3"]))
                story.append(Paragraph(" — ".join(line), styles["body"]))
        if bull.get("what_would_make_you_wrong"):
            story.append(Paragraph("What would make you wrong", styles["h3"]))
            _bullets(story, styles, bull["what_would_make_you_wrong"])

    if bear:
        story.append(Paragraph("Bear Case", styles["h2"]))
        if bear.get("bear_thesis"):
            story.append(Paragraph(_escape(_compact(bear["bear_thesis"])), styles["body"]))
        if bear.get("key_risks"):
            story.append(Paragraph("Key risks", styles["h3"]))
            _bullets(story, styles, bear["key_risks"])
        dn = bear.get("downside_scenario")
        if isinstance(dn, dict):
            line = []
            if dn.get("implied_return"):
                line.append(f"<b>Implied return:</b> {_escape(dn['implied_return'])}")
            if dn.get("description"):
                line.append(_escape(dn["description"]))
            if dn.get("timeline"):
                line.append(f"<i>{_escape(dn['timeline'])}</i>")
            if dn.get("key_trigger"):
                line.append(f"Trigger: {_escape(dn['key_trigger'])}")
            if line:
                story.append(Paragraph("Downside scenario", styles["h3"]))
                story.append(Paragraph(" — ".join(line), styles["body"]))


def _debate_section(story: list, styles: dict, dbt: dict) -> None:
    if not dbt:
        return
    story.append(Paragraph("Debate Verdict", styles["h2"]))
    verdict = dbt.get("verdict") or dbt.get("recommendation")
    if verdict:
        story.append(Paragraph(_escape(verdict), styles["accent"]))
    if dbt.get("debate_summary"):
        story.append(Paragraph(_escape(_compact(dbt["debate_summary"])), styles["body"]))

    # Probability split
    probs = []
    for k in ("bull_probability", "base_probability", "bear_probability"):
        if dbt.get(k) is not None:
            try:
                probs.append(f"{k.split('_')[0].capitalize()} {int(float(dbt[k])*100)}%")
            except (TypeError, ValueError):
                pass
    if probs:
        story.append(Paragraph("Probability split", styles["h3"]))
        story.append(Paragraph(" · ".join(probs), styles["body"]))

    base = dbt.get("base_scenario")
    if isinstance(base, dict) and base.get("description"):
        story.append(Paragraph("Base case", styles["h3"]))
        story.append(Paragraph(_escape(_compact(base["description"])), styles["body"]))


def _transcript_section(story: list, styles: dict, tr: dict) -> None:
    if not tr:
        return
    story.append(Paragraph("Transcript Analysis", styles["h2"]))
    tone = tr.get("management_tone")
    if isinstance(tone, dict):
        bits = []
        if tone.get("overall") is not None:
            bits.append(f"<b>Overall tone:</b> {_escape(tone['overall'])}/5")
        if tone.get("confidence_level"):
            bits.append(_escape(tone["confidence_level"]))
        if bits:
            story.append(Paragraph(" — ".join(bits), styles["body"]))

    if tr.get("guidance_statements"):
        story.append(Paragraph("Guidance statements", styles["h3"]))
        for g in tr["guidance_statements"][:10]:
            if not isinstance(g, dict):
                continue
            d = g.get("direction", "")
            m = g.get("metric", "")
            s = g.get("statement", "")
            story.append(Paragraph(
                f"<b>{_escape(d)}</b> {_escape(m)}: {_escape(s)}",
                styles["body"],
            ))

    if tr.get("evasion_signals"):
        story.append(Paragraph("Evasion signals", styles["h3"]))
        for e in tr["evasion_signals"][:8]:
            if not isinstance(e, dict):
                continue
            q = e.get("what_was_asked", "") or e.get("question", "")
            d = e.get("how_they_deflected", "") or e.get("deflection", "")
            line = ""
            if q:
                line += f"<b>Q:</b> {_escape(q)}<br/>"
            if d:
                line += f"<b>Deflection:</b> {_escape(d)}"
            if line:
                story.append(Paragraph(line, styles["body"]))

    if tr.get("key_quotes"):
        story.append(Paragraph("Key quotes", styles["h3"]))
        for q in tr["key_quotes"][:5]:
            if not isinstance(q, dict):
                continue
            quote = q.get("quote", "")
            speaker = q.get("speaker", "")
            if quote:
                story.append(Paragraph(
                    f"<i>&ldquo;{_escape(quote)}&rdquo;</i> — {_escape(speaker)}",
                    styles["body"],
                ))


def _presentation_section(story: list, styles: dict, pr: dict) -> None:
    if not pr:
        return
    story.append(Paragraph("Presentation Analysis", styles["h2"]))
    if pr.get("management_message"):
        story.append(Paragraph(_escape(_compact(pr["management_message"])), styles["body"]))
    if pr.get("notable_omissions"):
        story.append(Paragraph("Notable omissions", styles["h3"]))
        for om in pr["notable_omissions"][:8]:
            if isinstance(om, dict):
                topic = om.get("topic", "")
                why = om.get("why_notable", "")
                story.append(Paragraph(f"• <b>{_escape(topic)}</b> — {_escape(why)}", styles["body"]))
            else:
                story.append(Paragraph(f"• {_escape(om)}", styles["body"]))
    if pr.get("new_metrics"):
        story.append(Paragraph("New metrics introduced", styles["h3"]))
        for nm in pr["new_metrics"][:8]:
            if isinstance(nm, dict):
                metric = nm.get("metric", "")
                why = nm.get("why_introduced", "")
                story.append(Paragraph(f"• <b>{_escape(metric)}</b> — {_escape(why)}", styles["body"]))
            else:
                story.append(Paragraph(f"• {_escape(nm)}", styles["body"]))


def _competitive_section(story: list, styles: dict, cp: dict) -> None:
    if not cp:
        return
    story.append(Paragraph("Competitive Positioning", styles["h2"]))
    for label, key in [
        ("Position vs peers",  "position_summary"),
        ("Strengths",          "strengths"),
        ("Vulnerabilities",    "vulnerabilities"),
        ("Moat assessment",    "moat_assessment"),
    ]:
        v = cp.get(key)
        if v:
            story.append(Paragraph(label, styles["h3"]))
            if isinstance(v, list):
                _bullets(story, styles, v)
            else:
                story.append(Paragraph(_escape(_compact(v)), styles["body"]))


def _guidance_tracker_section(story: list, styles: dict, gt: dict) -> None:
    if not gt:
        return
    story.append(Paragraph("Guidance Tracker", styles["h2"]))
    bits = []
    if gt.get("overall_signal"):
        bits.append(f"<b>Overall signal:</b> {_escape(gt['overall_signal'])}")
    if gt.get("track_record_signal"):
        bits.append(f"<b>Track record:</b> {_escape(gt['track_record_signal'])}")
    if bits:
        story.append(Paragraph(" — ".join(bits), styles["body"]))
    if gt.get("methodology_changes"):
        story.append(Paragraph("Methodology changes", styles["h3"]))
        _bullets(story, styles, gt["methodology_changes"])
    if gt.get("notable_walkbacks"):
        story.append(Paragraph("Notable walkbacks", styles["h3"]))
        _bullets(story, styles, gt["notable_walkbacks"])


def _qc_section(story: list, styles: dict, qc: dict) -> None:
    if not qc:
        return
    story.append(Paragraph("Quality Control", styles["h2"]))
    if qc.get("recommendation"):
        story.append(Paragraph(_escape(qc["recommendation"]), styles["accent"]))
    scores: list[str] = []
    for k in ("overall_score", "completeness_score", "consistency_score", "evidence_score", "calibration_score"):
        if qc.get(k) is not None:
            label = k.replace("_score", "").capitalize()
            try:
                scores.append(f"{label} {float(qc[k]):.1f}/5")
            except (TypeError, ValueError):
                pass
    if scores:
        story.append(Paragraph(" · ".join(scores), styles["body"]))
    if qc.get("flags"):
        story.append(Paragraph("Flags", styles["h3"]))
        _bullets(story, styles, qc["flags"])


def _compact(v, max_len: int = 20000) -> str:
    """Coerce anything reasonable to a single string for the PDF.

    Default cap of 20000 chars is effectively no truncation for
    narrative agent outputs — bull_thesis / bear_thesis / debate_summary
    / revenue_assessment etc. are 2-3 paragraph writeups that can run
    3-6K chars. The previous 2500 cap was clipping them mid-sentence
    ("The bull case was stro..."). reportlab Paragraph wraps long
    text fine, so the only reason to cap is for sanity against
    runaway LLM output (200K+).

    Nested dict / list serialisation passes a smaller cap (400) per
    sub-value so a single rich-dict field doesn't dominate."""
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
    elif isinstance(v, dict):
        s = " · ".join(
            f"{k}: {_compact(val, 400)}" for k, val in v.items()
            if val not in (None, "", [], {})
        )
    elif isinstance(v, list):
        s = "; ".join(_compact(x, 400) for x in v if x not in (None, ""))
    else:
        s = str(v)
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


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
    agent_outs: dict[str, dict] | None = None,
    fin_summary: dict | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Render a briefing PDF and return the bytes.

    `briefing` is the synthesis JSON (headline / bottom_line / etc.).
    `thesis` is the active ThesisVersion serialised to dict.
    `scenarios` is a dict with optional bull/base/bear sub-dicts.
    `kpi_rows` is a list of {name, periods: {period_label: {value, ...}}}.
    `decisions` is the recent decision log (list of dicts).
    `agent_outs` is the raw {agent_id: output_dict} map from agent_outputs
        — used to render per-agent sections that mirror the UI cards.
    `fin_summary` is the /results-summary payload (rows + prior_period etc.)
        — drives the analyst-template Financial Summary table at the top.

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
    agents = agent_outs or {}

    _header_section(story, styles, company_name=company_name,
                    ticker=ticker, period=period, generated_at=now)
    # Order mirrors the UI: Financial Summary on top, then per-agent
    # cards, then the formal thesis + scenarios + KPI + decisions tail.
    _financial_summary_section(story, styles, fin_summary or {})
    _briefing_section(story, styles, briefing or {})
    _financial_analyst_section(story, styles, agents.get("financial_analyst") or {})
    _bull_bear_section(story, styles, agents.get("bull_case") or {}, agents.get("bear_case") or {})
    _debate_section(story, styles, agents.get("debate_agent") or {})
    _transcript_section(story, styles, agents.get("transcript_deep_dive") or {})
    _presentation_section(story, styles, agents.get("presentation_analysis") or {})
    _competitive_section(story, styles, agents.get("competitive_positioning") or {})
    _guidance_tracker_section(story, styles, agents.get("guidance_tracker") or {})
    _qc_section(story, styles, agents.get("quality_control") or {})
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
