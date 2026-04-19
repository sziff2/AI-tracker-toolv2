"""
Weekly Harvest Report — generation, storage, and Teams notification.

Called by Celery Beat (weekly Monday 06:00 UTC) or manually via API.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from configs.settings import settings

logger = logging.getLogger(__name__)


async def run_weekly_harvest() -> dict:
    """Run a cost-controlled harvest (no LLM) and return enriched result.
    Routed through the Ingestion Orchestrator so Document Triage runs on
    each candidate and future source-quality logic has a single hook point.
    """
    from agents.ingestion.orchestrator import IngestionOrchestrator
    return await IngestionOrchestrator().run_scheduled_scan(
        tier="portfolio", skip_llm=True
    )


async def save_report(harvest_result: dict, trigger: str = "auto_weekly") -> str:
    """Persist a HarvestReport row and return its ID."""
    from apps.api.database import AsyncSessionLocal
    from sqlalchemy import text

    report_id = str(uuid.uuid4())
    summary = {
        "new": harvest_result.get("new", 0),
        "skipped": harvest_result.get("skipped", 0),
        "failed": harvest_result.get("failed", 0),
    }
    details = harvest_result.get("details", [])

    async with AsyncSessionLocal() as db:
        await db.execute(text("""
            INSERT INTO harvest_reports (id, run_at, "trigger", summary_json, details_json, teams_sent)
            VALUES (:id, :run_at, :trigger, :summary, :details, false)
        """), {
            "id": report_id,
            "run_at": datetime.now(timezone.utc),
            "trigger": trigger,
            "summary": json.dumps(summary),
            "details": json.dumps(details),
        })
        await db.commit()

    logger.info("[REPORT] Saved harvest report %s (trigger=%s)", report_id, trigger)
    return report_id


def _classify_details(details: list[dict]) -> dict:
    """Classify companies into found / no_new / errored / unconfigured."""
    found = []
    no_new = []
    errored = []
    unconfigured = []

    for d in details:
        if d.get("errors"):
            errored.append(d)
        elif not d.get("sources_tried"):
            unconfigured.append(d)
        elif d.get("candidates_found", 0) > 0:
            found.append(d)
        else:
            no_new.append(d)

    return {
        "found": found,
        "no_new": no_new,
        "errored": errored,
        "unconfigured": unconfigured,
    }


# ─────────────────────────────────────────────────────────────────
# Triage + Coverage Monitor stats for the weekly Teams report
# ─────────────────────────────────────────────────────────────────

async def collect_triage_stats(since_ts: datetime) -> dict:
    """Count Document Triage decisions made during this harvest window.
    Returns {total, auto_ingested, needs_review, skipped_by_triage}.
    Safe fallback: returns zeros if the query fails — never blocks the
    weekly report."""
    from apps.api.database import AsyncSessionLocal
    from apps.api.models import IngestionTriage
    from sqlalchemy import and_, func, select as sa_select

    stats = {"total": 0, "auto_ingested": 0, "needs_review": 0, "skipped_by_triage": 0}
    try:
        async with AsyncSessionLocal() as db:
            total_q = await db.execute(
                sa_select(func.count(IngestionTriage.id))
                .where(IngestionTriage.created_at >= since_ts)
            )
            stats["total"] = total_q.scalar() or 0

            auto_q = await db.execute(
                sa_select(func.count(IngestionTriage.id))
                .where(and_(
                    IngestionTriage.created_at >= since_ts,
                    IngestionTriage.was_ingested == True,  # noqa: E712
                ))
            )
            stats["auto_ingested"] = auto_q.scalar() or 0

            review_q = await db.execute(
                sa_select(func.count(IngestionTriage.id))
                .where(and_(
                    IngestionTriage.created_at >= since_ts,
                    IngestionTriage.needs_review == True,  # noqa: E712
                    IngestionTriage.was_ingested == False,  # noqa: E712
                ))
            )
            stats["needs_review"] = review_q.scalar() or 0

            skip_q = await db.execute(
                sa_select(func.count(IngestionTriage.id))
                .where(and_(
                    IngestionTriage.created_at >= since_ts,
                    IngestionTriage.priority == "skip",
                ))
            )
            stats["skipped_by_triage"] = skip_q.scalar() or 0
    except Exception as exc:
        logger.warning("[REPORT] collect_triage_stats failed: %s", exc)

    return stats


async def collect_coverage_monitor_stats(lookback_days: int = 7) -> dict:
    """Live Coverage Monitor snapshot + last-N-days auto-rescan activity.
    Returns {total_gaps, by_severity, rescans, rescan_successes,
    rescan_errors, lookback_days}. Safe fallback on query failure."""
    from apps.api.database import AsyncSessionLocal
    from apps.api.models import CoverageRescanLog
    from sqlalchemy import and_, func, select as sa_select

    stats: dict = {
        "total_gaps":      0,
        "by_severity":     {"warning": 0, "overdue": 0, "critical": 0, "source_broken": 0},
        "rescans":         0,
        "rescan_successes": 0,
        "rescan_errors":   0,
        "lookback_days":   lookback_days,
    }
    try:
        # Live gap snapshot (no auto-trigger — we're just reporting).
        from agents.ingestion.coverage_monitor import CoverageMonitor
        cm_result = await CoverageMonitor().run_daily_check(auto_trigger=False)
        stats["total_gaps"] = cm_result.gaps_found
        for g in cm_result.gap_details:
            sev = g.get("severity", "warning")
            if sev in stats["by_severity"]:
                stats["by_severity"][sev] += 1
    except Exception as exc:
        logger.warning("[REPORT] coverage gap snapshot failed: %s", exc)

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        async with AsyncSessionLocal() as db:
            total_q = await db.execute(
                sa_select(func.count(CoverageRescanLog.id))
                .where(and_(
                    CoverageRescanLog.triggered_at >= cutoff,
                    CoverageRescanLog.triggered_by == "auto",
                ))
            )
            stats["rescans"] = total_q.scalar() or 0

            succ_q = await db.execute(
                sa_select(func.count(CoverageRescanLog.id))
                .where(and_(
                    CoverageRescanLog.triggered_at >= cutoff,
                    CoverageRescanLog.triggered_by == "auto",
                    CoverageRescanLog.result == "success",
                ))
            )
            stats["rescan_successes"] = succ_q.scalar() or 0

            err_q = await db.execute(
                sa_select(func.count(CoverageRescanLog.id))
                .where(and_(
                    CoverageRescanLog.triggered_at >= cutoff,
                    CoverageRescanLog.triggered_by == "auto",
                    CoverageRescanLog.result == "error",
                ))
            )
            stats["rescan_errors"] = err_q.scalar() or 0
    except Exception as exc:
        logger.warning("[REPORT] rescan log query failed: %s", exc)

    return stats


def format_teams_message(harvest_result: dict) -> dict:
    """Build a Microsoft Teams Adaptive Card payload."""
    summary = {
        "new": harvest_result.get("new", 0),
        "skipped": harvest_result.get("skipped", 0),
        "failed": harvest_result.get("failed", 0),
    }
    details = harvest_result.get("details", [])
    classified = _classify_details(details)
    now = datetime.now(timezone.utc).strftime("%A %d %b %Y")

    # Build text sections
    lines = []
    lines.append(f"**Weekly Harvest Report — {now}**\n")
    lines.append(
        f"**Summary:** {summary['new']} new documents | "
        f"{summary['skipped']} skipped (already seen) | "
        f"{summary['failed']} failed\n"
    )

    if classified["found"]:
        lines.append("**New documents found:**")
        for d in sorted(classified["found"], key=lambda x: x["ticker"]):
            sources = ", ".join(d["sources_tried"])
            lines.append(f"- {d['ticker']}: {d['candidates_found']} ({sources})")
        lines.append("")

    if classified["errored"]:
        lines.append("**Errors:**")
        for d in sorted(classified["errored"], key=lambda x: x["ticker"]):
            for err in d["errors"]:
                lines.append(f"- {d['ticker']}: {err}")
        lines.append("")

    if classified["unconfigured"]:
        lines.append("**Unconfigured (no source set up):**")
        for d in sorted(classified["unconfigured"], key=lambda x: x["ticker"]):
            lines.append(f"- {d['ticker']} ({d['name']})")
        lines.append("")

    gap_companies = [d for d in classified["no_new"] if not d.get("errors")]
    if gap_companies:
        lines.append(f"**No new documents ({len(gap_companies)} companies):** "
                      + ", ".join(d["ticker"] for d in sorted(gap_companies, key=lambda x: x["ticker"])))

    # Triage summary — how the Document Triage Agent handled this harvest
    triage = harvest_result.get("triage_stats") or {}
    if triage.get("total", 0) > 0:
        lines.append("")
        lines.append(
            f"**Triage:** {triage.get('total', 0)} candidates classified — "
            f"{triage.get('auto_ingested', 0)} auto-ingested, "
            f"{triage.get('needs_review', 0)} flagged for review, "
            f"{triage.get('skipped_by_triage', 0)} dropped as junk"
        )

    # Parity — one-line validation signal between old and new coverage
    # engines. Omitted entirely once the old engine is retired.
    parity = harvest_result.get("parity_stats") or {}
    if parity and parity.get("total", 0) > 0:
        total = parity.get("total", 0)
        agree = parity.get("agree", 0)
        if parity.get("disagree", 0) == 0:
            lines.append("")
            lines.append(f"**Parity:** all {total} companies agree (old ↔ new coverage)")
        else:
            lines.append("")
            lines.append(
                f"**Parity:** {agree}/{total} agree — "
                f"{parity.get('new_flagged_old_clean', 0)} new-flagged-old-clean, "
                f"{parity.get('old_flagged_new_clean', 0)} old-flagged-new-clean "
                f"(see /harvester/coverage-compare)"
            )

    # Coverage Monitor — current gap snapshot + last-week rescan activity
    cov_mon = harvest_result.get("coverage_monitor_stats") or {}
    if cov_mon:
        sev = cov_mon.get("by_severity") or {}
        total = cov_mon.get("total_gaps", 0)
        rescans = cov_mon.get("rescans", 0)
        lookback = cov_mon.get("lookback_days", 7)
        lines.append("")
        lines.append(
            f"**Coverage Monitor:** {total} gaps "
            f"({sev.get('critical', 0)} critical, "
            f"{sev.get('overdue', 0)} overdue, "
            f"{sev.get('source_broken', 0)} source-broken, "
            f"{sev.get('warning', 0)} approaching) — "
            f"{rescans} auto-rescans in last {lookback}d "
            f"({cov_mon.get('rescan_successes', 0)} found new docs, "
            f"{cov_mon.get('rescan_errors', 0)} errors)"
        )

    # Legacy static coverage check — kept until the new Coverage Monitor
    # has several weeks of production validation. See the plan §8 / CLAUDE.md.
    coverage_text = harvest_result.get("coverage_text")
    if coverage_text:
        lines.append("")
        lines.append(coverage_text)

    body_text = "\n".join(lines)

    # Power Automate Workflows webhook format — Adaptive Card as top-level body
    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": f"Weekly Harvest Report — {now}",
                "weight": "Bolder",
                "size": "Medium",
            },
            {
                "type": "TextBlock",
                "text": body_text,
                "wrap": True,
            },
        ],
        "actions": [
            {
                "type": "Action.OpenUrl",
                "title": "Open Platform",
                "url": settings.app_base_url,
            }
        ],
    }


async def post_teams_report(harvest_result: dict, report_id: str | None = None) -> bool:
    """Post the harvest report to Microsoft Teams. Returns True if sent."""
    webhook_url = settings.teams_webhook_url
    if not webhook_url:
        logger.debug("[REPORT] No TEAMS_WEBHOOK_URL configured — skipping Teams notification")
        return False

    payload = format_teams_message(harvest_result)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()

        # Mark report as sent
        if report_id:
            from apps.api.database import AsyncSessionLocal
            from sqlalchemy import text
            async with AsyncSessionLocal() as db:
                await db.execute(
                    text("UPDATE harvest_reports SET teams_sent = true WHERE id = :id"),
                    {"id": report_id},
                )
                await db.commit()

        logger.info("[REPORT] Teams notification sent successfully")
        return True
    except Exception as exc:
        logger.error("[REPORT] Teams notification failed: %s", exc)
        return False


async def run_and_report(trigger: str = "auto_weekly") -> dict:
    """Full weekly flow: harvest → save report → run coverage check → post to Teams."""
    # Capture harvest start so we can count the triage decisions that this
    # specific harvest produced (IngestionTriage.created_at >= this).
    harvest_start = datetime.now(timezone.utc)

    result = await run_weekly_harvest()
    logger.info("[REPORT] Harvest complete — new=%d skipped=%d failed=%d, saving report...",
                result.get("new", 0), result.get("skipped", 0), result.get("failed", 0))

    # Triage stats for candidates classified during this harvest
    try:
        result["triage_stats"] = await collect_triage_stats(harvest_start)
        logger.info("[REPORT] Triage: %s", result["triage_stats"])
    except Exception as exc:
        logger.warning("[REPORT] Triage stats collection failed: %s", exc)

    # Coverage Monitor snapshot (learned-cadence gaps + 7d rescan activity)
    try:
        result["coverage_monitor_stats"] = await collect_coverage_monitor_stats(lookback_days=7)
        logger.info("[REPORT] Coverage Monitor: %s", result["coverage_monitor_stats"])
    except Exception as exc:
        logger.warning("[REPORT] Coverage Monitor stats collection failed: %s", exc)

    # Parity check — validation window comparing old static check vs new
    # learned-cadence monitor. Drop this section once the migration is done.
    try:
        from apps.api.database import AsyncSessionLocal
        from services.harvester.coverage_compare import compare_coverage
        async with AsyncSessionLocal() as db:
            parity = await compare_coverage(db)
        result["parity_stats"] = parity.to_dict()["summary"]
        logger.info("[REPORT] Parity: %s", result["parity_stats"])
    except Exception as exc:
        logger.warning("[REPORT] Parity check failed: %s", exc)

    # Append legacy static coverage gap summary so it lands in the Teams message
    try:
        from apps.api.database import AsyncSessionLocal
        from services.harvester.coverage import (
            check_coverage, format_coverage_for_teams, format_coverage_summary,
        )
        async with AsyncSessionLocal() as db:
            coverage = await check_coverage(db)
        result["coverage_text"] = format_coverage_for_teams(coverage)
        logger.info("[REPORT] Coverage: %s", format_coverage_summary(coverage))
    except Exception as exc:
        logger.warning("[REPORT] Coverage check failed: %s", exc)

    report_id = None
    try:
        report_id = await save_report(result, trigger=trigger)
    except Exception as exc:
        logger.error("[REPORT] Failed to save report: %s", exc, exc_info=True)

    try:
        await post_teams_report(result, report_id=report_id)
    except Exception as exc:
        logger.error("[REPORT] Failed to post to Teams: %s", exc, exc_info=True)

    result["report_id"] = report_id
    return result
