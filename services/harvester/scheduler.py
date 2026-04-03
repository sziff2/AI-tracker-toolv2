"""
Weekly Harvest Report — generation, storage, and Teams notification.

Called by Celery Beat (weekly Monday 06:00 UTC) or manually via API.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

import httpx

from configs.settings import settings

logger = logging.getLogger(__name__)


async def run_weekly_harvest() -> dict:
    """Run a cost-controlled harvest (no LLM) and return enriched result."""
    from services.harvester import run_harvest
    return await run_harvest(skip_llm=True)


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
            INSERT INTO harvest_reports (id, run_at, trigger, summary_json, details_json, teams_sent)
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


def format_teams_message(harvest_result: dict) -> dict:
    """Build a Microsoft Teams Adaptive Card payload."""
    summary = {
        "new": harvest_result.get("new", 0),
        "skipped": harvest_result.get("skipped", 0),
        "failed": harvest_result.get("failed", 0),
    }
    details = harvest_result.get("details", [])
    classified = _classify_details(details)
    now = datetime.now(timezone.utc).strftime("%A %-d %b %Y")

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

    body_text = "\n".join(lines)

    # Teams Adaptive Card format (works with Power Automate Workflows webhooks)
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
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
                },
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
    """Full weekly flow: harvest → save report → post to Teams."""
    result = await run_weekly_harvest()
    report_id = await save_report(result, trigger=trigger)
    await post_teams_report(result, report_id=report_id)
    result["report_id"] = report_id
    return result
