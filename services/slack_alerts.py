"""
Slack webhook alerts for budget warnings and pipeline errors.
"""

import logging

import httpx

from configs.settings import settings

logger = logging.getLogger(__name__)

_LEVEL_EMOJI = {
    "info": ":information_source:",
    "warn": ":warning:",
    "error": ":x:",
}


async def send_slack(message: str, level: str = "info") -> None:
    """Post a message to the configured Slack webhook.

    Args:
        message: Plain-text message body.
        level: One of "info", "warn", "error" (controls emoji prefix).
    """
    url = settings.slack_webhook_url
    if not url:
        logger.debug("[SLACK] No webhook URL configured, skipping alert")
        return

    emoji = _LEVEL_EMOJI.get(level, ":information_source:")
    payload = {"text": f"{emoji} *AI Tracker* | {message}"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                logger.warning("[SLACK] Webhook returned %s: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        # Never let a Slack failure break the pipeline
        logger.warning("[SLACK] Failed to send alert: %s", e)
