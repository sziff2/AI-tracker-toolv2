"""
Alert notifications via Microsoft Teams webhook.

Used for budget warnings, pipeline errors, and other operational alerts.
Sends Adaptive Cards to the Teams channel configured via TEAMS_WEBHOOK_URL.
"""

import logging

import httpx

from configs.settings import settings

logger = logging.getLogger(__name__)

_LEVEL_EMOJI = {
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "❌",
}

_LEVEL_COLOUR = {
    "info": "default",
    "warning": "warning",
    "error": "attention",
}


async def send_alert(message: str, level: str = "warning") -> bool:
    """Post an alert to the configured Teams webhook.

    Args:
        message: Plain-text message body.
        level: One of "info", "warning", "error".

    Returns:
        True if sent successfully, False otherwise.
    """
    url = settings.teams_webhook_url
    if not url:
        logger.debug("[ALERT] No TEAMS_WEBHOOK_URL configured, skipping")
        return False

    emoji = _LEVEL_EMOJI.get(level, "ℹ️")
    colour = _LEVEL_COLOUR.get(level, "default")

    payload = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": f"{emoji} AI Tracker Alert",
                "weight": "Bolder",
                "size": "Medium",
            },
            {
                "type": "TextBlock",
                "text": message,
                "wrap": True,
                "color": colour,
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code < 300:
                logger.info("[ALERT] Teams notification sent: %s", message[:80])
                return True
            else:
                logger.warning("[ALERT] Teams webhook returned %s: %s", resp.status_code, resp.text[:200])
                return False
    except Exception as e:
        logger.warning("[ALERT] Failed to send Teams alert: %s", e)
        return False


# Backward compat alias — budget_guard and autorun import send_slack
send_slack = send_alert
