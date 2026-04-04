"""robots.txt compliance for IR page scraping."""
import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)


@lru_cache(maxsize=100)
def _get_robots_parser(domain: str) -> RobotFileParser | None:
    """Fetch and parse robots.txt for a domain. Cached per domain."""
    rp = RobotFileParser()
    robots_url = f"https://{domain}/robots.txt"
    try:
        resp = httpx.get(robots_url, timeout=5.0, follow_redirects=True)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            return rp
    except Exception as exc:
        logger.debug("Could not fetch robots.txt for %s: %s", domain, exc)
    return None


def can_fetch(url: str, user_agent: str = "OldfieldHarvester") -> bool:
    """Check if the URL is allowed by the site's robots.txt."""
    domain = urlparse(url).netloc
    rp = _get_robots_parser(domain)
    if rp is None:
        return True  # No robots.txt = allowed
    return rp.can_fetch(user_agent, url)


def get_crawl_delay(url: str, user_agent: str = "OldfieldHarvester", default: float = 1.0) -> float:
    """Get crawl-delay from robots.txt, or default."""
    domain = urlparse(url).netloc
    rp = _get_robots_parser(domain)
    if rp is None:
        return default
    delay = rp.crawl_delay(user_agent)
    return float(delay) if delay else default
