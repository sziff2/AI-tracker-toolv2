"""
Shared SlowAPI rate-limiter instance.

Import `limiter` in route modules to apply per-endpoint limits:

    from apps.api.rate_limit import limiter

    @router.post("/heavy-endpoint")
    @limiter.limit("10/minute")
    async def heavy_endpoint(request: Request, ...):
        ...
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],
)
