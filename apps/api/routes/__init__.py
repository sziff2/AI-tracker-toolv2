"""API route registration."""
from apps.api.routes.companies import router as companies_router
from apps.api.routes.documents import router as documents_router
from apps.api.routes.outputs import router as outputs_router
from apps.api.routes.review import router as review_router
from apps.api.routes.kpi_tracker import router as kpi_tracker_router

__all__ = ["companies_router", "documents_router", "outputs_router", "review_router", "kpi_tracker_router"]
