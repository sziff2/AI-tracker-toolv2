"""API route registration."""
from apps.api.routes.companies import router as companies_router
from apps.api.routes.documents import router as documents_router
from apps.api.routes.outputs import router as outputs_router
from apps.api.routes.review import router as review_router
from apps.api.routes.kpi_tracker import router as kpi_tracker_router
from apps.api.routes.cockpit import router as cockpit_router
from apps.api.routes.experiments import router as experiments_router
from apps.api.routes.esg import router as esg_router
from apps.api.routes.portfolio import router as portfolio_router
from apps.api.routes.execution import router as execution_router
from apps.api.routes.autorun import router as autorun_router

__all__ = ["companies_router", "documents_router", "outputs_router", "review_router", "kpi_tracker_router", "cockpit_router", "experiments_router", "esg_router", "portfolio_router", "execution_router", "autorun_router"]
