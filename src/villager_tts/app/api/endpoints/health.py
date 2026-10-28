"""Liveness, readiness, and metrics endpoints."""

from fastapi import APIRouter, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from villager_tts.app.dependencies import get_runtime
from villager_tts.app.schemas.speech import HealthResponse
from villager_tts.core.runtime.state import Readiness

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> HealthResponse:
    """Report whether the web process can serve requests."""
    return HealthResponse(status="ok")


@router.get("/health/ready")
async def ready(request: Request, response: Response) -> HealthResponse:
    """Report whether both inference backends are usable."""
    runtime = get_runtime(request)
    if runtime.status is not Readiness.READY or runtime.engines is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status=runtime.status, detail=runtime.failure)
    return HealthResponse(
        status=runtime.status,
        tts_backend=runtime.engines.tts.name,
        vc_backend=runtime.engines.voice_converter.name,
    )


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Expose Prometheus metrics."""
    payload = generate_latest(request.app.state.metrics.registry)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
