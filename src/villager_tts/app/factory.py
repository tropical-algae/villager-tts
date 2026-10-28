"""FastAPI application construction."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from villager_tts import __version__
from villager_tts.app.api.constants import OPENAI_SPEECH_PATH
from villager_tts.app.api.router import router
from villager_tts.app.metrics import Metrics
from villager_tts.app.services.speech import SpeechService
from villager_tts.common.errors import ServiceError
from villager_tts.common.logging import bind_request_id, configure_logging, reset_request_id
from villager_tts.config import Settings
from villager_tts.core.runtime.loader import load_engine_bundle
from villager_tts.core.runtime.state import EngineBundle, Readiness, RuntimeState

LOGGER = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    engines: EngineBundle | None = None,
) -> FastAPI:
    """Create an isolated application, optionally with injected test engines."""
    resolved_settings = settings or Settings()
    configure_logging(resolved_settings.log_level)
    runtime = RuntimeState()
    if engines is not None:
        runtime.set_ready(engines)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if runtime.engines is None:
            load_started = time.monotonic()
            try:
                loaded = await asyncio.to_thread(load_engine_bundle, resolved_settings)
                runtime.set_ready(loaded)
                app.state.speech_service = SpeechService(resolved_settings, loaded)
                LOGGER.info(
                    "inference engines ready",
                    extra={
                        "stage": "model_loading",
                        "tts_backend": loaded.tts.name,
                        "vc_backend": loaded.voice_converter.name,
                        "tts_device": "cpu",
                        "rvc_device": resolved_settings.rvc_device,
                        "duration_ms": round((time.monotonic() - load_started) * 1_000, 2),
                    },
                )
            except Exception as error:  # model failures must leave liveness available
                runtime.set_failed(str(error))
                LOGGER.exception(
                    "inference engine initialization failed",
                    extra={
                        "stage": "model_loading",
                        "duration_ms": round((time.monotonic() - load_started) * 1_000, 2),
                    },
                )
        yield
        runtime.status = Readiness.STOPPING
        if runtime.engines is not None:
            await _close_backend(runtime.engines.voice_converter)
            await _close_backend(runtime.engines.tts)

    app = FastAPI(
        title="Villager TTS",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.runtime = runtime
    app.state.metrics = Metrics.create()
    if engines is not None:
        app.state.speech_service = SpeechService(resolved_settings, engines)
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(ServiceError, _service_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.include_router(router)
    return app


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a generated request ID and emit one access-log event."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Add correlation metadata to one HTTP request."""
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        context_token = bind_request_id(request_id)
        started = time.monotonic()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            LOGGER.info(
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.monotonic() - started) * 1_000, 2),
                },
            )
            return response
        finally:
            reset_request_id(context_token)


async def _service_error_handler(request: Request, error: Exception) -> JSONResponse:
    service_error = error
    if not isinstance(service_error, ServiceError):
        raise TypeError from error
    headers = {"Retry-After": "1"} if service_error.status_code in {429, 503} else None
    if request.url.path == OPENAI_SPEECH_PATH:
        error_type, code = _openai_error_classification(service_error.status_code)
        return JSONResponse(
            status_code=service_error.status_code,
            content={
                "error": {
                    "message": service_error.detail,
                    "type": error_type,
                    "param": None,
                    "code": code,
                }
            },
            headers=headers,
        )
    return JSONResponse(
        status_code=service_error.status_code,
        media_type="application/problem+json",
        content={
            "type": service_error.error_type,
            "title": service_error.title,
            "status": service_error.status_code,
            "detail": service_error.detail,
            "instance": str(request.url.path),
            "request_id": request.state.request_id,
        },
        headers=headers,
    )


async def _validation_error_handler(request: Request, error: Exception) -> JSONResponse:
    validation_error = error
    if not isinstance(validation_error, RequestValidationError):
        raise TypeError from error
    if request.url.path == OPENAI_SPEECH_PATH:
        errors = validation_error.errors()
        first = errors[0] if errors else {}
        location = first.get("loc", ())
        param = str(location[-1]) if isinstance(location, tuple) and location else None
        message = str(first.get("msg", "The request body does not match the API contract."))
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": "invalid_request",
                }
            },
        )
    return JSONResponse(
        status_code=422,
        media_type="application/problem+json",
        content={
            "type": "https://villager-tts.dev/problems/validation-error",
            "title": "Request validation failed",
            "status": 422,
            "detail": "The request body does not match the API contract.",
            "instance": str(request.url.path),
            "request_id": request.state.request_id,
            "errors": jsonable_encoder(validation_error.errors()),
        },
    )


def _openai_error_classification(status_code: int) -> tuple[str, str]:
    if status_code == status.HTTP_401_UNAUTHORIZED:
        return "authentication_error", "invalid_api_key"
    if status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        return "rate_limit_error", "rate_limit_exceeded"
    if status_code in {
        status.HTTP_413_CONTENT_TOO_LARGE,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
    }:
        return "invalid_request_error", "invalid_request"
    return "server_error", "server_error"


async def _close_backend(backend: object) -> None:
    close = getattr(backend, "aclose", None)
    if close is not None:
        await close()
