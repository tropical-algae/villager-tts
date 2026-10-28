"""FastAPI dependencies."""

import hmac
from typing import Annotated

from fastapi import Header, Request

from villager_tts.common.errors import ServiceError, ServiceUnavailableError
from villager_tts.config import Settings
from villager_tts.core.runtime.state import RuntimeState


def get_settings(request: Request) -> Settings:
    """Return this application's immutable settings object."""
    return request.app.state.settings  # type: ignore[no-any-return]


def get_runtime(request: Request) -> RuntimeState:
    """Return the model runtime state."""
    return request.app.state.runtime  # type: ignore[no-any-return]


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Validate the optional shared key from native or OpenAI-compatible headers."""
    settings = get_settings(request)
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value()
    candidates = [x_api_key] if x_api_key is not None else []
    if authorization is not None:
        scheme, separator, credentials = authorization.partition(" ")
        if separator and scheme.casefold() == "bearer":
            candidates.append(credentials)
    valid = False
    for candidate in candidates:
        valid |= hmac.compare_digest(candidate, expected)
    if not valid:
        raise ServiceError(
            status_code=401,
            title="Unauthorized",
            detail="A valid Bearer token or X-API-Key header is required.",
            error_type="https://villager-tts.dev/problems/unauthorized",
        )


async def require_ready(request: Request) -> RuntimeState:
    """Reject inference until all model backends are loaded."""
    runtime = get_runtime(request)
    if runtime.engines is None:
        raise ServiceUnavailableError
    return runtime
