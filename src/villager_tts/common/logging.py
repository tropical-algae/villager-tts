"""Small structured logging setup without global framework dependencies."""

import json
import logging
import time
from contextvars import ContextVar, Token
from typing import Any

_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_STRUCTURED_FIELDS = (
    "request_id",
    "method",
    "path",
    "status_code",
    "endpoint",
    "model",
    "response_format",
    "stage",
    "duration_ms",
    "input_text",
    "input_characters",
    "speed",
    "gender",
    "enable_villager",
    "phrase_index",
    "phrase_count",
    "phrase_characters",
    "input_sample_rate",
    "output_sample_rate",
    "input_samples",
    "output_samples",
    "input_audio_ms",
    "output_audio_ms",
    "audio_duration_ms",
    "chunk_count",
    "output_bytes",
    "tts_backend",
    "vc_backend",
    "tts_device",
    "rvc_device",
    "torch_version",
    "cuda_version",
    "gpu_name",
    "worker_pid",
    "worker_pool_size",
    "error_kind",
)


class JsonFormatter(logging.Formatter):
    """Render standard log records as one-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize safe, operationally useful fields."""
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if field == "request_id" and value is None:
                value = _REQUEST_ID.get()
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str) -> None:
    """Install one JSON handler on the root logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def bind_request_id(request_id: str) -> Token[str | None]:
    """Bind a request ID to logs emitted in the current async context."""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the request context after one HTTP request finishes."""
    _REQUEST_ID.reset(token)
