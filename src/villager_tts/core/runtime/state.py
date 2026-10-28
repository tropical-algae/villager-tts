"""Inference engine lifecycle state."""

from dataclasses import dataclass
from enum import StrEnum

from villager_tts.core.tts.protocol import TtsEngine
from villager_tts.core.vc.protocol import VoiceConverter


@dataclass(frozen=True, slots=True)
class EngineBundle:
    """Models loaded once and shared by request services."""

    tts: TtsEngine
    voice_converter: VoiceConverter


class Readiness(StrEnum):
    """Application readiness state."""

    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"


class RuntimeState:
    """Mutable state owned by one FastAPI application."""

    def __init__(self) -> None:
        self.status = Readiness.STARTING
        self.engines: EngineBundle | None = None
        self.failure: str | None = None

    def set_ready(self, engines: EngineBundle) -> None:
        """Publish successfully loaded engines."""
        self.engines = engines
        self.failure = None
        self.status = Readiness.READY

    def set_failed(self, failure: str) -> None:
        """Record a safe initialization error."""
        self.engines = None
        self.failure = failure
        self.status = Readiness.FAILED
