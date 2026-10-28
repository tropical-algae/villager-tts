"""Shared deterministic test doubles."""

from collections.abc import AsyncIterator

import numpy as np
import pytest
from fastapi import FastAPI

from villager_tts.app.factory import create_app
from villager_tts.config import Settings
from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.runtime.state import EngineBundle
from villager_tts.core.tts.protocol import TtsRequest


class FakeTts:
    """Encode text length as predictable audio samples."""

    @property
    def name(self) -> str:
        return "fake-tts"

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        cancellation.raise_if_cancelled()
        amplitude = 0.25 if request.gender == "male" else 0.5
        samples = np.full(len(request.text), amplitude, dtype=np.float32)
        midpoint = max(1, samples.size // 2)
        yield AudioChunk(samples[:midpoint], 48_000, 0)
        if midpoint < samples.size:
            yield AudioChunk(samples[midpoint:], 48_000, 1, is_final=True)


class FakeVoiceConverter:
    """Invert input PCM while preserving phrase boundaries."""

    @property
    def name(self) -> str:
        return "fake-vc"

    async def convert(
        self,
        chunks: AsyncIterator[AudioChunk],
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        async for chunk in chunks:
            cancellation.raise_if_cancelled()
            yield AudioChunk(
                -chunk.samples,
                chunk.sample_rate,
                chunk.sequence,
                chunk.is_final,
            )


@pytest.fixture
def settings() -> Settings:
    """Return settings that avoid external model loading."""
    return Settings(
        verify_model_artifacts=False,
        tts_backend="tone",
        vc_backend="passthrough",
        phrase_target_characters=8,
        phrase_max_characters=16,
    )


@pytest.fixture
def engines() -> EngineBundle:
    """Return deterministic fake inference engines."""
    return EngineBundle(tts=FakeTts(), voice_converter=FakeVoiceConverter())


@pytest.fixture
def app(settings: Settings, engines: EngineBundle) -> FastAPI:
    """Create an application with already-loaded engines."""
    return create_app(settings=settings, engines=engines)
