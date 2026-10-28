"""Speech orchestration tests."""

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from villager_tts.app.services.speech import SpeechService, SynthesisRequest
from villager_tts.common.errors import InferenceTimeoutError, SynthesisError
from villager_tts.config import Settings
from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.runtime.state import EngineBundle
from villager_tts.core.tts.protocol import TtsRequest


async def test_service_resequences_and_marks_only_last_chunk_final(
    settings: Settings,
    engines: EngineBundle,
) -> None:
    service = SpeechService(settings, engines)
    chunks = [
        chunk
        async for chunk in service.synthesize(
            SynthesisRequest("你好。世界。", 1.0),
            CancellationToken(),
        )
    ]

    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.is_final for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert np.all(np.concatenate([chunk.samples for chunk in chunks]) == -0.5)


class EmptyTts:
    @property
    def name(self) -> str:
        return "empty"

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        if cancellation.is_cancelled and request.text:
            yield AudioChunk(np.zeros(0, dtype=np.float32), 48_000, 0)


async def test_service_rejects_empty_tts_output(settings: Settings, engines: EngineBundle) -> None:
    service = SpeechService(
        settings,
        EngineBundle(tts=EmptyTts(), voice_converter=engines.voice_converter),
    )
    with pytest.raises(SynthesisError, match="TTS"):
        async for _ in service.synthesize(
            SynthesisRequest("hello", 1.0),
            CancellationToken(),
        ):
            pass


class SlowTts:
    @property
    def name(self) -> str:
        return "slow"

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        del request, cancellation
        await asyncio.Event().wait()
        yield AudioChunk(np.zeros(1, dtype=np.float32), 48_000, 0, is_final=True)


async def test_service_enforces_the_tts_inference_deadline(
    settings: Settings,
    engines: EngineBundle,
) -> None:
    timed_settings = Settings(
        **settings.model_dump(exclude={"tts_request_timeout_seconds"}),
        tts_request_timeout_seconds=0.001,
    )
    service = SpeechService(
        timed_settings,
        EngineBundle(tts=SlowTts(), voice_converter=engines.voice_converter),
    )

    with pytest.raises(InferenceTimeoutError, match="deadline") as error:
        async for _ in service.synthesize(
            SynthesisRequest("hello", 1.0),
            CancellationToken(),
        ):
            pass

    assert error.value.status_code == 504


async def test_service_selects_female_tts_and_can_bypass_voice_conversion(
    settings: Settings,
    engines: EngineBundle,
) -> None:
    service = SpeechService(settings, engines)

    chunks = [
        chunk
        async for chunk in service.synthesize(
            SynthesisRequest(
                "你好。",
                1.0,
                gender="female",
                enable_villager=False,
            ),
            CancellationToken(),
        )
    ]

    assert np.all(np.concatenate([chunk.samples for chunk in chunks]) == 0.5)
