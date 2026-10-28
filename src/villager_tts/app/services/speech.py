"""End-to-end speech synthesis orchestration."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass

from villager_tts.common.errors import InferenceTimeoutError, SynthesisError
from villager_tts.config import Settings
from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.runtime.capacity import CapacityLimiter
from villager_tts.core.runtime.state import EngineBundle
from villager_tts.core.text.segmenter import segment_text
from villager_tts.core.tts.protocol import DEFAULT_VOICE_GENDER, TtsRequest, VoiceGender
from villager_tts.core.vc.passthrough import PassthroughVoiceConverter

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """Transport-independent speech request."""

    text: str
    speed: float
    gender: VoiceGender = DEFAULT_VOICE_GENDER
    enable_villager: bool = True


class SpeechService:
    """Run text segmentation, TTS and voice conversion as one audio stream."""

    def __init__(self, settings: Settings, engines: EngineBundle) -> None:
        self._settings = settings
        self._engines = engines
        self._capacity = CapacityLimiter(
            settings.max_concurrent_syntheses,
            settings.capacity_wait_seconds,
        )
        self._passthrough = PassthroughVoiceConverter(settings.output_sample_rate)

    async def synthesize(
        self,
        request: SynthesisRequest,
        cancellation: CancellationToken,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Yield converted audio and retain admission until the stream closes."""
        segmentation_started = time.monotonic()
        phrases = segment_text(
            request.text,
            target_chars=self._settings.phrase_target_characters,
            max_chars=self._settings.phrase_max_characters,
        )
        if not phrases:
            raise SynthesisError("Input produced no speakable text.")
        LOGGER.info(
            "text segmented",
            extra={
                "stage": "text_segmentation",
                "duration_ms": round((time.monotonic() - segmentation_started) * 1_000, 2),
                "input_characters": len(request.text),
                "phrase_count": len(phrases),
            },
        )

        capacity_started = time.monotonic()
        async with self._capacity.slot():
            LOGGER.info(
                "synthesis capacity acquired",
                extra={
                    "stage": "capacity_wait",
                    "duration_ms": round((time.monotonic() - capacity_started) * 1_000, 2),
                },
            )
            source = self._source_chunks(phrases, request, cancellation)
            converter = (
                self._engines.voice_converter if request.enable_villager else self._passthrough
            )
            converted = converter.convert(source, cancellation)
            yielded = False
            sequence = 0
            pending: AudioChunk | None = None
            async for chunk in converted:
                cancellation.raise_if_cancelled()
                if chunk.sample_rate != self._settings.output_sample_rate:
                    msg = f"voice converter returned an unexpected sample rate: {chunk.sample_rate}"
                    raise SynthesisError(msg)
                if pending is not None:
                    yield AudioChunk(
                        samples=pending.samples,
                        sample_rate=pending.sample_rate,
                        sequence=sequence,
                    )
                    yielded = True
                    sequence += 1
                pending = chunk
            if pending is not None:
                yield AudioChunk(
                    samples=pending.samples,
                    sample_rate=pending.sample_rate,
                    sequence=sequence,
                    is_final=True,
                )
                yielded = True
            if not yielded:
                raise SynthesisError("The inference pipeline produced no audio.")

    async def _source_chunks(
        self,
        phrases: list[str],
        request: SynthesisRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        sequence = 0
        phrase_count = len(phrases)
        for phrase_index, phrase in enumerate(phrases, start=1):
            cancellation.raise_if_cancelled()
            tts_started = time.monotonic()
            sample_count = 0
            pending: AudioChunk | None = None
            tts_stream = self._engines.tts.synthesize(
                TtsRequest(
                    text=phrase,
                    speed=request.speed,
                    gender=request.gender,
                ),
                cancellation,
            )
            remaining_tts_seconds = self._settings.tts_request_timeout_seconds
            try:
                while True:
                    inference_wait_started = time.monotonic()
                    try:
                        chunk = await asyncio.wait_for(
                            anext(tts_stream),
                            timeout=remaining_tts_seconds,
                        )
                    except StopAsyncIteration:
                        break
                    finally:
                        remaining_tts_seconds -= time.monotonic() - inference_wait_started
                    cancellation.raise_if_cancelled()
                    sample_count += chunk.samples.size
                    if pending is not None:
                        yield AudioChunk(
                            samples=pending.samples,
                            sample_rate=pending.sample_rate,
                            sequence=sequence,
                        )
                        sequence += 1
                    pending = chunk
            except TimeoutError as error:
                LOGGER.warning(
                    "tts phrase timed out",
                    extra={
                        "stage": "tts",
                        "duration_ms": round((time.monotonic() - tts_started) * 1_000, 2),
                        "phrase_index": phrase_index,
                        "phrase_count": phrase_count,
                        "phrase_characters": len(phrase),
                        "error_kind": type(error).__name__,
                    },
                )
                msg = "TTS backend exceeded its inference deadline."
                raise InferenceTimeoutError(msg) from error
            finally:
                close = getattr(tts_stream, "aclose", None)
                if close is not None:
                    await close()
            if pending is None:
                raise SynthesisError("The TTS backend produced no audio.")
            LOGGER.info(
                "tts phrase completed",
                extra={
                    "stage": "tts",
                    "duration_ms": round((time.monotonic() - tts_started) * 1_000, 2),
                    "phrase_index": phrase_index,
                    "phrase_count": phrase_count,
                    "phrase_characters": len(phrase),
                    "output_sample_rate": pending.sample_rate,
                    "output_samples": sample_count,
                    "output_audio_ms": round(sample_count / pending.sample_rate * 1_000, 2),
                },
            )
            yield AudioChunk(
                samples=pending.samples,
                sample_rate=pending.sample_rate,
                sequence=sequence,
                is_final=True,
            )
            sequence += 1
