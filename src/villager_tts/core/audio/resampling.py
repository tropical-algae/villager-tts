"""Stateful streaming sample-rate conversion."""

import logging
import time
from collections.abc import AsyncIterator

import numpy as np
import soxr

from villager_tts.core.audio.models import AudioChunk

LOGGER = logging.getLogger(__name__)


async def resample_stream(
    chunks: AsyncIterator[AudioChunk],
    *,
    output_rate: int,
) -> AsyncIterator[AudioChunk]:
    """Resample phrase-delimited audio while retaining per-phrase filter state."""
    resampler: soxr.ResampleStream | None = None
    input_rate: int | None = None
    sequence = 0
    phrase_index = 0
    input_samples = 0
    output_samples = 0
    processing_seconds = 0.0
    async for chunk in chunks:
        if input_rate is None:
            input_rate = chunk.sample_rate
            if input_rate != output_rate:
                resampler = soxr.ResampleStream(
                    input_rate,
                    output_rate,
                    1,
                    dtype="float32",
                    quality="HQ",
                )
        elif chunk.sample_rate != input_rate:
            msg = "sample rate changed within an audio phrase"
            raise ValueError(msg)

        started = time.monotonic()
        if resampler is None:
            samples = chunk.samples
        else:
            samples = np.asarray(
                resampler.resample_chunk(chunk.samples, last=chunk.is_final),
                dtype=np.float32,
            )
        processing_seconds += time.monotonic() - started
        input_samples += chunk.samples.size
        output_samples += samples.size
        current_input_rate = input_rate
        if chunk.is_final:
            phrase_index += 1
            LOGGER.info(
                "source phrase resampled",
                extra={
                    "stage": "resampling",
                    "duration_ms": round(processing_seconds * 1_000, 2),
                    "phrase_index": phrase_index,
                    "input_sample_rate": current_input_rate,
                    "output_sample_rate": output_rate,
                    "input_samples": input_samples,
                    "output_samples": output_samples,
                    "input_audio_ms": round(input_samples / current_input_rate * 1_000, 2),
                    "output_audio_ms": round(output_samples / output_rate * 1_000, 2),
                },
            )
        if samples.size:
            yield AudioChunk(
                samples=samples,
                sample_rate=output_rate,
                sequence=sequence,
                is_final=chunk.is_final,
            )
            sequence += 1
        if chunk.is_final:
            resampler = None
            input_rate = None
            input_samples = 0
            output_samples = 0
            processing_seconds = 0.0
    if input_rate is not None:
        msg = "audio stream ended without a phrase boundary"
        raise ValueError(msg)
