"""Speech file and streaming endpoints."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from starlette.responses import StreamingResponse

from villager_tts.app.dependencies import require_api_key, require_ready
from villager_tts.app.metrics import Metrics
from villager_tts.app.schemas.speech import (
    OpenAISpeechRequest,
    SpeechRequest,
    SpeechResponseFormat,
    StreamSpeechRequest,
)
from villager_tts.app.services.speech import SpeechService, SynthesisRequest
from villager_tts.common.errors import InputTooLongError, SynthesisError
from villager_tts.core.audio.encoding import AudioEncodingError, chunk_to_l16, encode_audio
from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.runtime.state import RuntimeState

router = APIRouter(dependencies=[Depends(require_api_key)])
ReadyRuntime = Annotated[RuntimeState, Depends(require_ready)]
LOGGER = logging.getLogger(__name__)


@router.post("/speech")
async def create_openai_speech(
    payload: OpenAISpeechRequest,
    request: Request,
    runtime: ReadyRuntime,
) -> Response:
    """Synthesize speech using the OpenAI-compatible request contract."""
    del runtime
    synthesis = SynthesisRequest(
        text=payload.input,
        speed=payload.speed,
        gender=payload.voice,
        enable_villager=True,
    )
    _enforce_input_limit(payload.input, request)
    _log_request(
        synthesis,
        "openai",
        model=payload.model,
        response_format=payload.response_format,
    )
    return await _create_file_response(
        synthesis,
        request,
        endpoint="openai",
        response_format=payload.response_format,
    )


@router.post("/speech/file")
async def create_speech_file(
    payload: SpeechRequest,
    request: Request,
    runtime: ReadyRuntime,
) -> Response:
    """Synthesize a complete mono PCM16 WAV file."""
    del runtime
    synthesis = _domain_request(payload)
    _enforce_input_limit(payload.input, request)
    _log_request(synthesis, "file", response_format="wav")
    return await _create_file_response(
        synthesis,
        request,
        endpoint="file",
        response_format="wav",
    )


async def _create_file_response(
    synthesis: SynthesisRequest,
    request: Request,
    *,
    endpoint: str,
    response_format: SpeechResponseFormat,
) -> Response:
    """Run one complete synthesis and encode its accumulated chunks."""
    metrics = request.app.state.metrics
    metrics.requests.labels(endpoint).inc()
    started = time.monotonic()
    token = CancellationToken()
    service: SpeechService = request.app.state.speech_service
    chunks: list[AudioChunk] = []
    metrics.inflight.labels(endpoint).inc()
    try:
        async for chunk in service.synthesize(synthesis, token):
            chunks.append(chunk)
        encoding_started = time.monotonic()
        try:
            encoded = await asyncio.to_thread(
                encode_audio,
                chunks,
                sample_rate=request.app.state.settings.output_sample_rate,
                response_format=response_format,
            )
        except AudioEncodingError as error:
            msg = f"Failed to encode speech as {response_format}."
            raise SynthesisError(msg) from error
        sample_count = sum(chunk.samples.size for chunk in chunks)
        audio_duration_ms = _audio_duration_ms(
            sample_count,
            request.app.state.settings.output_sample_rate,
        )
        LOGGER.info(
            "audio encoded",
            extra={
                "endpoint": endpoint,
                "stage": "audio_encoding",
                "duration_ms": _elapsed_ms(encoding_started),
                "response_format": response_format,
                "chunk_count": len(chunks),
                "output_bytes": len(encoded.content),
                "audio_duration_ms": audio_duration_ms,
            },
        )
        LOGGER.info(
            "speech synthesis completed",
            extra={
                "endpoint": endpoint,
                "stage": "total",
                "duration_ms": _elapsed_ms(started),
                "response_format": response_format,
                "chunk_count": len(chunks),
                "output_bytes": len(encoded.content),
                "audio_duration_ms": audio_duration_ms,
            },
        )
    except BaseException as error:
        metrics.failures.labels(endpoint, type(error).__name__).inc()
        _log_failure(endpoint, started, error)
        raise
    finally:
        token.cancel()
        metrics.inflight.labels(endpoint).dec()
        metrics.synthesis_seconds.labels(endpoint).observe(time.monotonic() - started)
    request_id = request.state.request_id
    return Response(
        content=encoded.content,
        media_type=encoded.media_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="villager-{request_id}.{encoded.extension}"'
            ),
            "X-Audio-Sample-Rate": str(request.app.state.settings.output_sample_rate),
            "X-Audio-Channels": "1",
        },
    )


@router.post("/speech/stream")
async def stream_speech(
    payload: StreamSpeechRequest,
    request: Request,
    runtime: ReadyRuntime,
) -> StreamingResponse:
    """Stream RFC 2586 L16 audio after prefetching the first block."""
    del runtime
    synthesis = _domain_request(payload)
    _enforce_input_limit(payload.input, request)
    endpoint = "stream"
    _log_request(synthesis, endpoint)
    metrics = request.app.state.metrics
    metrics.requests.labels(endpoint).inc()
    started = time.monotonic()
    token = CancellationToken()
    service: SpeechService = request.app.state.speech_service
    audio = service.synthesize(synthesis, token)
    metrics.inflight.labels(endpoint).inc()
    try:
        first = await anext(audio)
    except StopAsyncIteration as error:
        metrics.failures.labels(endpoint, "empty").inc()
        _log_failure(endpoint, started, error)
        metrics.inflight.labels(endpoint).dec()
        token.cancel()
        await audio.aclose()
        raise SynthesisError("The inference pipeline produced no audio.") from error
    except BaseException as error:
        metrics.failures.labels(endpoint, type(error).__name__).inc()
        _log_failure(endpoint, started, error)
        metrics.inflight.labels(endpoint).dec()
        token.cancel()
        await audio.aclose()
        raise
    first_audio_seconds = time.monotonic() - started
    metrics.first_audio_seconds.observe(first_audio_seconds)
    LOGGER.info(
        "first audio ready",
        extra={
            "endpoint": endpoint,
            "stage": "first_audio",
            "duration_ms": round(first_audio_seconds * 1_000, 2),
            "output_sample_rate": first.sample_rate,
            "output_samples": first.samples.size,
        },
    )

    sample_rate = request.app.state.settings.output_sample_rate
    return StreamingResponse(
        _stream_body(
            first,
            audio,
            token=token,
            metrics=metrics,
            endpoint=endpoint,
            started=started,
            sample_rate=sample_rate,
        ),
        media_type=f"audio/L16;rate={sample_rate};channels=1",
        headers={
            "Cache-Control": "no-store",
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Format": "s16be",
        },
    )


def _domain_request(payload: SpeechRequest | StreamSpeechRequest) -> SynthesisRequest:
    return SynthesisRequest(
        text=payload.input,
        speed=payload.speed,
        gender=payload.gender,
        enable_villager=payload.enable_villager,
    )


def _enforce_input_limit(text: str, request: Request) -> None:
    maximum = request.app.state.settings.max_text_characters
    if len(text) > maximum:
        raise InputTooLongError(maximum)


def _log_request(
    synthesis: SynthesisRequest,
    endpoint: str,
    *,
    model: str | None = None,
    response_format: SpeechResponseFormat | None = None,
) -> None:
    LOGGER.info(
        "speech request accepted",
        extra={
            "endpoint": endpoint,
            "model": model,
            "response_format": response_format,
            "input_text": synthesis.text,
            "input_characters": len(synthesis.text),
            "speed": synthesis.speed,
            "gender": synthesis.gender,
            "enable_villager": synthesis.enable_villager,
        },
    )


def _log_failure(endpoint: str, started: float, error: BaseException) -> None:
    extra = {
        "endpoint": endpoint,
        "stage": "total",
        "duration_ms": _elapsed_ms(started),
        "error_kind": type(error).__name__,
    }
    if isinstance(error, Exception):
        LOGGER.error("speech synthesis failed", extra=extra, exc_info=error)
    else:
        LOGGER.info("speech synthesis interrupted", extra=extra)


async def _stream_body(
    first: AudioChunk,
    audio: AsyncGenerator[AudioChunk, None],
    *,
    token: CancellationToken,
    metrics: Metrics,
    endpoint: str,
    started: float,
    sample_rate: int,
) -> AsyncIterator[bytes]:
    chunk_count = 0
    sample_count = 0
    output_bytes = 0
    completed = False
    try:
        encoded = chunk_to_l16(first)
        chunk_count += 1
        sample_count += first.samples.size
        output_bytes += len(encoded)
        yield encoded
        async for chunk in audio:
            encoded = chunk_to_l16(chunk)
            chunk_count += 1
            sample_count += chunk.samples.size
            output_bytes += len(encoded)
            yield encoded
        completed = True
    except BaseException as error:
        metrics.failures.labels(endpoint, type(error).__name__).inc()
        _log_failure(endpoint, started, error)
        raise
    finally:
        token.cancel()
        await audio.aclose()
        metrics.inflight.labels(endpoint).dec()
        metrics.synthesis_seconds.labels(endpoint).observe(time.monotonic() - started)
        if completed:
            LOGGER.info(
                "speech stream completed",
                extra={
                    "endpoint": endpoint,
                    "stage": "total",
                    "duration_ms": _elapsed_ms(started),
                    "chunk_count": chunk_count,
                    "output_bytes": output_bytes,
                    "audio_duration_ms": _audio_duration_ms(sample_count, sample_rate),
                },
            )


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1_000, 2)


def _audio_duration_ms(sample_count: int, sample_rate: int) -> float:
    return round(sample_count / sample_rate * 1_000, 2)
