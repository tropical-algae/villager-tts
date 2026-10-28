"""Public HTTP contract tests."""

import io
import json
import logging
import wave

import httpx
import numpy as np
from fastapi import FastAPI

from villager_tts.app.factory import create_app
from villager_tts.app.schemas.speech import OpenAISpeechRequest, SpeechRequest, StreamSpeechRequest
from villager_tts.common.logging import JsonFormatter
from villager_tts.config import Settings
from villager_tts.core.runtime.state import EngineBundle


async def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


async def test_health_file_stream_and_metrics_contract(app: FastAPI) -> None:
    async with await _client(app) as client:
        live = await client.get("/health/live")
        ready = await client.get("/health/ready")
        file_response = await client.post(
            "/v1/audio/speech/file",
            json={"input": "你好。"},
        )
        stream_response = await client.post(
            "/v1/audio/speech/stream",
            json={"input": "你好。"},
        )
        raw_female_response = await client.post(
            "/v1/audio/speech/file",
            json={
                "input": "你好。",
                "gender": "female",
                "enable_villager": False,
            },
        )
        metrics = await client.get("/metrics")

    assert live.json() == {
        "status": "ok",
        "tts_backend": None,
        "vc_backend": None,
        "detail": None,
    }
    assert ready.json()["status"] == "ready"
    assert ready.json()["tts_backend"] == "fake-tts"
    assert file_response.status_code == 200
    assert file_response.headers["content-type"] == "audio/wav"
    assert "attachment;" in file_response.headers["content-disposition"]
    assert file_response.headers["x-audio-channels"] == "1"
    assert file_response.headers["x-request-id"]
    assert stream_response.status_code == 200
    assert stream_response.headers["content-type"] == "audio/L16;rate=48000;channels=1"
    assert stream_response.headers["cache-control"] == "no-store"
    assert stream_response.headers["x-audio-format"] == "s16be"

    with wave.open(io.BytesIO(file_response.content), "rb") as wav_file:
        little_endian_pcm = wav_file.readframes(wav_file.getnframes())
    file_samples = np.frombuffer(little_endian_pcm, dtype="<i2")
    stream_samples = np.frombuffer(stream_response.content, dtype=">i2")
    assert np.array_equal(file_samples, stream_samples)
    with wave.open(io.BytesIO(raw_female_response.content), "rb") as raw_wav:
        raw_samples = np.frombuffer(raw_wav.readframes(raw_wav.getnframes()), dtype="<i2")
    assert np.all(file_samples < 0)
    assert np.all(raw_samples > 0)
    assert np.array_equal(raw_samples, -file_samples)
    assert b"villager_tts_requests_total" in metrics.content


def test_speech_request_fields_have_openapi_descriptions() -> None:
    for request_model in (SpeechRequest, StreamSpeechRequest):
        properties = request_model.model_json_schema()["properties"]
        assert all(field.get("description") for field in properties.values())
        assert properties["gender"]["default"] == "female"
    openai_properties = OpenAISpeechRequest.model_json_schema()["properties"]
    assert all(field.get("description") for field in openai_properties.values())
    assert openai_properties["response_format"]["default"] == "mp3"
    assert openai_properties["stream_format"]["default"] == "audio"


async def test_openai_speech_contract_returns_the_requested_format(app: FastAPI) -> None:
    async with await _client(app) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": "OpenAI compatibility.",
                "voice": "female",
                "instructions": "Accepted for protocol compatibility.",
                "response_format": "wav",
                "speed": 1.0,
                "stream_format": "audio",
            },
        )
        old_prefix = await client.post(
            "/api/v1/audio/speech",
            json={"input": "removed"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"].endswith('.wav"')
    assert old_prefix.status_code == 404
    with wave.open(io.BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getframerate() == 48_000


async def test_openai_validation_errors_use_the_openai_envelope(app: FastAPI) -> None:
    async with await _client(app) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={"input": "missing model and voice"},
        )

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] in {"model", "voice"}


async def test_stream_request_emits_parameters_and_stage_timings(app: FastAPI) -> None:
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        async with await _client(app) as client:
            response = await client.post(
                "/v1/audio/speech/stream",
                json={
                    "input": "日志测试。",
                    "speed": 1.25,
                    "gender": "female",
                    "enable_villager": False,
                },
            )
    finally:
        root.removeHandler(handler)
        handler.close()

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    request_id = response.headers["x-request-id"]
    request_event = next(event for event in events if event["message"] == "speech request accepted")
    assert request_event["request_id"] == request_id
    assert request_event["endpoint"] == "stream"
    assert request_event["input_text"] == "日志测试。"
    assert request_event["speed"] == 1.25
    assert request_event["gender"] == "female"
    assert request_event["enable_villager"] is False
    request_events = [event for event in events if event.get("request_id") == request_id]
    stages = {event.get("stage") for event in request_events}
    assert {
        "text_segmentation",
        "capacity_wait",
        "tts",
        "resampling",
        "first_audio",
        "total",
    } <= stages
    completed = next(
        event for event in request_events if event["message"] == "speech stream completed"
    )
    assert completed["output_bytes"] == len(response.content)
    assert completed["duration_ms"] >= 0


async def test_validation_errors_use_problem_details(app: FastAPI) -> None:
    async with await _client(app) as client:
        response = await client.post(
            "/v1/audio/speech/file",
            json={"input": "   ", "unknown": True},
        )

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["type"].endswith("validation-error")
    assert body["request_id"] == response.headers["x-request-id"]


async def test_configured_input_limit_is_enforced(
    settings: Settings,
    engines: EngineBundle,
) -> None:
    limited = create_app(
        settings=Settings(
            **settings.model_dump(exclude={"max_text_characters"}),
            max_text_characters=4,
        ),
        engines=engines,
    )
    async with await _client(limited) as client:
        file_response = await client.post(
            "/v1/audio/speech/file",
            json={"input": "12345"},
        )
        stream_response = await client.post(
            "/v1/audio/speech/stream",
            json={"input": "12345"},
        )

    for response in (file_response, stream_response):
        assert response.status_code == 413
        assert response.headers["content-type"] == "application/problem+json"
        assert response.json()["type"].endswith("input-too-long")


async def test_optional_api_key_is_enforced(
    settings: Settings,
    engines: EngineBundle,
) -> None:
    protected = create_app(
        settings=Settings(**settings.model_dump(exclude={"api_key"}), api_key="secret"),
        engines=engines,
    )
    async with await _client(protected) as client:
        denied = await client.post("/v1/audio/speech/file", json={"input": "hello"})
        allowed = await client.post(
            "/v1/audio/speech/file",
            json={"input": "hello"},
            headers={"X-API-Key": "secret"},
        )
        denied_openai = await client.post(
            "/v1/audio/speech",
            json={"model": "villager-tts", "input": "hello", "voice": "female"},
        )
        allowed_openai = await client.post(
            "/v1/audio/speech",
            json={
                "model": "villager-tts",
                "input": "hello",
                "voice": "female",
                "response_format": "wav",
            },
            headers={"Authorization": "Bearer secret"},
        )

    assert denied.status_code == 401
    assert denied.headers["content-type"] == "application/problem+json"
    assert allowed.status_code == 200
    assert denied_openai.status_code == 401
    assert denied_openai.json()["error"]["type"] == "authentication_error"
    assert allowed_openai.status_code == 200


async def test_unready_application_reports_503(settings: Settings) -> None:
    unready = create_app(settings=settings)
    async with await _client(unready) as client:
        readiness = await client.get("/health/ready")
        speech = await client.post("/v1/audio/speech/file", json={"input": "hello"})

    assert readiness.status_code == 503
    assert readiness.json()["status"] == "starting"
    assert speech.status_code == 503
    assert speech.headers["retry-after"] == "1"
