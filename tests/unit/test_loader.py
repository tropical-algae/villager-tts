"""Configured backend loading and application lifecycle tests."""

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from villager_tts.app.factory import create_app
from villager_tts.config import Device, Settings
from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime import loader
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.runtime.state import Readiness
from villager_tts.core.tts.protocol import TtsRequest
from villager_tts.core.vc.worker import ConvertedAudio, RvcWorkerVoiceConverter


def test_loader_builds_development_backends() -> None:
    bundle = loader.load_engine_bundle(
        Settings(
            verify_model_artifacts=False,
            tts_backend="tone",
            vc_backend="passthrough",
        )
    )
    assert bundle.tts.name == "tone-development"
    assert bundle.voice_converter.name == "passthrough-development"


def test_loader_rejects_unknown_backends() -> None:
    with pytest.raises(ValueError, match="TTS backend"):
        loader.load_engine_bundle(
            Settings(
                verify_model_artifacts=False,
                tts_backend="unknown",
                vc_backend="passthrough",
            )
        )
    with pytest.raises(ValueError, match="voice-conversion"):
        loader.load_engine_bundle(
            Settings(
                verify_model_artifacts=False,
                tts_backend="tone",
                vc_backend="unknown",
            )
        )


def test_loader_selects_real_adapter_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    created_tts: list[dict[str, object]] = []

    def create_tts(**kwargs: object) -> _StubTts:
        created_tts.append(kwargs)
        return _StubTts(kwargs)

    monkeypatch.setattr(loader, "SherpaOnnxTtsEngine", create_tts)
    monkeypatch.setattr(
        RvcWorkerVoiceConverter,
        "from_settings",
        lambda settings: _StubConverter(settings.output_sample_rate),
    )
    bundle = loader.load_engine_bundle(
        Settings(
            verify_model_artifacts=False,
            max_concurrent_syntheses=2,
            tts_instance_count=2,
            tts_backend="sherpa_onnx",
            vc_backend="rvc_worker",
        )
    )
    assert bundle.tts.name == "stub-cpu"
    assert len(created_tts) == 4
    assert all(kwargs["provider"] == "cpu" for kwargs in created_tts)
    assert bundle.voice_converter.name == "stub-rvc"


async def test_application_lifespan_loads_and_stops_engines() -> None:
    app = create_app(
        Settings(
            verify_model_artifacts=False,
            tts_backend="tone",
            vc_backend="passthrough",
        )
    )
    async with app.router.lifespan_context(app):
        assert app.state.runtime.status is Readiness.READY
        assert app.state.speech_service is not None
    _assert_stopping(app.state.runtime.status)


async def test_application_lifespan_keeps_liveness_on_load_failure() -> None:
    app = create_app(
        Settings(
            verify_model_artifacts=False,
            tts_backend="unknown",
            vc_backend="passthrough",
        )
    )
    async with app.router.lifespan_context(app):
        assert app.state.runtime.status is Readiness.FAILED
        assert "unsupported TTS backend" in app.state.runtime.failure


def test_settings_reject_invalid_log_and_phrase_limits() -> None:
    with pytest.raises(ValidationError, match="log level"):
        Settings(log_level="verbose")
    with pytest.raises(ValidationError, match="phrase_max_characters"):
        Settings(phrase_target_characters=100, phrase_max_characters=50)
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(capacity_wait_seconds=0)
    with pytest.raises(ValidationError, match="rvc_worker_count"):
        Settings(max_concurrent_syntheses=1, rvc_worker_count=2)
    with pytest.raises(ValidationError, match="tts_instance_count"):
        Settings(max_concurrent_syntheses=1, tts_instance_count=2)


class _StubTts:
    def __init__(self, kwargs: dict[str, object]) -> None:
        self._provider = kwargs["provider"]

    @property
    def name(self) -> str:
        return f"stub-{self._provider}"

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        del request, cancellation
        yield AudioChunk(np.zeros(1, dtype=np.float32), 48_000, 0, is_final=True)


class _StubConverter:
    def __init__(self, output_rate: int) -> None:
        self._output_rate = output_rate

    @property
    def name(self) -> str:
        return "stub-rvc"

    async def convert(
        self,
        chunks: AsyncIterator[AudioChunk],
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        del cancellation
        async for chunk in chunks:
            yield AudioChunk(chunk.samples, self._output_rate, chunk.sequence, chunk.is_final)


class _SettingsClient:
    def convert(self, samples: NDArray[np.float32], sample_rate: int) -> ConvertedAudio:
        return ConvertedAudio(np.asarray(samples, dtype=np.float32), sample_rate)

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass


def test_rvc_converter_builds_worker_command_from_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []

    def client(command: list[str], *, startup_timeout_seconds: float) -> _SettingsClient:
        assert startup_timeout_seconds == 180
        captured.extend(command)
        return _SettingsClient()

    monkeypatch.setattr("villager_tts.core.vc.worker.ProcessRvcClient", client)
    converter = RvcWorkerVoiceConverter.from_settings(
        Settings(
            verify_model_artifacts=False,
            max_concurrent_syntheses=2,
            rvc_worker_count=2,
            rvc_device=Device.CUDA,
            rvc_model_path=tmp_path / "voice.pth",
            rvc_index_path=tmp_path / "voice.index",
            rvc_hubert_path=tmp_path / "hubert.pt",
            rvc_rmvpe_path=tmp_path / "rmvpe.pt",
            rvc_f0_method="pm",
            rvc_index_rate=0.6,
            rvc_filter_radius=5,
            rvc_rms_mix_rate=0.9,
        )
    )
    assert converter.name == "rvc-worker"
    assert captured.count("--model") == 2
    assert "cuda:0" in captured
    assert str(tmp_path / "voice.pth") in captured
    assert captured[captured.index("--f0-method") + 1] == "pm"
    assert captured[captured.index("--index-rate") + 1] == "0.6"
    assert captured[captured.index("--filter-radius") + 1] == "5"
    assert captured[captured.index("--rms-mix-rate") + 1] == "0.9"


def _assert_stopping(status: Readiness) -> None:
    assert status is Readiness.STOPPING
