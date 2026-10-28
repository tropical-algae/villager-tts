"""Inference adapter tests that do not require model files."""

import asyncio
import importlib
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from villager_tts.common.errors import InferenceTimeoutError, SynthesisError
from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.audio.resampling import resample_stream
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.tts.pool import TtsEnginePool
from villager_tts.core.tts.protocol import TtsRequest
from villager_tts.core.tts.selector import GenderedTtsEngine
from villager_tts.core.tts.sherpa import SherpaOnnxTtsEngine
from villager_tts.core.tts.tone import ToneTtsEngine
from villager_tts.core.vc.passthrough import PassthroughVoiceConverter
from villager_tts.core.vc.worker import (
    ConvertedAudio,
    ProcessRvcClient,
    RvcWorkerError,
    RvcWorkerPool,
    RvcWorkerVoiceConverter,
)


async def _chunks(*chunks: AudioChunk) -> AsyncIterator[AudioChunk]:
    for chunk in chunks:
        yield chunk


async def test_resample_stream_preserves_state_and_rejects_rate_changes() -> None:
    source = _chunks(
        AudioChunk(np.ones(100, dtype=np.float32), 1_000, 0),
        AudioChunk(np.ones(100, dtype=np.float32), 1_000, 1, is_final=True),
    )
    converted = [chunk async for chunk in resample_stream(source, output_rate=2_000)]
    assert sum(chunk.samples.size for chunk in converted) == pytest.approx(400, abs=2)
    assert converted[-1].is_final

    unchanged = [
        chunk
        async for chunk in resample_stream(
            _chunks(AudioChunk(np.ones(2, dtype=np.float32), 48_000, 0, is_final=True)),
            output_rate=48_000,
        )
    ]
    assert unchanged[0].samples.tolist() == [1.0, 1.0]

    changing = _chunks(
        AudioChunk(np.ones(2, dtype=np.float32), 24_000, 0),
        AudioChunk(np.ones(2, dtype=np.float32), 48_000, 1),
    )
    with pytest.raises(ValueError, match="changed"):
        async for _ in resample_stream(changing, output_rate=48_000):
            pass


async def test_tone_and_passthrough_backends_form_a_valid_pipeline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="villager_tts.core.audio.resampling")
    token = CancellationToken()
    tone = ToneTtsEngine(sample_rate=24_000)
    converter = PassthroughVoiceConverter(output_sample_rate=48_000)
    source = tone.synthesize(TtsRequest("hello", 1.0), token)
    output = [chunk async for chunk in converter.convert(source, token)]

    assert tone.name == "tone-development"
    assert converter.name == "passthrough-development"
    assert output[0].sample_rate == 48_000
    assert output[0].samples.size > 0
    resampling_record = next(
        record for record in caplog.records if getattr(record, "stage", None) == "resampling"
    )
    assert vars(resampling_record)["input_sample_rate"] == 24_000
    assert vars(resampling_record)["output_sample_rate"] == 48_000


async def test_passthrough_yields_before_the_source_phrase_finishes() -> None:
    release_source = asyncio.Event()

    async def source() -> AsyncIterator[AudioChunk]:
        yield AudioChunk(np.ones(4, dtype=np.float32), 48_000, 0)
        await release_source.wait()
        yield AudioChunk(np.ones(4, dtype=np.float32), 48_000, 1, is_final=True)

    converter = PassthroughVoiceConverter(output_sample_rate=48_000)
    output = converter.convert(source(), CancellationToken())
    first = await asyncio.wait_for(anext(output), timeout=0.1)

    assert first.samples.tolist() == [1.0] * 4
    assert not first.is_final
    release_source.set()
    remaining = [chunk async for chunk in output]
    assert remaining[-1].is_final


async def test_resampler_resets_state_at_phrase_boundaries() -> None:
    source = _chunks(
        AudioChunk(np.ones(8, dtype=np.float32), 24_000, 0, is_final=True),
        AudioChunk(np.ones(8, dtype=np.float32), 48_000, 1, is_final=True),
    )

    converted = [chunk async for chunk in resample_stream(source, output_rate=48_000)]

    assert len([chunk for chunk in converted if chunk.is_final]) == 2
    assert all(chunk.sample_rate == 48_000 for chunk in converted)


async def test_gendered_tts_routes_to_the_requested_single_speaker() -> None:
    male = _FixedTts("male", 0.25)
    female = _FixedTts("female", 0.5)
    selector = GenderedTtsEngine(male=male, female=female)

    output = [
        chunk
        async for chunk in selector.synthesize(
            TtsRequest("你好", 1.0, gender="female"),
            CancellationToken(),
        )
    ]

    assert selector.name == "gendered-tts"
    assert output[0].samples.tolist() == [0.5]


class _FixedTts:
    def __init__(self, name: str, sample: float) -> None:
        self._name = name
        self._sample = sample

    @property
    def name(self) -> str:
        return self._name

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        del request
        cancellation.raise_if_cancelled()
        yield AudioChunk(np.asarray([self._sample], dtype=np.float32), 16_000, 0, is_final=True)


class _TtsConcurrencyState:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.both_started = asyncio.Event()


class _CoordinatedTts(_FixedTts):
    def __init__(self, state: _TtsConcurrencyState, sample: float) -> None:
        super().__init__("coordinated", sample)
        self._state = state

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        del request
        cancellation.raise_if_cancelled()
        self._state.active += 1
        self._state.maximum_active = max(self._state.maximum_active, self._state.active)
        if self._state.active == 2:
            self._state.both_started.set()
        try:
            await self._state.both_started.wait()
            yield AudioChunk(np.asarray([self._sample], dtype=np.float32), 16_000, 0, is_final=True)
        finally:
            self._state.active -= 1


async def _collect_tts(engine: TtsEnginePool, text: str) -> list[AudioChunk]:
    return [
        chunk
        async for chunk in engine.synthesize(
            TtsRequest(text, 1.0),
            CancellationToken(),
        )
    ]


async def test_tts_pool_uses_distinct_instances_for_parallel_requests() -> None:
    state = _TtsConcurrencyState()
    pool = TtsEnginePool(
        [
            _CoordinatedTts(state, 0.1),
            _CoordinatedTts(state, 0.2),
        ]
    )

    first, second = await asyncio.wait_for(
        asyncio.gather(_collect_tts(pool, "first"), _collect_tts(pool, "second")),
        timeout=0.5,
    )

    assert pool.name == "coordinated"
    assert state.maximum_active == 2
    assert sorted([first[0].samples[0], second[0].samples[0]]) == pytest.approx([0.1, 0.2])


def test_tts_pool_rejects_an_empty_instance_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TtsEnginePool([])


class _FakeSherpaTts:
    sample_rate = 1_000
    callback_enabled = True
    error: Exception | None = None

    def generate(
        self,
        text: str,
        sid: int,
        speed: float,
        callback: Any,
    ) -> SimpleNamespace:
        del text, sid, speed
        if self.error is not None:
            raise self.error
        samples = np.arange(6, dtype=np.float32) / 10
        if self.callback_enabled:
            callback(samples[:3], 0.5)
            callback(samples[3:], 1.0)
        return SimpleNamespace(samples=samples, sample_rate=self.sample_rate)


class _Config:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def validate(self) -> bool:
        return True


def _fake_sherpa_module() -> SimpleNamespace:
    return SimpleNamespace(
        OfflineTtsVitsModelConfig=_Config,
        OfflineTtsModelConfig=_Config,
        OfflineTtsConfig=_Config,
        OfflineTts=lambda _: _FakeSherpaTts(),
    )


def _make_sherpa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SherpaOnnxTtsEngine:
    model = tmp_path / "model.onnx"
    tokens = tmp_path / "tokens.txt"
    lexicon = tmp_path / "lexicon.txt"
    data_dir = tmp_path / "dict"
    model.write_bytes(b"model")
    tokens.write_text("tokens", encoding="utf-8")
    lexicon.write_text("lexicon", encoding="utf-8")
    data_dir.mkdir()
    date_fst = tmp_path / "date.fst"
    number_fst = tmp_path / "number.fst"
    date_fst.write_bytes(b"fst")
    number_fst.write_bytes(b"fst")
    original = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: _fake_sherpa_module() if name == "sherpa_onnx" else original(name),
    )
    return SherpaOnnxTtsEngine(
        model=model,
        tokens=tokens,
        lexicon=lexicon,
        data_dir=data_dir,
        rule_fsts=[date_fst, number_fst],
        speaker_id=0,
        num_threads=1,
        provider="cpu",
        warmup_text=None,
        chunk_milliseconds=2,
    )


async def test_sherpa_adapter_streams_callbacks_and_falls_back_to_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_sherpa(tmp_path, monkeypatch)
    output = [
        chunk
        async for chunk in engine.synthesize(
            TtsRequest("hello", 1.0),
            CancellationToken(),
        )
    ]
    assert engine.name == "sherpa-onnx"
    assert [chunk.samples.size for chunk in output] == [2, 1, 2, 1]

    fake_tts = cast("_FakeSherpaTts", engine._tts)  # noqa: SLF001
    fake_tts.callback_enabled = False
    fallback = [
        chunk
        async for chunk in engine.synthesize(
            TtsRequest("hello", 1.0),
            CancellationToken(),
        )
    ]
    assert sum(chunk.samples.size for chunk in fallback) == 6


async def test_sherpa_adapter_propagates_generation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_sherpa(tmp_path, monkeypatch)
    fake_tts = cast("_FakeSherpaTts", engine._tts)  # noqa: SLF001
    fake_tts.error = RuntimeError("generation failed")
    with pytest.raises(RuntimeError, match="generation failed"):
        async for _ in engine.synthesize(
            TtsRequest("hello", 1.0),
            CancellationToken(),
        ):
            pass


class _FakeRvcClient:
    def __init__(self, *, rate: int = 1_000, delay: float = 0) -> None:
        self.rate = rate
        self.delay = delay
        self.closed = False
        self.calls = 0

    def convert(
        self, samples: np.ndarray[Any, np.dtype[np.float32]], sample_rate: int
    ) -> ConvertedAudio:
        del sample_rate
        self.calls += 1
        time.sleep(self.delay)
        return ConvertedAudio(samples=samples + 0.1, sample_rate=self.rate)

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.closed = True


class _ConcurrentState:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()
        self.both_started = threading.Event()


class _ConcurrentRvcClient(_FakeRvcClient):
    def __init__(self, state: _ConcurrentState) -> None:
        super().__init__()
        self._state = state

    def convert(
        self, samples: np.ndarray[Any, np.dtype[np.float32]], sample_rate: int
    ) -> ConvertedAudio:
        with self._state.lock:
            self._state.active += 1
            self._state.maximum_active = max(self._state.maximum_active, self._state.active)
            if self._state.active == 2:
                self._state.both_started.set()
        try:
            if not self._state.both_started.wait(timeout=0.5):
                raise RuntimeError("conversions were serialized")
            return ConvertedAudio(samples=samples, sample_rate=sample_rate)
        finally:
            with self._state.lock:
                self._state.active -= 1


class _AbortableRvcClient(_FakeRvcClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self._aborted = threading.Event()

    def convert(
        self, samples: np.ndarray[Any, np.dtype[np.float32]], sample_rate: int
    ) -> ConvertedAudio:
        del samples, sample_rate
        self.started.set()
        if not self._aborted.wait(timeout=1):
            raise RuntimeError("test worker was not aborted")
        raise RvcWorkerError("worker aborted")

    def abort(self) -> None:
        super().abort()
        self._aborted.set()


async def _collect_conversion(
    converter: RvcWorkerVoiceConverter,
    value: float = 0,
) -> list[AudioChunk]:
    return [
        chunk
        async for chunk in converter.convert(
            _chunks(
                AudioChunk(
                    np.full(4, value, dtype=np.float32),
                    1_000,
                    0,
                    is_final=True,
                )
            ),
            CancellationToken(),
        )
    ]


async def test_rvc_worker_pool_runs_independent_requests_concurrently() -> None:
    state = _ConcurrentState()
    clients = [_ConcurrentRvcClient(state), _ConcurrentRvcClient(state)]
    converter = RvcWorkerVoiceConverter(
        RvcWorkerPool(clients),
        output_sample_rate=1_000,
        timeout_seconds=1,
        crossfade_milliseconds=0,
        output_chunk_milliseconds=20,
    )

    first, second = await asyncio.gather(
        _collect_conversion(converter, 0.1),
        _collect_conversion(converter, 0.2),
    )

    assert state.maximum_active == 2
    assert first[0].samples.tolist() == pytest.approx([0.1] * 4)
    assert second[0].samples.tolist() == pytest.approx([0.2] * 4)
    await converter.aclose()


async def test_cancelled_rvc_request_retires_and_replaces_its_worker() -> None:
    blocked = _AbortableRvcClient()
    replacements: list[_FakeRvcClient] = []

    def replace() -> _FakeRvcClient:
        client = _FakeRvcClient()
        replacements.append(client)
        return client

    converter = RvcWorkerVoiceConverter(
        RvcWorkerPool([blocked], factory=replace),
        output_sample_rate=1_000,
        timeout_seconds=1,
        crossfade_milliseconds=0,
        output_chunk_milliseconds=20,
    )
    stream = converter.convert(
        _chunks(AudioChunk(np.zeros(4, dtype=np.float32), 1_000, 0, is_final=True)),
        CancellationToken(),
    )
    pending: asyncio.Future[AudioChunk] = asyncio.ensure_future(anext(stream))
    assert await asyncio.to_thread(blocked.started.wait, 0.5)

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    output = await asyncio.wait_for(_collect_conversion(converter), timeout=0.5)
    assert blocked.closed
    assert len(replacements) == 1
    assert output[-1].is_final
    await converter.aclose()
    assert replacements[0].closed


async def test_rvc_converter_processes_phrases_crossfades_and_closes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="villager_tts.core.vc.worker")
    client = _FakeRvcClient()
    converter = RvcWorkerVoiceConverter(
        client,
        output_sample_rate=1_000,
        timeout_seconds=1,
        crossfade_milliseconds=2,
        output_chunk_milliseconds=3,
    )
    source = _chunks(
        AudioChunk(np.zeros(6, dtype=np.float32), 24_000, 0, is_final=True),
        AudioChunk(np.ones(6, dtype=np.float32), 24_000, 1, is_final=True),
    )
    output = [chunk async for chunk in converter.convert(source, CancellationToken())]

    assert converter.name == "rvc-worker"
    assert client.calls == 2
    assert sum(chunk.samples.size for chunk in output) == 10
    assert output[-1].is_final
    assert [chunk.sequence for chunk in output] == list(range(len(output)))
    conversion_records = [
        record for record in caplog.records if getattr(record, "stage", None) == "rvc"
    ]
    assert len(conversion_records) == 2
    assert vars(conversion_records[0])["input_sample_rate"] == 24_000
    assert vars(conversion_records[0])["output_sample_rate"] == 1_000
    await converter.aclose()
    assert client.closed


async def test_rvc_converter_rejects_bad_rate_and_times_out() -> None:
    bad_rate = RvcWorkerVoiceConverter(
        _FakeRvcClient(rate=2_000),
        output_sample_rate=1_000,
        timeout_seconds=1,
        crossfade_milliseconds=0,
        output_chunk_milliseconds=20,
    )
    with pytest.raises(SynthesisError, match=r"2,000|2000"):
        async for _ in bad_rate.convert(
            _chunks(AudioChunk(np.zeros(2, dtype=np.float32), 1_000, 0, is_final=True)),
            CancellationToken(),
        ):
            pass

    slow_client = _FakeRvcClient(delay=0.02)
    timeout = RvcWorkerVoiceConverter(
        slow_client,
        output_sample_rate=1_000,
        timeout_seconds=0.001,
        crossfade_milliseconds=0,
        output_chunk_milliseconds=20,
    )
    with pytest.raises(InferenceTimeoutError, match="deadline") as timeout_error:
        async for _ in timeout.convert(
            _chunks(AudioChunk(np.zeros(2, dtype=np.float32), 1_000, 0, is_final=True)),
            CancellationToken(),
        ):
            pass
    assert timeout_error.value.status_code == 504
    assert slow_client.closed


def test_process_rvc_client_uses_length_framed_protocol() -> None:
    worker = Path(__file__).parents[1] / "fixtures" / "fake_rvc_worker.py"
    client = ProcessRvcClient([sys.executable, str(worker)])
    source = np.array([-0.5, 0.5], dtype=np.float32)
    converted = client.convert(source, 24_000)
    assert converted.sample_rate == 48_000
    assert np.array_equal(converted.samples, source)
    client.close()
    client.close()
    with pytest.raises(RvcWorkerError, match="not running"):
        client.convert(source, 24_000)
