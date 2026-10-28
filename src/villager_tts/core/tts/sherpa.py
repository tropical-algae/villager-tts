"""sherpa-onnx VITS/Melo adapter."""

import asyncio
import importlib
import queue
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.tts.protocol import TtsRequest

NDArrayFloat32 = np.ndarray[Any, np.dtype[np.float32]]


class _GeneratedAudio(Protocol):
    samples: list[float]
    sample_rate: int


class _OfflineTts(Protocol):
    sample_rate: int

    def generate(
        self,
        text: str,
        sid: int,
        speed: float,
        callback: Callable[[NDArrayFloat32, float], int] | None = None,
    ) -> _GeneratedAudio: ...


class _CallbackBridge:
    """Apply bounded backpressure between sherpa's thread and asyncio."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        cancellation: CancellationToken,
    ) -> None:
        self._items: queue.Queue[NDArrayFloat32 | Exception | None] = queue.Queue(maxsize=4)
        self._available = asyncio.Event()
        self.stopped = threading.Event()
        self.callback_count = 0
        self._loop = loop
        self._cancellation = cancellation

    def enqueue(self, item: NDArrayFloat32 | Exception | None) -> bool:
        """Block the producer when the bounded cross-thread queue is full."""
        if self.stopped.is_set() or self._loop.is_closed():
            return False
        while not self.stopped.is_set():
            try:
                self._items.put(item, timeout=0.1)
            except queue.Full:
                continue
            try:
                self._loop.call_soon_threadsafe(self._available.set)
            except RuntimeError:
                return False
            return True
        return False

    async def dequeue(self) -> NDArrayFloat32 | Exception | None:
        """Receive one item without blocking the event loop on the thread queue."""
        while True:
            try:
                return self._items.get_nowait()
            except queue.Empty:
                self._available.clear()
                try:
                    return self._items.get_nowait()
                except queue.Empty:
                    await self._available.wait()

    def callback(self, samples: NDArrayFloat32, progress: float) -> int:
        """Receive one sherpa callback block on its inference thread."""
        del progress
        if self.stopped.is_set() or self._cancellation.is_cancelled:
            return 0
        self.callback_count += 1
        copied = np.asarray(samples, dtype=np.float32).copy()
        return int(self.enqueue(copied))


def _run_generation(
    tts: _OfflineTts,
    request: TtsRequest,
    speaker_id: int,
    bridge: _CallbackBridge,
) -> None:
    try:
        generated = tts.generate(request.text, speaker_id, request.speed, bridge.callback)
        if bridge.callback_count == 0 and len(generated.samples):
            fallback = np.asarray(generated.samples, dtype=np.float32).copy()
            bridge.enqueue(fallback)
    except Exception as error:  # noqa: BLE001 - propagate across the thread boundary
        bridge.enqueue(error)
    finally:
        bridge.enqueue(None)


class SherpaOnnxTtsEngine:
    """Run a sherpa-onnx VITS-compatible model off the ASGI event loop."""

    def __init__(
        self,
        *,
        model: Path,
        tokens: Path,
        lexicon: Path | None,
        data_dir: Path | None,
        rule_fsts: Sequence[Path],
        speaker_id: int,
        num_threads: int,
        provider: str,
        warmup_text: str | None,
        chunk_milliseconds: int = 200,
    ) -> None:
        self._validate_file(model, "TTS model")
        self._validate_file(tokens, "TTS tokens")
        if lexicon is not None:
            self._validate_file(lexicon, "TTS lexicon")
        if data_dir is not None and not data_dir.is_dir():
            msg = f"TTS data directory is missing: {data_dir}"
            raise FileNotFoundError(msg)
        for rule_fst in rule_fsts:
            self._validate_file(rule_fst, "TTS text normalization FST")

        module: Any = importlib.import_module("sherpa_onnx")
        vits = module.OfflineTtsVitsModelConfig(
            model=str(model),
            lexicon=str(lexicon) if lexicon is not None else "",
            tokens=str(tokens),
            data_dir=str(data_dir) if data_dir is not None else "",
        )
        model_config = module.OfflineTtsModelConfig(
            vits=vits,
            num_threads=num_threads,
            debug=False,
            provider=provider,
        )
        config = module.OfflineTtsConfig(
            model=model_config,
            rule_fsts=",".join(map(str, rule_fsts)),
            max_num_sentences=1,
        )
        if not config.validate():
            msg = "sherpa-onnx rejected the TTS configuration"
            raise ValueError(msg)
        self._tts = cast("_OfflineTts", module.OfflineTts(config))
        self._speaker_id = speaker_id
        self._chunk_milliseconds = chunk_milliseconds
        if warmup_text:
            self._tts.generate(warmup_text, speaker_id, 1.0)

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "sherpa-onnx"

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Bridge sherpa's worker-thread callback into an async PCM stream."""
        cancellation.raise_if_cancelled()
        bridge = _CallbackBridge(asyncio.get_running_loop(), cancellation)
        generation = asyncio.create_task(
            asyncio.to_thread(
                _run_generation,
                self._tts,
                request,
                self._speaker_id,
                bridge,
            )
        )
        sequence = 0
        try:
            while True:
                item = await bridge.dequeue()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                cancellation.raise_if_cancelled()
                chunk_size = max(1, self._tts.sample_rate * self._chunk_milliseconds // 1_000)
                for start in range(0, item.size, chunk_size):
                    end = min(item.size, start + chunk_size)
                    yield AudioChunk(
                        samples=item[start:end],
                        sample_rate=self._tts.sample_rate,
                        sequence=sequence,
                    )
                    sequence += 1
            await generation
        finally:
            bridge.stopped.set()
            if not generation.done():
                generation.cancel()
            with suppress(asyncio.CancelledError):
                await generation

    @staticmethod
    def _validate_file(path: Path, label: str) -> None:
        if not path.is_file():
            msg = f"{label} is missing: {path}"
            raise FileNotFoundError(msg)
