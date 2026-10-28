"""Persistent, dependency-isolated RVC worker adapter."""

import asyncio
import concurrent.futures
import logging
import shlex
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from villager_rvc_protocol import ProtocolError, read_frame, write_frame

from villager_tts.common.errors import InferenceTimeoutError, SynthesisError
from villager_tts.config import Device, Settings
from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken

LOGGER = logging.getLogger(__name__)


class RvcWorkerError(RuntimeError):
    """A worker lifecycle, protocol, or inference failure."""


class RvcInferenceError(RvcWorkerError):
    """A conversion failure reported by an otherwise healthy worker."""


@dataclass(frozen=True, slots=True)
class ConvertedAudio:
    """One converted phrase returned by the worker."""

    samples: NDArray[np.float32]
    sample_rate: int


class RvcClient(Protocol):
    """Synchronous client boundary suitable for a worker thread."""

    def convert(self, samples: NDArray[np.float32], sample_rate: int) -> ConvertedAudio:
        """Convert one phrase."""
        ...

    def close(self) -> None:
        """Release worker resources."""
        ...

    def abort(self) -> None:
        """Terminate a conversion that must no longer continue."""
        ...


class ProcessRvcClient:
    """Length-framed RPC client for one persistent RVC subprocess."""

    def __init__(
        self,
        command: Sequence[str],
        environment: Mapping[str, str] | None = None,
        *,
        startup_timeout_seconds: float = 180,
    ) -> None:
        if not command:
            msg = "RVC worker command cannot be empty"
            raise ValueError(msg)
        startup_started = time.monotonic()
        self._closed = False
        self._process = subprocess.Popen(  # noqa: S603 - administrator-owned argv, no shell
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=environment,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            msg = "RVC worker pipes were not created"
            raise RvcWorkerError(msg)
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(read_frame, self._stdout)
        try:
            frame = future.result(timeout=startup_timeout_seconds)
        except concurrent.futures.TimeoutError as error:
            self.abort()
            msg = "RVC worker exceeded its startup deadline"
            raise RvcWorkerError(msg) from error
        except Exception as error:
            self.abort()
            msg = "RVC worker failed before advertising readiness"
            raise RvcWorkerError(msg) from error
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        header = frame.header
        if frame.payload or header.get("status") != "ready":
            self.close()
            msg = str(header.get("error", "RVC worker failed during startup"))
            raise RvcWorkerError(msg)
        LOGGER.info(
            "rvc worker ready",
            extra={
                "stage": "worker_startup",
                "duration_ms": _elapsed_ms(startup_started),
                "worker_pid": self._process.pid,
                "rvc_device": _optional_string(header, "device"),
                "torch_version": _optional_string(header, "torch_version"),
                "cuda_version": _optional_string(header, "cuda_version"),
                "gpu_name": _optional_string(header, "gpu_name"),
            },
        )

    def convert(self, samples: NDArray[np.float32], sample_rate: int) -> ConvertedAudio:
        """Convert one phrase while this client is exclusively leased from its pool."""
        contiguous = np.ascontiguousarray(samples, dtype="<f4")
        self._ensure_running()
        try:
            write_frame(
                self._stdin,
                {
                    "type": "convert",
                    "sample_rate": sample_rate,
                    "sample_count": int(contiguous.size),
                },
                contiguous.tobytes(),
            )
            frame = read_frame(self._stdout)
        except (EOFError, OSError, ProtocolError, ValueError) as error:
            msg = "RVC worker protocol exchange failed"
            raise RvcWorkerError(msg) from error
        header = frame.header
        payload = frame.payload
        if header.get("status") != "ok":
            msg = str(header.get("error", "RVC inference failed"))
            raise RvcInferenceError(msg)
        output_rate = header.get("sample_rate")
        output_count = header.get("sample_count")
        if type(output_rate) is not int or output_rate <= 0:
            msg = "RVC worker returned an invalid sample rate"
            raise RvcWorkerError(msg)
        if type(output_count) is not int or output_count < 0:
            msg = "RVC worker returned an invalid sample count"
            raise RvcWorkerError(msg)
        if len(payload) != output_count * 4:
            msg = "RVC worker payload length does not match its header"
            raise RvcWorkerError(msg)
        output = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
        return ConvertedAudio(samples=output, sample_rate=output_rate)

    def close(self) -> None:
        """Gracefully stop an idle worker."""
        if self._closed:
            return
        self._closed = True
        self._stop_process(self._process, graceful=True)
        self._close_pipes()

    def abort(self) -> None:
        """Terminate this worker so an in-flight blocking read is interrupted."""
        if self._closed:
            return
        self._closed = True
        self._stop_process(self._process, graceful=False)
        self._close_pipes()

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes], *, graceful: bool) -> None:
        if process.poll() is not None:
            return
        if graceful:
            try:
                if process.stdin is not None:
                    write_frame(process.stdin, {"type": "shutdown"})
            except (BrokenPipeError, OSError, ProtocolError):
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if graceful:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.kill()
                process.wait()

    def _close_pipes(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stdout is not None:
            self._process.stdout.close()

    def _ensure_running(self) -> None:
        if self._closed or self._process.poll() is not None:
            msg = "RVC worker is not running"
            raise RvcWorkerError(msg)


RvcClientFactory = Callable[[], RvcClient]


class RvcWorkerPool:
    """Lease independent persistent workers without sharing a conversion lock."""

    def __init__(
        self,
        clients: Sequence[RvcClient],
        *,
        factory: RvcClientFactory | None = None,
    ) -> None:
        if not clients:
            msg = "RVC worker pool requires at least one client"
            raise ValueError(msg)
        self._available: asyncio.Queue[RvcClient] = asyncio.Queue(maxsize=len(clients))
        self._clients = list(clients)
        self._factory = factory
        self._replacement_tasks: set[asyncio.Task[None]] = set()
        self._closing = False
        for client in clients:
            self._available.put_nowait(client)

    @property
    def size(self) -> int:
        """Return the number of usable workers currently owned by the pool."""
        return len(self._clients)

    async def acquire(self) -> RvcClient:
        """Wait until one worker is exclusively available."""
        if self._closing:
            msg = "RVC worker pool is closed"
            raise RvcWorkerError(msg)
        return await self._available.get()

    async def release(self, client: RvcClient) -> None:
        """Return a healthy worker to the available queue."""
        if self._closing:
            await asyncio.to_thread(client.close)
            return
        self._available.put_nowait(client)

    async def retire(self, client: RvcClient) -> None:
        """Abort one unhealthy worker and start an isolated replacement."""
        await asyncio.to_thread(client.abort)
        with suppress(ValueError):
            self._clients.remove(client)
        if self._closing or self._factory is None:
            return
        task = asyncio.create_task(self._replace_one())
        self._replacement_tasks.add(task)
        task.add_done_callback(self._replacement_tasks.discard)

    async def close(self) -> None:
        """Stop replacement work and every worker owned by this pool."""
        if self._closing:
            return
        self._closing = True
        if self._replacement_tasks:
            await asyncio.gather(*tuple(self._replacement_tasks), return_exceptions=True)
        clients = tuple(self._clients)
        self._clients.clear()
        await asyncio.gather(
            *(asyncio.to_thread(client.close) for client in clients),
            return_exceptions=True,
        )

    async def _replace_one(self) -> None:
        factory = self._factory
        if factory is None:
            return
        try:
            client = await asyncio.to_thread(factory)
        except Exception as error:
            LOGGER.exception(
                "rvc worker replacement failed",
                extra={"stage": "worker_startup", "error_kind": type(error).__name__},
            )
            return
        if self._closing:
            await asyncio.to_thread(client.close)
            return
        self._clients.append(client)
        self._available.put_nowait(client)
        LOGGER.info(
            "rvc worker replaced",
            extra={"stage": "worker_startup", "worker_pool_size": self.size},
        )


class RvcWorkerVoiceConverter:
    """Convert each completed TTS phrase through a persistent RVC process."""

    def __init__(
        self,
        client_or_pool: RvcClient | RvcWorkerPool,
        *,
        output_sample_rate: int,
        timeout_seconds: float,
        crossfade_milliseconds: int,
        output_chunk_milliseconds: int,
    ) -> None:
        self._pool = (
            client_or_pool
            if isinstance(client_or_pool, RvcWorkerPool)
            else RvcWorkerPool([client_or_pool])
        )
        self._output_sample_rate = output_sample_rate
        self._timeout_seconds = timeout_seconds
        self._crossfade_samples = output_sample_rate * crossfade_milliseconds // 1_000
        self._output_chunk_samples = output_sample_rate * output_chunk_milliseconds // 1_000

    @classmethod
    def from_settings(cls, settings: Settings) -> "RvcWorkerVoiceConverter":
        """Build the process command from validated administrator settings."""
        device = {
            Device.AUTO: "auto",
            Device.CPU: "cpu",
            Device.CUDA: "cuda:0",
        }[settings.rvc_device]
        command = [
            *shlex.split(settings.rvc_worker_command),
            "--model",
            str(settings.rvc_model_path),
            "--index",
            str(settings.rvc_index_path),
            "--hubert",
            str(settings.rvc_hubert_path),
            "--rmvpe",
            str(settings.rvc_rmvpe_path),
            "--device",
            device,
            "--speaker-id",
            str(settings.rvc_speaker_id),
            "--pitch",
            str(settings.rvc_pitch),
            "--f0-method",
            settings.rvc_f0_method,
            "--index-rate",
            str(settings.rvc_index_rate),
            "--filter-radius",
            str(settings.rvc_filter_radius),
            "--rms-mix-rate",
            str(settings.rvc_rms_mix_rate),
            "--protect",
            str(settings.rvc_protect),
        ]

        def create_client() -> RvcClient:
            return ProcessRvcClient(
                command,
                startup_timeout_seconds=settings.rvc_startup_timeout_seconds,
            )

        clients: list[RvcClient] = []
        try:
            for _ in range(settings.rvc_worker_count):
                clients.append(create_client())
        except Exception:
            for client in clients:
                client.close()
            raise
        pool = RvcWorkerPool(clients, factory=create_client)
        LOGGER.info(
            "rvc worker pool ready",
            extra={"stage": "worker_startup", "worker_pool_size": pool.size},
        )
        return cls(
            pool,
            output_sample_rate=settings.output_sample_rate,
            timeout_seconds=settings.rvc_request_timeout_seconds,
            crossfade_milliseconds=settings.rvc_crossfade_milliseconds,
            output_chunk_milliseconds=settings.output_chunk_milliseconds,
        )

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "rvc-worker"

    async def convert(  # noqa: C901 - phrase streaming has explicit state transitions
        self,
        chunks: AsyncIterator[AudioChunk],
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Convert phrase blocks and crossfade adjacent RVC results."""
        phrase: list[NDArray[np.float32]] = []
        input_rate: int | None = None
        pending_tail = np.empty(0, dtype=np.float32)
        sequence = 0
        phrase_index = 0

        async for chunk in chunks:
            cancellation.raise_if_cancelled()
            if input_rate is None:
                input_rate = chunk.sample_rate
            elif chunk.sample_rate != input_rate:
                msg = "TTS sample rate changed within a phrase"
                raise SynthesisError(msg)
            phrase.append(chunk.samples)
            if not chunk.is_final:
                continue

            source = np.concatenate(phrase)
            phrase_index += 1
            conversion_started = time.monotonic()
            converted = await self._convert_phrase(
                source,
                input_rate,
                phrase_index=phrase_index,
                started=conversion_started,
            )
            cancellation.raise_if_cancelled()
            if converted.sample_rate != self._output_sample_rate:
                msg = f"RVC worker returned {converted.sample_rate} Hz audio"
                raise SynthesisError(msg)
            LOGGER.info(
                "rvc phrase completed",
                extra={
                    "stage": "rvc",
                    "duration_ms": _elapsed_ms(conversion_started),
                    "phrase_index": phrase_index,
                    "input_sample_rate": input_rate,
                    "output_sample_rate": converted.sample_rate,
                    "input_samples": source.size,
                    "output_samples": converted.samples.size,
                    "input_audio_ms": round(source.size / input_rate * 1_000, 2),
                    "output_audio_ms": round(
                        converted.samples.size / converted.sample_rate * 1_000,
                        2,
                    ),
                },
            )

            combined = _crossfade(pending_tail, converted.samples)
            tail_size = min(self._crossfade_samples, combined.size)
            ready_size = combined.size - tail_size
            if ready_size:
                async for output in self._chunk_audio(
                    combined[:ready_size],
                    sequence,
                    final=self._crossfade_samples == 0,
                ):
                    yield output
                    sequence += 1
            pending_tail = combined[ready_size:].copy()
            phrase = []
            input_rate = None

        if phrase:
            msg = "TTS stream ended without a phrase boundary"
            raise SynthesisError(msg)
        if pending_tail.size:
            async for output in self._chunk_audio(pending_tail, sequence, final=True):
                yield output

    async def aclose(self) -> None:
        """Stop every persistent worker without blocking the event loop."""
        await self._pool.close()

    async def _convert_phrase(
        self,
        source: NDArray[np.float32],
        input_rate: int,
        *,
        phrase_index: int,
        started: float,
    ) -> ConvertedAudio:
        client: RvcClient | None = None
        task: asyncio.Task[ConvertedAudio] | None = None
        reusable = False
        try:
            async with asyncio.timeout(self._timeout_seconds):
                client = await self._pool.acquire()
                task = asyncio.create_task(
                    asyncio.to_thread(client.convert, source, input_rate),
                )
                try:
                    converted = await asyncio.shield(task)
                except RvcInferenceError:
                    reusable = True
                    raise
                reusable = True
                return converted
        except TimeoutError as error:
            self._log_conversion_failure(
                "rvc phrase timed out",
                error,
                source=source,
                input_rate=input_rate,
                phrase_index=phrase_index,
                started=started,
            )
            msg = "RVC worker exceeded its inference deadline."
            raise InferenceTimeoutError(msg) from error
        except RvcWorkerError as error:
            self._log_conversion_failure(
                "rvc phrase failed",
                error,
                source=source,
                input_rate=input_rate,
                phrase_index=phrase_index,
                started=started,
            )
            msg = f"RVC worker failed: {error}"
            raise SynthesisError(msg) from error
        finally:
            if client is not None:
                if reusable:
                    await self._pool.release(client)
                else:
                    await self._pool.retire(client)
                    if task is not None:
                        with suppress(asyncio.CancelledError, Exception):
                            await task

    @staticmethod
    def _log_conversion_failure(
        message: str,
        error: BaseException,
        *,
        source: NDArray[np.float32],
        input_rate: int,
        phrase_index: int,
        started: float,
    ) -> None:
        LOGGER.warning(
            message,
            extra={
                "stage": "rvc",
                "duration_ms": _elapsed_ms(started),
                "phrase_index": phrase_index,
                "input_sample_rate": input_rate,
                "input_samples": source.size,
                "error_kind": type(error).__name__,
            },
        )

    async def _chunk_audio(
        self,
        samples: NDArray[np.float32],
        first_sequence: int,
        *,
        final: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        starts = range(0, samples.size, self._output_chunk_samples)
        for offset, start in enumerate(starts):
            end = min(samples.size, start + self._output_chunk_samples)
            yield AudioChunk(
                samples[start:end],
                sample_rate=self._output_sample_rate,
                sequence=first_sequence + offset,
                is_final=final and end == samples.size,
            )


def _crossfade(
    previous_tail: NDArray[np.float32],
    current: NDArray[np.float32],
) -> NDArray[np.float32]:
    if not previous_tail.size:
        return current
    overlap = min(previous_tail.size, current.size)
    if overlap == 0:
        return np.concatenate((previous_tail, current))
    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    blended = previous_tail[-overlap:] * (1.0 - fade_in) + current[:overlap] * fade_in
    prefix = previous_tail[:-overlap]
    result = np.concatenate((prefix, blended, current[overlap:])).astype(np.float32, copy=False)
    return cast("NDArray[np.float32]", result)


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1_000, 2)


def _optional_string(header: Mapping[str, object], field: str) -> str | None:
    value = header.get(field)
    return value if isinstance(value, str) else None
