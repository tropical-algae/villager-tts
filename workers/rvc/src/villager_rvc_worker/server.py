"""Long-lived request loop for an initialized RVC engine."""

import traceback
from collections.abc import Mapping
from typing import BinaryIO, Protocol, TextIO

import numpy as np
from numpy.typing import NDArray
from villager_rvc_protocol import ProtocolError, read_frame

from villager_rvc_worker.engine import ConvertedAudio
from villager_rvc_worker.protocol import (
    ShutdownRequest,
    decode_request,
    write_audio,
    write_error,
    write_ready,
)


class InferenceEngine(Protocol):
    """Minimal engine contract consumed by the protocol server."""

    @property
    def runtime_metadata(self) -> Mapping[str, object]:
        """Return runtime details included in the readiness frame."""
        ...

    def convert(self, samples: NDArray[np.float32], sample_rate: int) -> ConvertedAudio:
        """Convert one complete phrase."""
        ...


class WorkerServer:
    """Serve framed conversion requests until EOF or graceful shutdown."""

    def __init__(
        self,
        engine: InferenceEngine,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        error_stream: TextIO,
    ) -> None:
        self._engine = engine
        self._input = input_stream
        self._output = output_stream
        self._error = error_stream

    def serve_forever(self) -> None:
        """Advertise readiness and process requests sequentially."""
        write_ready(self._output, dict(self._engine.runtime_metadata))
        while True:
            try:
                frame = read_frame(self._input)
            except EOFError:
                return
            except ProtocolError as error:
                self._report_error(error)
                return

            try:
                request = decode_request(frame)
                if isinstance(request, ShutdownRequest):
                    return
                converted = self._engine.convert(request.samples, request.sample_rate)
            except Exception as error:  # noqa: BLE001 - errors must cross the process boundary
                self._report_error(error)
                continue
            write_audio(self._output, converted.samples, converted.sample_rate)

    def _report_error(self, error: Exception) -> None:
        traceback.print_exception(error, file=self._error)
        write_error(self._output, error)
