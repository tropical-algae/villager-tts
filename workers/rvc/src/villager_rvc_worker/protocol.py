"""Length-framed binary protocol used by the persistent RVC worker."""

from dataclasses import dataclass
from typing import BinaryIO, TypeAlias

import numpy as np
from numpy.typing import NDArray
from villager_rvc_protocol import (
    Frame,
    Header,
    ProtocolError,
    write_frame,
)

_FLOAT32_BYTES = np.dtype("<f4").itemsize


@dataclass(frozen=True, slots=True)
class ConvertRequest:
    """Validated mono float32 audio submitted for conversion."""

    samples: NDArray[np.float32]
    sample_rate: int


@dataclass(frozen=True, slots=True)
class ShutdownRequest:
    """Graceful worker shutdown request."""


WorkerRequest: TypeAlias = ConvertRequest | ShutdownRequest


def decode_request(frame: Frame) -> WorkerRequest:
    """Validate a generic frame and return its typed request."""
    message_type = frame.header.get("type")
    if message_type == "shutdown":
        if frame.payload:
            raise ProtocolError("shutdown request must not contain audio")
        return ShutdownRequest()
    if message_type != "convert":
        raise ProtocolError("unsupported RVC worker request")

    sample_rate = frame.header.get("sample_rate")
    sample_count = frame.header.get("sample_count")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ProtocolError("invalid input sample rate")
    if type(sample_count) is not int or sample_count < 1:
        raise ProtocolError("invalid input sample count")
    if len(frame.payload) != sample_count * _FLOAT32_BYTES:
        raise ProtocolError("input payload length mismatch")
    samples = np.frombuffer(frame.payload, dtype="<f4").astype(np.float32, copy=True)
    if not np.all(np.isfinite(samples)):
        raise ProtocolError("input audio contains non-finite samples")
    return ConvertRequest(samples=samples, sample_rate=sample_rate)


def write_ready(stream: BinaryIO, metadata: Header) -> None:
    """Advertise readiness and immutable runtime details."""
    write_frame(stream, {"status": "ready", **metadata})


def write_audio(
    stream: BinaryIO,
    samples: NDArray[np.float32],
    sample_rate: int,
) -> None:
    """Write normalized output audio as little-endian float32."""
    contiguous = np.ascontiguousarray(samples, dtype="<f4")
    write_frame(
        stream,
        {
            "status": "ok",
            "sample_rate": sample_rate,
            "sample_count": int(contiguous.size),
        },
        contiguous.tobytes(),
    )


def write_error(stream: BinaryIO, error: Exception) -> None:
    """Return a stable error response without serializing implementation state."""
    write_frame(
        stream,
        {
            "status": "error",
            "error": str(error),
            "error_kind": type(error).__name__,
        },
    )
