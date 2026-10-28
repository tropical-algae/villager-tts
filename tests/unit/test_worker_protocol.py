"""RVC worker protocol and server tests."""

import io
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray
from villager_rvc_protocol import Frame, ProtocolError, read_frame, write_frame

from villager_rvc_worker.engine import ConvertedAudio
from villager_rvc_worker.protocol import (
    ConvertRequest,
    ShutdownRequest,
    decode_request,
)
from villager_rvc_worker.server import WorkerServer


def test_frame_round_trip_and_typed_request_decoding() -> None:
    samples = np.asarray([-0.5, 0.5], dtype="<f4")
    stream = io.BytesIO()
    write_frame(
        stream,
        {"type": "convert", "sample_rate": 24_000, "sample_count": samples.size},
        samples.tobytes(),
    )
    stream.seek(0)

    request = decode_request(read_frame(stream))

    assert isinstance(request, ConvertRequest)
    assert request.sample_rate == 24_000
    assert np.array_equal(request.samples, samples)


def test_request_decoder_rejects_ambiguous_or_invalid_values() -> None:
    with pytest.raises(ProtocolError, match="sample rate"):
        decode_request(
            Frame(
                {"type": "convert", "sample_rate": True, "sample_count": 1},
                np.zeros(1, dtype="<f4").tobytes(),
            )
        )
    with pytest.raises(ProtocolError, match="non-finite"):
        decode_request(
            Frame(
                {"type": "convert", "sample_rate": 24_000, "sample_count": 1},
                np.asarray([np.nan], dtype="<f4").tobytes(),
            )
        )
    with pytest.raises(ProtocolError, match="shutdown"):
        decode_request(Frame({"type": "shutdown"}, b"unexpected"))
    assert isinstance(decode_request(Frame({"type": "shutdown"}, b"")), ShutdownRequest)


class _FakeEngine:
    runtime_metadata: ClassVar[Mapping[str, object]] = {
        "device": "cpu",
        "torch_version": "2.1.2+cpu",
        "cuda_version": None,
    }

    def convert(
        self,
        samples: NDArray[np.float32],
        sample_rate: int,
    ) -> ConvertedAudio:
        return ConvertedAudio(samples=samples + 0.25, sample_rate=sample_rate * 2)


def test_server_reports_request_errors_and_continues_until_shutdown() -> None:
    input_stream = io.BytesIO()
    write_frame(input_stream, {"type": "unknown"})
    samples = np.asarray([0.0, 0.25], dtype="<f4")
    write_frame(
        input_stream,
        {"type": "convert", "sample_rate": 24_000, "sample_count": samples.size},
        samples.tobytes(),
    )
    write_frame(input_stream, {"type": "shutdown"})
    input_stream.seek(0)
    output_stream = io.BytesIO()
    error_stream = io.StringIO()

    WorkerServer(_FakeEngine(), input_stream, output_stream, error_stream).serve_forever()

    output_stream.seek(0)
    ready = read_frame(output_stream)
    error = read_frame(output_stream)
    converted = read_frame(output_stream)
    assert ready.header["status"] == "ready"
    assert ready.header["torch_version"] == "2.1.2+cpu"
    assert error.header["status"] == "error"
    assert error.header["error_kind"] == "ProtocolError"
    assert converted.header["sample_rate"] == 48_000
    assert np.frombuffer(converted.payload, dtype="<f4").tolist() == pytest.approx([0.25, 0.5])
    assert "unsupported RVC worker request" in error_stream.getvalue()


def test_server_stops_after_an_unrecoverable_framing_error() -> None:
    output_stream = io.BytesIO()
    error_stream = io.StringIO()

    WorkerServer(_FakeEngine(), io.BytesIO(b"\x00"), output_stream, error_stream).serve_forever()

    output_stream.seek(0)
    assert read_frame(output_stream).header["status"] == "ready"
    assert read_frame(output_stream).header["status"] == "error"
    with pytest.raises(EOFError):
        read_frame(output_stream)
