"""Bounded length-framed binary transport shared across both environments."""

import json
import struct
from dataclasses import dataclass
from typing import IO, TypeAlias

MAX_HEADER_BYTES = 64 * 1024
MAX_AUDIO_BYTES = 256 * 1024 * 1024
_HEADER_PREFIX_BYTES = struct.calcsize(">I")

Header: TypeAlias = dict[str, object]


class ProtocolError(ValueError):
    """A malformed, truncated, or oversized transport frame."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One fully buffered transport frame."""

    header: Header
    payload: bytes


def read_frame(stream: IO[bytes]) -> Frame:
    """Read one bounded frame, distinguishing clean EOF from truncation."""
    prefix = stream.read(_HEADER_PREFIX_BYTES)
    if not prefix:
        raise EOFError
    if len(prefix) < _HEADER_PREFIX_BYTES:
        prefix += _read_exact(stream, _HEADER_PREFIX_BYTES - len(prefix))
    header_size = struct.unpack(">I", prefix)[0]
    if header_size > MAX_HEADER_BYTES:
        raise ProtocolError("protocol header exceeds safety limit")
    raw_header = _read_exact(stream, header_size)
    try:
        header = json.loads(raw_header)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProtocolError("protocol header is not valid JSON") from error
    if not isinstance(header, dict):
        raise ProtocolError("protocol header must be an object")
    payload_size = struct.unpack(">Q", _read_exact(stream, 8))[0]
    if payload_size > MAX_AUDIO_BYTES:
        raise ProtocolError("protocol payload exceeds safety limit")
    return Frame(header=header, payload=_read_exact(stream, payload_size))


def write_frame(stream: IO[bytes], header: Header, payload: bytes = b"") -> None:
    """Write and flush one bounded frame."""
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(raw_header) > MAX_HEADER_BYTES or len(payload) > MAX_AUDIO_BYTES:
        raise ProtocolError("protocol frame exceeds safety limit")
    stream.write(struct.pack(">I", len(raw_header)))
    stream.write(raw_header)
    stream.write(struct.pack(">Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream: IO[bytes], size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise ProtocolError("truncated protocol frame")
        chunks.extend(chunk)
    return bytes(chunks)
