"""Protocol-compatible subprocess used by integration tests."""

import json
import struct
import sys
from typing import BinaryIO


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if value is None or len(value) != size:
        raise EOFError
    return value


def _read_frame(stream: BinaryIO) -> tuple[dict[str, object], bytes]:
    header_size = struct.unpack(">I", _read_exact(stream, 4))[0]
    header = json.loads(_read_exact(stream, header_size))
    payload_size = struct.unpack(">Q", _read_exact(stream, 8))[0]
    return header, _read_exact(stream, payload_size)


def _write_frame(stream: BinaryIO, header: dict[str, object], payload: bytes) -> None:
    raw = json.dumps(header).encode()
    stream.write(struct.pack(">I", len(raw)) + raw + struct.pack(">Q", len(payload)) + payload)
    stream.flush()


def main() -> None:
    _write_frame(sys.stdout.buffer, {"status": "ready"}, b"")
    while True:
        header, payload = _read_frame(sys.stdin.buffer)
        if header["type"] == "shutdown":
            return
        _write_frame(
            sys.stdout.buffer,
            {
                "status": "ok",
                "sample_rate": 48_000,
                "sample_count": len(payload) // 4,
            },
            payload,
        )


if __name__ == "__main__":
    main()
