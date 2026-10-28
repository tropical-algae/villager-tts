"""PCM and WAV encoders."""

import io
import subprocess
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from villager_tts.core.audio.models import AudioChunk

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
_FFMPEG_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class EncodedAudio:
    """Encoded audio bytes and their HTTP representation metadata."""

    content: bytes
    media_type: str
    extension: str


class AudioEncodingError(RuntimeError):
    """Raised when the external audio encoder cannot produce valid output."""


def float32_to_pcm16(samples: NDArray[np.float32], *, byteorder: str) -> bytes:
    """Encode normalized floats as signed PCM16 with explicit byte order."""
    if byteorder not in {"little", "big"}:
        msg = "byteorder must be 'little' or 'big'"
        raise ValueError(msg)
    normalized = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = np.clip(normalized, -1.0, 1.0)
    scaled = np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0)
    dtype = np.dtype("<i2" if byteorder == "little" else ">i2")
    return cast("bytes", np.rint(scaled).astype(dtype, copy=False).tobytes())


def chunk_to_l16(chunk: AudioChunk) -> bytes:
    """Encode one chunk as RFC 2586 network-byte-order L16."""
    return float32_to_pcm16(chunk.samples, byteorder="big")


def chunks_to_wav(chunks: Iterable[AudioChunk], *, sample_rate: int) -> bytes:
    """Encode same-rate mono chunks into a PCM16 WAV file."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for chunk in chunks:
            if chunk.sample_rate != sample_rate:
                msg = "all WAV chunks must use the requested sample rate"
                raise ValueError(msg)
            wav_file.writeframesraw(float32_to_pcm16(chunk.samples, byteorder="little"))
    return output.getvalue()


def encode_audio(
    chunks: Iterable[AudioChunk],
    *,
    sample_rate: int,
    response_format: AudioFormat,
) -> EncodedAudio:
    """Encode mono PCM chunks into an OpenAI speech response format."""
    materialized = tuple(chunks)
    if response_format == "wav":
        return EncodedAudio(
            chunks_to_wav(materialized, sample_rate=sample_rate), "audio/wav", "wav"
        )

    pcm = chunks_to_pcm16(materialized, sample_rate=sample_rate)
    if response_format == "pcm":
        return EncodedAudio(pcm, "audio/pcm", "pcm")

    output_arguments, media_type = {
        "mp3": (("-codec:a", "libmp3lame", "-f", "mp3"), "audio/mpeg"),
        "opus": (("-codec:a", "libopus", "-f", "ogg"), "audio/ogg"),
        "aac": (("-codec:a", "aac", "-f", "adts"), "audio/aac"),
        "flac": (("-codec:a", "flac", "-f", "flac"), "audio/flac"),
    }[response_format]
    command = (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-vn",
        *output_arguments,
        "pipe:1",
    )
    try:
        completed = subprocess.run(  # noqa: S603 - static ffmpeg command and validated format
            command,
            input=pcm,
            capture_output=True,
            check=False,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise AudioEncodingError("ffmpeg executable is unavailable") from error
    except subprocess.TimeoutExpired as error:
        raise AudioEncodingError("ffmpeg exceeded its encoding deadline") from error
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        msg = f"ffmpeg failed to encode {response_format}: {detail or 'empty output'}"
        raise AudioEncodingError(msg)
    return EncodedAudio(completed.stdout, media_type, response_format)


def chunks_to_pcm16(chunks: Iterable[AudioChunk], *, sample_rate: int) -> bytes:
    """Encode same-rate mono chunks as little-endian signed PCM16."""
    encoded: list[bytes] = []
    for chunk in chunks:
        if chunk.sample_rate != sample_rate:
            msg = "all PCM chunks must use the requested sample rate"
            raise ValueError(msg)
        encoded.append(float32_to_pcm16(chunk.samples, byteorder="little"))
    return b"".join(encoded)
