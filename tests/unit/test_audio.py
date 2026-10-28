"""Audio domain and codec tests."""

import io
import subprocess
import wave

import numpy as np
import pytest

from villager_tts.core.audio.encoding import (
    AudioEncodingError,
    AudioFormat,
    chunk_to_l16,
    chunks_to_pcm16,
    chunks_to_wav,
    encode_audio,
    float32_to_pcm16,
)
from villager_tts.core.audio.models import AudioChunk


def test_audio_chunk_detaches_clamps_and_freezes_samples() -> None:
    source = np.array([-2.0, 0.5, 2.0], dtype=np.float32)
    chunk = AudioChunk(source, 48_000, 0, is_final=True)
    source[:] = 0

    assert chunk.samples.tolist() == [-1.0, 0.5, 1.0]
    assert chunk.samples.dtype == np.float32
    assert not chunk.samples.flags.writeable
    assert chunk.duration_seconds == pytest.approx(3 / 48_000)


@pytest.mark.parametrize(
    ("sample_rate", "sequence", "message"),
    [
        (0, 0, "sample_rate"),
        (48_000, -1, "sequence"),
    ],
)
def test_audio_chunk_rejects_invalid_metadata(
    sample_rate: int,
    sequence: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AudioChunk(np.array([0], dtype=np.float32), sample_rate, sequence)


def test_audio_chunk_rejects_multichannel_and_nonfinite_audio() -> None:
    with pytest.raises(ValueError, match="mono"):
        AudioChunk(np.zeros((2, 2), dtype=np.float32), 48_000, 0)
    with pytest.raises(ValueError, match="finite"):
        AudioChunk(np.array([np.nan], dtype=np.float32), 48_000, 0)


def test_pcm16_conversion_uses_asymmetric_full_scale_and_byte_order() -> None:
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, np.nan], dtype=np.float32)
    assert np.frombuffer(float32_to_pcm16(samples, byteorder="little"), dtype="<i2").tolist() == [
        -32768,
        -16384,
        0,
        16384,
        32767,
        0,
    ]
    chunk = AudioChunk(np.array([1.0, -1.0], dtype=np.float32), 48_000, 0)
    assert chunk_to_l16(chunk) == b"\x7f\xff\x80\x00"
    with pytest.raises(ValueError, match="byteorder"):
        float32_to_pcm16(samples, byteorder="middle")


def test_wav_encoder_writes_standard_mono_header_and_samples() -> None:
    chunks = [
        AudioChunk(np.array([0.0, 1.0], dtype=np.float32), 48_000, 0),
        AudioChunk(np.array([-1.0], dtype=np.float32), 48_000, 1, is_final=True),
    ]
    encoded = chunks_to_wav(chunks, sample_rate=48_000)

    with wave.open(io.BytesIO(encoded), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 48_000
        assert wav_file.getnframes() == 3
        assert np.frombuffer(wav_file.readframes(3), dtype="<i2").tolist() == [0, 32767, -32768]

    with pytest.raises(ValueError, match="sample rate"):
        chunks_to_wav([AudioChunk(np.zeros(1, dtype=np.float32), 24_000, 0)], sample_rate=48_000)


def test_pcm_encoder_writes_little_endian_samples() -> None:
    chunks = [AudioChunk(np.array([1.0, -1.0], dtype=np.float32), 48_000, 0, is_final=True)]

    encoded = encode_audio(chunks, sample_rate=48_000, response_format="pcm")

    assert encoded.media_type == "audio/pcm"
    assert encoded.extension == "pcm"
    assert encoded.content == b"\xff\x7f\x00\x80"
    assert chunks_to_pcm16(chunks, sample_rate=48_000) == encoded.content


@pytest.mark.parametrize(
    ("response_format", "media_type", "muxer"),
    [
        ("mp3", "audio/mpeg", "mp3"),
        ("opus", "audio/ogg", "ogg"),
        ("aac", "audio/aac", "adts"),
        ("flac", "audio/flac", "flac"),
    ],
)
def test_compressed_encoder_invokes_ffmpeg_with_a_fixed_command(
    monkeypatch: pytest.MonkeyPatch,
    response_format: AudioFormat,
    media_type: str,
    muxer: str,
) -> None:
    captured: list[tuple[tuple[str, ...], bytes]] = []

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        input_bytes = kwargs["input"]
        assert isinstance(input_bytes, bytes)
        captured.append((command, input_bytes))
        return subprocess.CompletedProcess(command, 0, stdout=b"encoded", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    chunk = AudioChunk(np.array([0.5], dtype=np.float32), 48_000, 0, is_final=True)

    encoded = encode_audio([chunk], sample_rate=48_000, response_format=response_format)

    assert encoded.content == b"encoded"
    assert encoded.media_type == media_type
    assert encoded.extension == response_format
    assert captured[0][0][0] == "ffmpeg"
    assert captured[0][0][-3:] == ("-f", muxer, "pipe:1")
    assert captured[0][1]


def test_compressed_encoder_reports_ffmpeg_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"codec failed")

    monkeypatch.setattr(subprocess, "run", fail)
    chunk = AudioChunk(np.zeros(1, dtype=np.float32), 48_000, 0, is_final=True)

    with pytest.raises(AudioEncodingError, match="codec failed"):
        encode_audio([chunk], sample_rate=48_000, response_format="mp3")
