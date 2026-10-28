"""Tests for the RVC subprocess audio boundary."""

import numpy as np
import pytest

from villager_rvc_worker.audio import pcm_to_float32


def test_signed_pcm_is_normalized_instead_of_clipped() -> None:
    source = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)

    result = pcm_to_float32(source)

    assert result.dtype == np.float32
    assert result.tolist() == pytest.approx([-1.0, -0.5, 0.0, 0.5, 32767 / 32768])
    assert np.count_nonzero(np.abs(result[1:4]) >= 0.999) == 0


def test_normalized_float_is_preserved_and_invalid_audio_is_rejected() -> None:
    source = np.array([-0.25, 0.0, 0.25], dtype=np.float64)

    assert pcm_to_float32(source).tolist() == pytest.approx(source.tolist())

    with pytest.raises(ValueError, match="non-normalized"):
        pcm_to_float32(np.array([2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        pcm_to_float32(np.array([np.nan], dtype=np.float32))
    with pytest.raises(TypeError, match="unsupported"):
        pcm_to_float32(np.array([True], dtype=np.bool_))
