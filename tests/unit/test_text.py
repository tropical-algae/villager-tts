"""Text normalization and segmentation tests."""

import pytest

from villager_tts.core.text.segmenter import normalize_text, segment_text


def test_normalize_text_preserves_meaningful_line_boundaries() -> None:
    assert normalize_text("  你好\t 世界  \n\n hello   world ") == "你好 世界\nhello world"


def test_segment_text_merges_short_sentences_and_splits_long_ones() -> None:
    assert segment_text("一。二。三三三三三三三三三三", target_chars=4, max_chars=6) == [
        "一。二。",
        "三三三三三三",
        "三三三三",
    ]
    assert segment_text("alpha,beta,gamma", target_chars=5, max_chars=7) == [
        "alpha,",
        "beta,",
        "gamma",
    ]


def test_segment_text_handles_empty_and_invalid_limits() -> None:
    assert segment_text(" \n ") == []
    with pytest.raises(ValueError, match="limits"):
        segment_text("text", target_chars=10, max_chars=2)
