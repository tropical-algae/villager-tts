"""Conservative phrase segmentation for low-latency synthesis."""

import re

_WHITESPACE = re.compile(r"[\t\r\f\v ]+")
_PRIMARY_BOUNDARY = re.compile(r"(?<=[。！？!?；;\n])")
_SECONDARY_BOUNDARY = re.compile(r"(?<=[，,、：:])")


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing language-specific symbols."""
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def segment_text(text: str, *, target_chars: int = 80, max_chars: int = 200) -> list[str]:
    """Split text at natural punctuation and cap pathological long phrases."""
    if target_chars < 1 or max_chars < target_chars:
        msg = "segment limits are invalid"
        raise ValueError(msg)
    normalized = normalize_text(text)
    if not normalized:
        return []

    primary = [part.strip() for part in _PRIMARY_BOUNDARY.split(normalized) if part.strip()]
    result: list[str] = []
    pending = ""
    for phrase in primary:
        if len(pending) + len(phrase) <= target_chars:
            pending += phrase
            continue
        if pending:
            result.append(pending)
            pending = ""
        if len(phrase) <= max_chars:
            pending = phrase
            continue
        result.extend(_split_long_phrase(phrase, max_chars=max_chars))
    if pending:
        result.append(pending)
    return result


def _split_long_phrase(phrase: str, *, max_chars: int) -> list[str]:
    secondary = [part for part in _SECONDARY_BOUNDARY.split(phrase) if part]
    result: list[str] = []
    pending = ""
    for part in secondary:
        remainder = part
        if len(pending) + len(remainder) <= max_chars:
            pending += remainder
            continue
        if pending:
            result.append(pending)
            pending = ""
        while len(remainder) > max_chars:
            result.append(remainder[:max_chars])
            remainder = remainder[max_chars:]
        pending = remainder
    if pending:
        result.append(pending)
    return result
