"""Canonical in-process audio representation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """An immutable, ordered block of mono floating-point PCM."""

    samples: NDArray[np.float32]
    sample_rate: int
    sequence: int
    is_final: bool = False

    def __post_init__(self) -> None:
        """Validate and detach the chunk from mutable backend buffers."""
        if self.sample_rate <= 0:
            msg = "sample_rate must be positive"
            raise ValueError(msg)
        if self.sequence < 0:
            msg = "sequence must be non-negative"
            raise ValueError(msg)
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1:
            msg = "audio samples must be mono"
            raise ValueError(msg)
        if not np.all(np.isfinite(samples)):
            msg = "audio samples must be finite"
            raise ValueError(msg)
        detached = np.ascontiguousarray(np.clip(samples, -1.0, 1.0)).copy()
        detached.flags.writeable = False
        object.__setattr__(self, "samples", detached)

    @property
    def duration_seconds(self) -> float:
        """Return the chunk duration in seconds."""
        return float(self.samples.size) / self.sample_rate
