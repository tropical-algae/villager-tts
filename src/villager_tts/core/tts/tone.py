"""Deterministic development backend used for smoke tests."""

from collections.abc import AsyncIterator

import numpy as np

from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.tts.protocol import TtsRequest


class ToneTtsEngine:
    """Generate a short tone per character; never use for production audio."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self._sample_rate = sample_rate

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "tone-development"

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Yield deterministic PCM in bounded chunks."""
        cancellation.raise_if_cancelled()
        seconds = max(0.08, min(3.0, len(request.text) * 0.035 / request.speed))
        count = round(self._sample_rate * seconds)
        time_axis = np.arange(count, dtype=np.float32) / self._sample_rate
        frequency = 180.0 + float(sum(map(ord, request.text)) % 80)
        envelope = np.minimum(1.0, np.minimum(time_axis * 30, (seconds - time_axis) * 30))
        samples = (0.15 * envelope * np.sin(2 * np.pi * frequency * time_axis)).astype(np.float32)
        yield AudioChunk(samples=samples, sample_rate=self._sample_rate, sequence=0, is_final=True)
