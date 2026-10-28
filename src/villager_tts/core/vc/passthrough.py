"""Streaming resampling converter used when voice conversion is disabled."""

from collections.abc import AsyncIterator

from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.audio.resampling import resample_stream
from villager_tts.core.runtime.cancellation import CancellationToken


class PassthroughVoiceConverter:
    """Resample source audio without changing its voice."""

    def __init__(self, output_sample_rate: int) -> None:
        self._output_sample_rate = output_sample_rate

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "passthrough-development"

    async def convert(
        self,
        chunks: AsyncIterator[AudioChunk],
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Resample incrementally while preserving source phrase boundaries."""
        async for chunk in resample_stream(chunks, output_rate=self._output_sample_rate):
            cancellation.raise_if_cancelled()
            yield chunk
