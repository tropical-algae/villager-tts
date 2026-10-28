"""Voice-conversion backend contract."""

from collections.abc import AsyncIterator
from typing import Protocol

from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken


class VoiceConverter(Protocol):
    """Backend that transforms a source PCM stream into the target voice."""

    @property
    def name(self) -> str:
        """Return a stable backend identifier."""
        ...

    def convert(
        self,
        chunks: AsyncIterator[AudioChunk],
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Convert source audio while preserving stream order."""
        ...
