"""Exclusive leasing for preloaded TTS inference instances."""

import asyncio
from collections.abc import AsyncIterator, Sequence

from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.tts.protocol import TtsEngine, TtsRequest


class TtsEnginePool:
    """Share preloaded engines without calling one instance concurrently."""

    def __init__(self, engines: Sequence[TtsEngine]) -> None:
        if not engines:
            msg = "TTS engine pool requires at least one engine"
            raise ValueError(msg)
        self._available: asyncio.Queue[TtsEngine] = asyncio.Queue(maxsize=len(engines))
        self._engines = tuple(engines)
        for engine in engines:
            self._available.put_nowait(engine)

    @property
    def name(self) -> str:
        """Return a stable backend identifier for readiness reporting."""
        names = {engine.name for engine in self._engines}
        return names.pop() if len(names) == 1 else "tts-pool"

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Lease one instance for the complete phrase and always return it."""
        cancellation.raise_if_cancelled()
        engine = await self._available.get()
        try:
            async for chunk in engine.synthesize(request, cancellation):
                yield chunk
        finally:
            self._available.put_nowait(engine)
