"""Request-level selection between fixed single-speaker TTS engines."""

from collections.abc import AsyncIterator

from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.tts.protocol import TtsEngine, TtsRequest, VoiceGender


class GenderedTtsEngine:
    """Route each request to a preloaded male or female engine."""

    def __init__(self, *, male: TtsEngine, female: TtsEngine) -> None:
        self._engines: dict[VoiceGender, TtsEngine] = {
            "male": male,
            "female": female,
        }

    @property
    def name(self) -> str:
        """Return one stable identifier for readiness reporting."""
        names = {engine.name for engine in self._engines.values()}
        return names.pop() if len(names) == 1 else "gendered-tts"

    async def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Yield audio from the engine selected by the normalized request."""
        async for chunk in self._engines[request.gender].synthesize(request, cancellation):
            yield chunk
