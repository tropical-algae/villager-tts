"""Text-to-speech backend contract."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from villager_tts.core.audio.models import AudioChunk
from villager_tts.core.runtime.cancellation import CancellationToken

VoiceGender = Literal["male", "female"]
DEFAULT_VOICE_GENDER: Final[VoiceGender] = "female"


@dataclass(frozen=True, slots=True)
class TtsRequest:
    """Normalized input sent to a TTS backend."""

    text: str
    speed: float
    gender: VoiceGender = DEFAULT_VOICE_GENDER


class TtsEngine(Protocol):
    """Backend capable of yielding one phrase as PCM chunks."""

    @property
    def name(self) -> str:
        """Return a stable backend identifier."""
        ...

    def synthesize(
        self,
        request: TtsRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[AudioChunk]:
        """Generate audio for one complete phrase."""
        ...
