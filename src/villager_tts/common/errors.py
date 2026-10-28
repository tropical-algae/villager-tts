"""Domain errors exposed through RFC 9457 problem responses."""

from dataclasses import dataclass


@dataclass(slots=True)
class ServiceError(Exception):
    """A safe, structured error that may be returned to a client."""

    status_code: int
    title: str
    detail: str
    error_type: str


class ServiceUnavailableError(ServiceError):
    """Raised when inference cannot currently be accepted."""

    def __init__(self, detail: str = "The speech model is not ready.") -> None:
        super().__init__(
            status_code=503,
            title="Service unavailable",
            detail=detail,
            error_type="https://villager-tts.dev/problems/service-unavailable",
        )


class CapacityExceededError(ServiceError):
    """Raised when the bounded inference queue is full."""

    def __init__(self) -> None:
        super().__init__(
            status_code=429,
            title="Too many requests",
            detail="All synthesis slots are busy. Retry later.",
            error_type="https://villager-tts.dev/problems/capacity-exceeded",
        )


class InputTooLongError(ServiceError):
    """Raised when text exceeds the configured inference limit."""

    def __init__(self, maximum_characters: int) -> None:
        super().__init__(
            status_code=413,
            title="Input too long",
            detail=f"input must contain at most {maximum_characters} characters.",
            error_type="https://villager-tts.dev/problems/input-too-long",
        )


class SynthesisError(ServiceError):
    """Raised when inference fails before any response audio is sent."""

    def __init__(self, detail: str = "Speech synthesis failed.") -> None:
        super().__init__(
            status_code=500,
            title="Synthesis failed",
            detail=detail,
            error_type="https://villager-tts.dev/problems/synthesis-failed",
        )


class InferenceTimeoutError(SynthesisError):
    """Raised when a bounded inference stage misses its deadline."""

    def __init__(self, detail: str = "Speech inference exceeded its deadline.") -> None:
        ServiceError.__init__(
            self,
            status_code=504,
            title="Inference timed out",
            detail=detail,
            error_type="https://villager-tts.dev/problems/inference-timeout",
        )
