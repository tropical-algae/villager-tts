"""Per-application Prometheus instrumentation."""

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


@dataclass(frozen=True, slots=True)
class Metrics:
    """Metrics collectors isolated for each app/test instance."""

    registry: CollectorRegistry
    requests: Counter
    failures: Counter
    inflight: Gauge
    synthesis_seconds: Histogram
    first_audio_seconds: Histogram

    @classmethod
    def create(cls) -> "Metrics":
        """Create collectors in a fresh registry."""
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests=Counter(
                "villager_tts_requests_total",
                "Speech synthesis requests",
                ["endpoint"],
                registry=registry,
            ),
            failures=Counter(
                "villager_tts_failures_total",
                "Speech synthesis failures",
                ["endpoint", "kind"],
                registry=registry,
            ),
            inflight=Gauge(
                "villager_tts_inflight_requests",
                "Speech synthesis requests currently open",
                ["endpoint"],
                registry=registry,
            ),
            synthesis_seconds=Histogram(
                "villager_tts_synthesis_seconds",
                "End-to-end synthesis duration",
                ["endpoint"],
                registry=registry,
            ),
            first_audio_seconds=Histogram(
                "villager_tts_first_audio_seconds",
                "Time until the first converted audio block",
                registry=registry,
            ),
        )
