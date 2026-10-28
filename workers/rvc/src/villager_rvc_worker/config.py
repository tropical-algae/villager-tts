"""Typed command-line configuration for the RVC worker."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

WorkerDevice = Literal["auto", "cpu", "cuda:0"]
F0Method = Literal["rmvpe", "pm"]
_MINIMUM_PITCH = -24
_MAXIMUM_PITCH = 24


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Validated model locations and inference settings for one worker."""

    model: Path
    index: Path
    hubert: Path
    rmvpe: Path
    device: WorkerDevice
    speaker_id: int = 0
    pitch: int = 0
    f0_method: F0Method = "rmvpe"
    index_rate: float = 0.75
    filter_radius: int = 3
    rms_mix_rate: float = 1.0
    protect: float = 0.33

    def validate(self) -> None:
        """Reject invalid values before importing the heavyweight RVC stack."""
        paths = [
            ("model", self.model),
            ("index", self.index),
            ("hubert", self.hubert),
        ]
        if self.f0_method == "rmvpe":
            paths.append(("rmvpe", self.rmvpe))
        for label, path in paths:
            if not path.is_file():
                msg = f"{label} file is missing: {path}"
                raise FileNotFoundError(msg)
        if self.speaker_id < 0:
            msg = "speaker ID must be non-negative"
            raise ValueError(msg)
        if not _MINIMUM_PITCH <= self.pitch <= _MAXIMUM_PITCH:
            msg = "pitch must be between -24 and 24"
            raise ValueError(msg)
        _require_range("index rate", self.index_rate, minimum=0.0, maximum=1.0)
        _require_range("filter radius", self.filter_radius, minimum=0, maximum=7)
        _require_range("RMS mix rate", self.rms_mix_rate, minimum=0.0, maximum=1.0)
        _require_range("protect", self.protect, minimum=0.0, maximum=0.5)


def parse_config(argv: Sequence[str] | None = None) -> WorkerConfig:
    """Parse process arguments into a strongly typed configuration object."""
    parser = argparse.ArgumentParser(description="Persistent Villager TTS RVC worker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--hubert", type=Path, required=True)
    parser.add_argument("--rmvpe", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), required=True)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0-method", choices=("rmvpe", "pm"), default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--rms-mix-rate", type=float, default=1.0)
    parser.add_argument("--protect", type=float, default=0.33)
    arguments = parser.parse_args(argv)
    return WorkerConfig(
        model=arguments.model,
        index=arguments.index,
        hubert=arguments.hubert,
        rmvpe=arguments.rmvpe,
        device=cast("WorkerDevice", arguments.device),
        speaker_id=arguments.speaker_id,
        pitch=arguments.pitch,
        f0_method=cast("F0Method", arguments.f0_method),
        index_rate=arguments.index_rate,
        filter_radius=arguments.filter_radius,
        rms_mix_rate=arguments.rms_mix_rate,
        protect=arguments.protect,
    )


def _require_range(
    label: str,
    value: float,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if not minimum <= value <= maximum:
        msg = f"{label} must be between {minimum} and {maximum}"
        raise ValueError(msg)
