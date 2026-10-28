"""Environment-backed service configuration."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Device(StrEnum):
    """Supported inference devices."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class Settings(BaseSettings):
    """Validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VILLAGER_TTS_",
        extra="ignore",
    )

    log_level: str = "INFO"
    api_key: SecretStr | None = None
    port: int = Field(default=8_000, ge=1, le=65_535)

    rvc_device: Device = Device.AUTO
    max_text_characters: int = Field(default=1_000, ge=1, le=10_000)
    max_concurrent_syntheses: int = Field(default=1, ge=1, le=64)
    capacity_wait_seconds: float = Field(default=2.0, gt=0, le=60)
    output_sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)
    phrase_target_characters: int = Field(default=80, ge=1, le=1_000)
    phrase_max_characters: int = Field(default=200, ge=1, le=2_000)

    tts_backend: str = "sherpa_onnx"
    tts_male_model_path: Path = Path("models/tts/male/vits-zh-hf-fanchen-wnj.onnx")
    tts_male_tokens_path: Path = Path("models/tts/male/tokens.txt")
    tts_male_lexicon_path: Path | None = Path("models/tts/male/lexicon.txt")
    tts_male_data_dir: Path | None = None
    tts_male_rule_fsts: list[Path] = Field(
        default_factory=lambda: [
            Path("models/tts/male/date.fst"),
            Path("models/tts/male/number.fst"),
        ]
    )
    tts_female_model_path: Path = Path("models/tts/female/model.onnx")
    tts_female_tokens_path: Path = Path("models/tts/female/tokens.txt")
    tts_female_lexicon_path: Path | None = Path("models/tts/female/lexicon.txt")
    tts_female_data_dir: Path | None = None
    tts_female_rule_fsts: list[Path] = Field(
        default_factory=lambda: [
            Path("models/tts/female/date.fst"),
            Path("models/tts/female/number.fst"),
        ]
    )
    tts_num_threads: int = Field(default=2, ge=1, le=64)
    tts_instance_count: int = Field(default=1, ge=1, le=16)
    tts_warmup_text: str | None = "你好"
    tts_request_timeout_seconds: float = Field(default=120, gt=0, le=600)

    vc_backend: str = "rvc_worker"
    rvc_model_path: Path = Path("models/villager/villagerminecraft.pth")
    rvc_index_path: Path = Path(
        "models/villager/added_IVF99_Flat_nprobe_1_villagerminecraft_v2.index"
    )
    rvc_speaker_id: int = Field(default=0, ge=0)
    rvc_pitch: int = Field(default=0, ge=-24, le=24)
    rvc_f0_method: Literal["rmvpe", "pm"] = "rmvpe"
    rvc_index_rate: float = Field(default=0.75, ge=0, le=1)
    rvc_filter_radius: int = Field(default=3, ge=0, le=7)
    rvc_rms_mix_rate: float = Field(default=1.0, ge=0, le=1)
    rvc_protect: float = Field(default=0.33, ge=0, le=0.5)
    rvc_worker_command: str = "workers/rvc/.venv/bin/python -m villager_rvc_worker"
    rvc_worker_count: int = Field(default=1, ge=1, le=16)
    rvc_startup_timeout_seconds: float = Field(default=180, gt=0, le=600)
    rvc_request_timeout_seconds: float = Field(default=120, gt=0, le=600)
    rvc_hubert_path: Path = Path("models/rvc/hubert_base.pt")
    rvc_rmvpe_path: Path = Path("models/rvc/rmvpe.pt")
    rvc_crossfade_milliseconds: int = Field(default=20, ge=0, le=200)
    output_chunk_milliseconds: int = Field(default=200, ge=20, le=1_000)

    model_manifest_path: Path = Path("models/manifest.yaml")
    model_artifact_root: Path | None = None
    verify_model_artifacts: bool = True

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate the configured log level."""
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            msg = f"unsupported log level: {value}"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_phrase_limits(self) -> "Settings":
        """Validate settings whose bounds depend on another setting."""
        if self.phrase_max_characters < self.phrase_target_characters:
            msg = "phrase_max_characters must be at least phrase_target_characters"
            raise ValueError(msg)
        if self.rvc_worker_count > self.max_concurrent_syntheses:
            msg = "rvc_worker_count must not exceed max_concurrent_syntheses"
            raise ValueError(msg)
        if self.tts_instance_count > self.max_concurrent_syntheses:
            msg = "tts_instance_count must not exceed max_concurrent_syntheses"
            raise ValueError(msg)
        return self
