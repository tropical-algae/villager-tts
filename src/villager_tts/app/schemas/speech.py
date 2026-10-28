"""Speech API schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from villager_tts.core.tts.protocol import DEFAULT_VOICE_GENDER, VoiceGender

SpeechResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeechRequestBase(BaseModel):
    """Fields shared by the file and streaming speech endpoints."""

    model_config = ConfigDict(extra="forbid")

    input: str = Field(
        min_length=1,
        description="要合成的文本；空白文本会被拒绝，最大长度由服务配置决定。",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="TTS 语速倍率，1.0 为模型原始语速。",
    )
    gender: VoiceGender = Field(
        default=DEFAULT_VOICE_GENDER,
        description="原始 TTS 声音。male 为中文男声；female 为中英双语女声。",
    )
    enable_villager: bool = Field(
        default=True,
        description="是否通过 RVC 转换为村民音色；关闭时返回原始 TTS。",
    )

    @field_validator("input")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject input that only contains whitespace."""
        if not value.strip():
            msg = "input must contain visible text"
            raise ValueError(msg)
        return value


class SpeechRequest(SpeechRequestBase):
    """Request for a complete WAV response."""


class StreamSpeechRequest(SpeechRequestBase):
    """Public streaming speech synthesis request."""


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        min_length=1,
        description="兼容 OpenAI 协议的模型标识；当前服务始终使用已部署的固定模型。",
    )
    input: str = Field(
        min_length=1,
        description="要合成的文本；最大长度由服务配置决定。",
    )
    voice: VoiceGender = Field(
        description="源 TTS 声音；female 为中英双语女声，male 为中文男声。",
    )
    instructions: str | None = Field(
        default=None,
        description="兼容 OpenAI 协议的风格指令；当前固定 TTS 后端不应用此字段。",
    )
    response_format: SpeechResponseFormat = Field(
        default="mp3",
        description="输出格式，可选 mp3、opus、aac、flac、wav 或 pcm。",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="TTS 语速倍率，1.0 为模型原始语速。",
    )
    stream_format: Literal["audio"] = Field(
        default="audio",
        description="OpenAI 音频流格式；当前支持 audio。",
    )

    @field_validator("input")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject input that only contains whitespace."""
        if not value.strip():
            msg = "input must contain visible text"
            raise ValueError(msg)
        return value


class HealthResponse(BaseModel):
    """Health state returned by liveness and readiness endpoints."""

    status: str
    tts_backend: str | None = None
    vc_backend: str | None = None
    detail: str | None = None
