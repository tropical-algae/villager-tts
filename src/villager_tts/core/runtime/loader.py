"""Construct configured inference engines."""

from pathlib import Path
from typing import TYPE_CHECKING

from villager_tts.config import Settings
from villager_tts.core.runtime.artifacts import verify_manifest
from villager_tts.core.runtime.state import EngineBundle
from villager_tts.core.tts.pool import TtsEnginePool
from villager_tts.core.tts.selector import GenderedTtsEngine
from villager_tts.core.tts.sherpa import SherpaOnnxTtsEngine
from villager_tts.core.tts.tone import ToneTtsEngine
from villager_tts.core.vc.passthrough import PassthroughVoiceConverter
from villager_tts.core.vc.worker import RvcWorkerVoiceConverter

if TYPE_CHECKING:
    from villager_tts.core.tts.protocol import TtsEngine
    from villager_tts.core.vc.protocol import VoiceConverter


def load_engine_bundle(settings: Settings) -> EngineBundle:
    """Verify artifacts, then load the selected backends exactly once."""
    if settings.verify_model_artifacts:
        verify_manifest(
            settings.model_manifest_path,
            artifact_root=settings.model_artifact_root,
        )

    if settings.tts_backend == "tone":
        tts: TtsEngine = ToneTtsEngine()
    elif settings.tts_backend == "sherpa_onnx":
        male = TtsEnginePool(
            [
                _load_sherpa_voice(
                    settings,
                    model=settings.tts_male_model_path,
                    tokens=settings.tts_male_tokens_path,
                    lexicon=settings.tts_male_lexicon_path,
                    data_dir=settings.tts_male_data_dir,
                    rule_fsts=settings.tts_male_rule_fsts,
                )
                for _ in range(settings.tts_instance_count)
            ]
        )
        female = TtsEnginePool(
            [
                _load_sherpa_voice(
                    settings,
                    model=settings.tts_female_model_path,
                    tokens=settings.tts_female_tokens_path,
                    lexicon=settings.tts_female_lexicon_path,
                    data_dir=settings.tts_female_data_dir,
                    rule_fsts=settings.tts_female_rule_fsts,
                )
                for _ in range(settings.tts_instance_count)
            ]
        )
        tts = GenderedTtsEngine(male=male, female=female)
    else:
        msg = f"unsupported TTS backend: {settings.tts_backend}"
        raise ValueError(msg)

    if settings.vc_backend == "passthrough":
        converter: VoiceConverter = PassthroughVoiceConverter(settings.output_sample_rate)
    elif settings.vc_backend == "rvc_worker":
        converter = RvcWorkerVoiceConverter.from_settings(settings)
    else:
        msg = f"unsupported voice-conversion backend: {settings.vc_backend}"
        raise ValueError(msg)

    return EngineBundle(tts=tts, voice_converter=converter)


def _load_sherpa_voice(
    settings: Settings,
    *,
    model: Path,
    tokens: Path,
    lexicon: Path | None,
    data_dir: Path | None,
    rule_fsts: list[Path],
) -> SherpaOnnxTtsEngine:
    """Load one fixed single-speaker sherpa model."""
    return SherpaOnnxTtsEngine(
        model=model,
        tokens=tokens,
        lexicon=lexicon,
        data_dir=data_dir,
        rule_fsts=rule_fsts,
        speaker_id=0,
        num_threads=settings.tts_num_threads,
        provider="cpu",
        warmup_text=settings.tts_warmup_text,
    )
