"""RVC model loading and phrase inference."""

import importlib
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from villager_rvc_worker.audio import pcm_to_float32
from villager_rvc_worker.config import WorkerConfig

EXPECTED_OUTPUT_SAMPLE_RATE = 48_000
_STEREO_DIMENSIONS = 2
_WARMUP_SAMPLE_RATE = 44_100
_WARMUP_SECONDS = 0.5
_WARMUP_FREQUENCY_HZ = 180


class _UpstreamConfig(Protocol):
    device: object
    is_half: bool


class _UpstreamPipeline(Protocol):
    model_rmvpe: object


class _UpstreamConverter(Protocol):
    config: _UpstreamConfig
    version: object
    tgt_sr: object
    if_f0: object
    n_spk: object
    pipeline: _UpstreamPipeline | None
    hubert_model: object

    def get_vc(self, model_path: str) -> object:
        """Load a voice checkpoint."""
        ...

    def vc_inference(
        self,
        speaker_id: int,
        audio_path: str,
        *,
        f0_up_key: int,
        f0_method: str,
        index_file: str,
        index_rate: float,
        filter_radius: int,
        resample_sr: int,
        rms_mix_rate: float,
        protect: float,
        hubert_path: str,
    ) -> tuple[int | None, ArrayLike | None, object, str | None]:
        """Run one upstream RVC inference."""
        ...


class _VcModule(Protocol):
    load_audio: Callable[[str, int], NDArray[np.float32]]
    VC: Callable[[], _UpstreamConverter]

    def load_hubert(self, config: _UpstreamConfig, path: str) -> object:
        """Load the ContentVec/HuBERT encoder."""
        ...


class _RmvpeFactory(Protocol):
    def __call__(self, path: str, *, is_half: bool, device: object) -> object:
        """Build an RMVPE model."""
        ...


class _RmvpeModule(Protocol):
    RMVPE: _RmvpeFactory


class _TorchVersion(Protocol):
    cuda: str | None


class _TorchCuda(Protocol):
    def is_available(self) -> bool:
        """Return whether CUDA can be used."""
        ...

    def current_device(self) -> int:
        """Return the current CUDA device index."""
        ...

    def get_device_name(self, device: int) -> str:
        """Return a display name for one CUDA device."""
        ...


class _TorchModule(Protocol):
    __version__: str
    version: _TorchVersion
    cuda: _TorchCuda


@dataclass(frozen=True, slots=True)
class ConvertedAudio:
    """One phrase converted to normalized mono float32."""

    samples: NDArray[np.float32]
    sample_rate: int


class RvcInferenceEngine:
    """Loaded RVC v2 pipeline with an explicit service-facing contract."""

    def __init__(self, converter: _UpstreamConverter, config: WorkerConfig) -> None:
        self._converter = converter
        self._config = config

    @classmethod
    def load(cls, config: WorkerConfig) -> "RvcInferenceEngine":
        """Validate configuration, initialize every model and run a warm-up."""
        config.validate()
        _prepare_environment(config)
        vc_module = cast("_VcModule", importlib.import_module("rvc.modules.vc.modules"))
        vc_module.load_audio = load_service_audio
        converter = vc_module.VC()
        if config.device not in ("auto", str(converter.config.device)):
            converter.config.device = config.device
            converter.config.is_half = config.device.startswith("cuda")
        converter.get_vc(str(config.model.resolve()))
        _validate_checkpoint(converter, config.speaker_id)
        converter.hubert_model = vc_module.load_hubert(
            converter.config,
            str(config.hubert.resolve()),
        )
        pipeline = _require_pipeline(converter)
        if config.f0_method == "rmvpe":
            rmvpe_module = cast("_RmvpeModule", importlib.import_module("rvc.lib.rmvpe"))
            pipeline.model_rmvpe = rmvpe_module.RMVPE(
                str(config.rmvpe.resolve()),
                is_half=converter.config.is_half,
                device=converter.config.device,
            )
        engine = cls(converter, config)
        engine.warm_up()
        return engine

    @property
    def runtime_metadata(self) -> Mapping[str, object]:
        """Return environment details used to diagnose image compatibility."""
        torch = cast("_TorchModule", importlib.import_module("torch"))
        device = str(self._converter.config.device)
        metadata: dict[str, object] = {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        if device.startswith("cuda") and torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        return metadata

    def convert(
        self,
        samples: NDArray[np.float32],
        sample_rate: int,
    ) -> ConvertedAudio:
        """Convert one complete TTS phrase through the loaded pipeline."""
        soundfile = importlib.import_module("soundfile")
        with tempfile.NamedTemporaryFile(suffix=".wav") as source:
            soundfile.write(source.name, samples, sample_rate, subtype="FLOAT")
            result_rate, result, _, error = self._converter.vc_inference(
                self._config.speaker_id,
                source.name,
                f0_up_key=self._config.pitch,
                f0_method=self._config.f0_method,
                index_file=str(self._config.index.resolve()),
                index_rate=self._config.index_rate,
                filter_radius=self._config.filter_radius,
                resample_sr=EXPECTED_OUTPUT_SAMPLE_RATE,
                rms_mix_rate=self._config.rms_mix_rate,
                protect=self._config.protect,
                hubert_path=str(self._config.hubert.resolve()),
            )
        if error is not None or result is None or result_rate is None:
            raise RuntimeError(error or "RVC returned no audio")
        return ConvertedAudio(
            samples=pcm_to_float32(result),
            sample_rate=result_rate,
        )

    def warm_up(self) -> None:
        """Exercise the complete pipeline before advertising readiness."""
        sample_count = round(_WARMUP_SAMPLE_RATE * _WARMUP_SECONDS)
        time_axis = np.arange(sample_count, dtype=np.float32) / _WARMUP_SAMPLE_RATE
        samples = (0.05 * np.sin(2 * np.pi * _WARMUP_FREQUENCY_HZ * time_axis)).astype(np.float32)
        result = self.convert(samples, _WARMUP_SAMPLE_RATE)
        if result.sample_rate != EXPECTED_OUTPUT_SAMPLE_RATE or not result.samples.size:
            msg = "RVC warm-up returned invalid audio"
            raise RuntimeError(msg)


def load_service_audio(path: str, sample_rate: int) -> NDArray[np.float32]:
    """Load the worker's temporary WAV and resample it for ContentVec."""
    librosa = importlib.import_module("librosa")
    soundfile = importlib.import_module("soundfile")
    samples, source_rate = soundfile.read(path, dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == _STEREO_DIMENSIONS:
        samples = samples.mean(axis=1)
    if source_rate != sample_rate:
        samples = np.asarray(
            librosa.resample(samples, orig_sr=source_rate, target_sr=sample_rate),
            dtype=np.float32,
        )
    return np.ascontiguousarray(samples, dtype=np.float32)


def _prepare_environment(config: WorkerConfig) -> None:
    if config.f0_method == "rmvpe":
        os.environ["rmvpe_root"] = str(config.rmvpe.parent)  # noqa: SIM112 - upstream name
    os.environ["hubert_path"] = str(config.hubert)  # noqa: SIM112 - upstream name
    os.environ["index_root"] = str(config.index.resolve().parent)  # noqa: SIM112
    if config.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _require_pipeline(converter: _UpstreamConverter) -> _UpstreamPipeline:
    if converter.pipeline is None:
        raise RuntimeError("RVC pipeline was not initialized")
    return converter.pipeline


def _validate_checkpoint(converter: _UpstreamConverter, speaker_id: int) -> None:
    if converter.version != "v2":
        msg = f"expected an RVC v2 checkpoint, got {converter.version!r}"
        raise RuntimeError(msg)
    if converter.tgt_sr != EXPECTED_OUTPUT_SAMPLE_RATE:
        msg = f"expected a 48 kHz checkpoint, got {converter.tgt_sr!r}"
        raise RuntimeError(msg)
    if converter.if_f0 != 1:
        msg = "the configured villager checkpoint must support F0"
        raise RuntimeError(msg)
    if type(converter.n_spk) is not int or not 0 <= speaker_id < converter.n_spk:
        msg = f"speaker ID {speaker_id} is outside the checkpoint range"
        raise RuntimeError(msg)
