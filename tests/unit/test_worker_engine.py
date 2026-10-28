"""RVC worker engine tests with lightweight upstream test doubles."""

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from villager_rvc_worker import engine as worker_engine
from villager_rvc_worker.config import WorkerConfig


class _FakeConverter:
    def __init__(self) -> None:
        self.config = SimpleNamespace(device="cpu", is_half=False)
        self.version: object = "v2"
        self.tgt_sr: object = 48_000
        self.if_f0: object = 1
        self.n_spk: object = 2
        self.pipeline: object | None = SimpleNamespace(model_rmvpe=None)
        self.hubert_model: object | None = None
        self.loaded_model: str | None = None
        self.result_rate: int | None = 48_000
        self.result: object | None = np.asarray([-32768, 0, 32767], dtype=np.int16)
        self.error: str | None = None

    def get_vc(self, model_path: str) -> None:
        self.loaded_model = model_path

    def vc_inference(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
        return self.result_rate, self.result, None, self.error


class _FakeSoundFile:
    def __init__(self) -> None:
        self.writes: list[tuple[int, int, str]] = []
        self.read_samples = np.asarray([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        self.read_rate = 24_000

    def write(
        self,
        _path: str,
        samples: NDArray[np.float32],
        sample_rate: int,
        *,
        subtype: str,
    ) -> None:
        self.writes.append((samples.size, sample_rate, subtype))

    def read(
        self,
        _path: str,
        *,
        dtype: str,
        always_2d: bool,
    ) -> tuple[NDArray[np.float32], int]:
        assert dtype == "float32"
        assert always_2d is False
        return self.read_samples, self.read_rate


class _FakeLibrosa:
    def __init__(self) -> None:
        self.calls = 0

    def resample(
        self,
        samples: NDArray[np.float32],
        *,
        orig_sr: int,
        target_sr: int,
    ) -> NDArray[np.float32]:
        assert orig_sr == 24_000
        assert target_sr == 16_000
        self.calls += 1
        return samples[:1]


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def current_device(self) -> int:
        return 0

    def get_device_name(self, device: int) -> str:
        assert device == 0
        return "Test GPU"


def _config(tmp_path: Path, *, device: str = "cpu", speaker_id: int = 0) -> WorkerConfig:
    paths = [
        tmp_path / "voice.pth",
        tmp_path / "voice.index",
        tmp_path / "hubert.pt",
        tmp_path / "rmvpe.pt",
    ]
    for path in paths:
        path.write_bytes(b"model")
    return WorkerConfig(
        model=paths[0],
        index=paths[1],
        hubert=paths[2],
        rmvpe=paths[3],
        device=cast("Any", device),
        speaker_id=speaker_id,
    )


def _install_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
    converter: _FakeConverter,
    soundfile: _FakeSoundFile,
    *,
    cuda_available: bool = False,
    imports: list[str] | None = None,
) -> None:
    original_import = importlib.import_module
    modules: dict[str, object] = {
        "rvc.modules.vc.modules": SimpleNamespace(
            load_audio=None,
            VC=lambda: converter,
            load_hubert=lambda _config, path: f"hubert:{path}",
        ),
        "rvc.lib.rmvpe": SimpleNamespace(RMVPE=lambda *_args, **_kwargs: "rmvpe"),
        "soundfile": soundfile,
        "torch": SimpleNamespace(
            __version__="2.1.2+cu121" if cuda_available else "2.1.2+cpu",
            version=SimpleNamespace(cuda="12.1" if cuda_available else None),
            cuda=_FakeCuda(available=cuda_available),
        ),
    }

    def fake_import(name: str, package: str | None = None) -> object:
        if imports is not None:
            imports.append(name)
        if name in modules:
            return modules[name]
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)


def test_engine_loads_models_warms_up_and_reports_cuda_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _FakeConverter()
    soundfile = _FakeSoundFile()
    _install_fake_modules(monkeypatch, converter, soundfile, cuda_available=True)

    engine = worker_engine.RvcInferenceEngine.load(_config(tmp_path, device="cuda:0"))
    result = engine.convert(np.zeros(4, dtype=np.float32), 24_000)

    assert converter.config.device == "cuda:0"
    assert converter.config.is_half is True
    assert converter.loaded_model == str((tmp_path / "voice.pth").resolve())
    assert converter.hubert_model == f"hubert:{(tmp_path / 'hubert.pt').resolve()}"
    assert cast("Any", converter.pipeline).model_rmvpe == "rmvpe"
    assert result.sample_rate == 48_000
    assert result.samples.tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768])
    assert soundfile.writes[0][1:] == (44_100, "FLOAT")
    assert engine.runtime_metadata == {
        "device": "cuda:0",
        "torch_version": "2.1.2+cu121",
        "cuda_version": "12.1",
        "gpu_name": "Test GPU",
    }


def test_engine_skips_rmvpe_loading_for_pm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _FakeConverter()
    soundfile = _FakeSoundFile()
    imports: list[str] = []
    _install_fake_modules(monkeypatch, converter, soundfile, imports=imports)
    config = replace(_config(tmp_path), f0_method="pm")
    config.rmvpe.unlink()

    worker_engine.RvcInferenceEngine.load(config)

    assert "rvc.lib.rmvpe" not in imports
    assert cast("Any", converter.pipeline).model_rmvpe is None


def test_engine_propagates_upstream_conversion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _FakeConverter()
    converter.result = None
    converter.error = "conversion failed"
    soundfile = _FakeSoundFile()
    _install_fake_modules(monkeypatch, converter, soundfile)
    engine = worker_engine.RvcInferenceEngine(
        converter=cast("Any", converter), config=_config(tmp_path)
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        engine.convert(np.zeros(4, dtype=np.float32), 24_000)


def test_service_audio_loader_downmixes_and_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    soundfile = _FakeSoundFile()
    librosa = _FakeLibrosa()
    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "soundfile":
            return soundfile
        if name == "librosa":
            return librosa
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    result = worker_engine.load_service_audio("source.wav", 16_000)

    assert result.tolist() == [0.25]
    assert result.flags.c_contiguous
    assert librosa.calls == 1


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("version", "v1", "v2 checkpoint"),
        ("tgt_sr", 40_000, "48 kHz"),
        ("if_f0", 0, "support F0"),
        ("n_spk", 0, "outside"),
    ],
)
def test_checkpoint_contract_is_enforced(
    attribute: str,
    value: object,
    message: str,
) -> None:
    converter = _FakeConverter()
    setattr(converter, attribute, value)

    with pytest.raises(RuntimeError, match=message):
        worker_engine._validate_checkpoint(cast("Any", converter), speaker_id=0)  # noqa: SLF001


def test_checkpoint_rejects_speaker_outside_model_range() -> None:
    with pytest.raises(RuntimeError, match="speaker ID 2"):
        worker_engine._validate_checkpoint(  # noqa: SLF001
            cast("Any", _FakeConverter()),
            speaker_id=2,
        )


def test_missing_pipeline_and_invalid_warmup_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _FakeConverter()
    converter.pipeline = None
    with pytest.raises(RuntimeError, match="not initialized"):
        worker_engine._require_pipeline(cast("Any", converter))  # noqa: SLF001

    converter.pipeline = SimpleNamespace(model_rmvpe=None)
    converter.result_rate = 44_100
    soundfile = _FakeSoundFile()
    _install_fake_modules(monkeypatch, converter, soundfile)
    engine = worker_engine.RvcInferenceEngine(
        converter=cast("Any", converter), config=_config(tmp_path)
    )
    with pytest.raises(RuntimeError, match="warm-up"):
        engine.warm_up()
