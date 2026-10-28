"""RVC worker configuration tests."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from villager_rvc_worker.config import WorkerConfig, parse_config

InvalidConfigFactory = Callable[[WorkerConfig], WorkerConfig]
INVALID_CONFIGS: tuple[tuple[InvalidConfigFactory, str], ...] = (
    (lambda config: replace(config, speaker_id=-1), "speaker ID"),
    (lambda config: replace(config, pitch=25), "pitch"),
    (lambda config: replace(config, index_rate=1.1), "index rate"),
    (lambda config: replace(config, filter_radius=8), "filter radius"),
    (lambda config: replace(config, rms_mix_rate=-0.1), "RMS mix rate"),
    (lambda config: replace(config, protect=0.6), "protect"),
)


def _model_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "model": tmp_path / "voice.pth",
        "index": tmp_path / "voice.index",
        "hubert": tmp_path / "hubert.pt",
        "rmvpe": tmp_path / "rmvpe.pt",
    }
    for path in paths.values():
        path.write_bytes(b"model")
    return paths


def _valid_config(tmp_path: Path) -> WorkerConfig:
    paths = _model_files(tmp_path)
    return WorkerConfig(
        model=paths["model"],
        index=paths["index"],
        hubert=paths["hubert"],
        rmvpe=paths["rmvpe"],
        device="cpu",
    )


def test_parse_config_returns_typed_validated_values(tmp_path: Path) -> None:
    paths = _model_files(tmp_path)
    arguments: list[str] = []
    for name, path in paths.items():
        arguments.extend((f"--{name}", str(path)))
    arguments.extend(("--device", "cuda:0", "--pitch", "6", "--f0-method", "pm"))

    config = parse_config(arguments)
    config.validate()

    assert config.device == "cuda:0"
    assert config.pitch == 6
    assert config.f0_method == "pm"


@pytest.mark.parametrize(
    ("factory", "message"),
    INVALID_CONFIGS,
)
def test_config_rejects_values_outside_service_contract(
    tmp_path: Path,
    factory: InvalidConfigFactory,
    message: str,
) -> None:
    config = factory(_valid_config(tmp_path))

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_config_rejects_missing_model_file(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)
    config.hubert.unlink()

    with pytest.raises(FileNotFoundError, match="hubert"):
        config.validate()


def test_pm_configuration_does_not_require_the_rmvpe_model(tmp_path: Path) -> None:
    config = replace(_valid_config(tmp_path), f0_method="pm")
    config.rmvpe.unlink()

    config.validate()
