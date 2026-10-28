"""Tests for pinned private-model acquisition."""

import hashlib
import sys
from pathlib import Path

import pytest

from villager_tts import model_assets
from villager_tts.model_assets import (
    DownloadAsset,
    download_asset,
    install_models,
    sha256_file,
)


def test_download_asset_is_atomic_and_verifies_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"trusted model")
    destination = tmp_path / "models" / "model.bin"
    asset = DownloadAsset("model.bin", source.as_uri(), source.stat().st_size, sha256_file(source))

    download_asset(asset, destination)
    assert destination.read_bytes() == b"trusted model"
    download_asset(asset, destination)

    bad = DownloadAsset("model.bin", source.as_uri(), source.stat().st_size, "f" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_asset(bad, destination)
    assert destination.read_bytes() == b"trusted model"


def test_download_asset_rejects_an_unexpected_size(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"trusted model")
    asset = DownloadAsset(
        "model.bin",
        source.as_uri(),
        source.stat().st_size + 1,
        sha256_file(source),
    )

    with pytest.raises(ValueError, match="size mismatch"):
        download_asset(asset, tmp_path / "destination.bin")


def test_private_model_assets_require_a_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HF_TOKEN"):
        install_models(tmp_path, ["villager"])


def test_private_villager_assets_use_authenticated_hugging_face_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[DownloadAsset, Path, str | None]] = []
    auth_value = "fixture-auth-value"

    def capture_download(
        asset: DownloadAsset,
        destination: Path,
        *,
        bearer_token: str | None = None,
    ) -> None:
        captured.append((asset, destination, bearer_token))

    monkeypatch.setattr(model_assets, "download_asset", capture_download)
    install_models(
        tmp_path,
        ["villager"],
        hf_token=auth_value,
    )

    assert [asset.name for asset, _, _ in captured] == [
        "villagerminecraft.pth",
        "added_IVF99_Flat_nprobe_1_villagerminecraft_v2.index",
    ]
    assert all(
        asset.url.startswith(
            "https://huggingface.co/tropicalalgae/tts-rvc-models/resolve/"
            "8291f081c191b58e8569e16068a15849eb185b2c/villager/"
        )
        for asset, _, _ in captured
    )
    assert all(token == auth_value for _, _, token in captured)
    assert all(destination.parent == tmp_path / "villager" for _, destination, _ in captured)


def test_all_components_are_selected_from_the_single_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[DownloadAsset, Path]] = []
    auth_value = "fixture-auth-value"

    def capture_download(
        asset: DownloadAsset,
        destination: Path,
        *,
        bearer_token: str | None = None,
    ) -> None:
        assert bearer_token == auth_value
        captured.append((asset, destination))

    monkeypatch.setattr(model_assets, "download_asset", capture_download)
    install_models(tmp_path, ["tts", "rvc", "villager"], hf_token=auth_value)

    assert len(captured) == 14
    assert {destination.relative_to(tmp_path).parts[0] for _, destination in captured} == {
        "tts",
        "rvc",
        "villager",
    }
    assert all(asset.size_bytes > 0 for asset, _ in captured)


def test_cli_installs_all_components_when_none_are_given(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[Path, tuple[str, ...], Path, str | None]] = []
    token_file = tmp_path / "hf-token"
    token_file.write_text("fixture-auth-value\n", encoding="utf-8")

    def capture_install(
        models_dir: Path,
        components: tuple[str, ...],
        *,
        manifest_path: Path,
        hf_token: str | None,
    ) -> None:
        captured.append((models_dir, components, manifest_path, hf_token))

    models_dir = tmp_path / "models"
    manifest_path = tmp_path / "manifest.yaml"
    monkeypatch.setattr(model_assets, "install_models", capture_install)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "villager-tts-download-models",
            "--models-dir",
            str(models_dir),
            "--manifest-path",
            str(manifest_path),
            "--hf-token-file",
            str(token_file),
        ],
    )

    model_assets.main()

    assert captured == [
        (
            models_dir,
            ("tts", "rvc", "villager"),
            manifest_path,
            "fixture-auth-value",
        )
    ]


def test_installer_rejects_unknown_components_and_a_missing_source(tmp_path: Path) -> None:
    auth_value = "fixture-auth-value"
    with pytest.raises(ValueError, match="unknown model components"):
        install_models(tmp_path, ["unknown"], hf_token=auth_value)

    content = b"model"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "version: 1\nartifacts:\n  - name: model\n    component: tts\n"
        f"    path: model.bin\n    size_bytes: {len(content)}\n"
        f"    sha256: {hashlib.sha256(content).hexdigest()}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not define"):
        install_models(tmp_path, ["tts"], manifest_path=manifest, hf_token=auth_value)
