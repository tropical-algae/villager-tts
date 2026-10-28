"""Runtime primitive tests."""

import asyncio
import hashlib
from pathlib import Path

import pytest

from villager_tts.common.errors import CapacityExceededError
from villager_tts.core.runtime.artifacts import read_manifest, verify_manifest
from villager_tts.core.runtime.cancellation import CancellationToken
from villager_tts.core.runtime.capacity import CapacityLimiter


async def test_cancellation_token_signals_and_raises() -> None:
    token = CancellationToken()
    waiter = asyncio.create_task(token.wait())
    assert not token.is_cancelled
    token.cancel()
    await waiter
    assert token.is_cancelled
    with pytest.raises(asyncio.CancelledError):  # type: ignore[unreachable]
        token.raise_if_cancelled()


async def test_capacity_limiter_rejects_when_slot_is_occupied() -> None:
    limiter = CapacityLimiter(1, acquire_timeout_seconds=0.001)
    async with limiter.slot():
        with pytest.raises(CapacityExceededError):
            async with limiter.slot():
                pass
    async with limiter.slot():
        pass


def test_capacity_limiter_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        CapacityLimiter(0, 1)


def test_manifest_verifies_hash_and_rejects_mismatch(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "model.bin"
    artifact.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    manifest = models / "manifest.yaml"
    manifest.write_text(
        f"version: 1\nartifacts:\n  - name: test\n    component: tts\n"
        f"    path: model.bin\n    size_bytes: {artifact.stat().st_size}\n"
        f"    sha256: {digest}\n    required: true\n",
        encoding="utf-8",
    )

    assert read_manifest(manifest).artifacts[0].name == "test"
    verify_manifest(manifest)
    artifact.write_bytes(b"untrust")
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(manifest)

    artifact.write_bytes(b"wrong size")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_manifest(manifest)


def test_manifest_rejects_missing_and_escaping_artifacts(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    manifest = models / "manifest.yaml"
    digest = "f" * 64
    manifest.write_text(
        f"version: 1\nartifacts:\n  - name: test\n    component: tts\n"
        "    path: missing.bin\n    size_bytes: 1\n"
        f"    sha256: {digest}\n    required: true\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_manifest(manifest)

    manifest.write_text(
        f"version: 1\nartifacts:\n  - name: test\n    component: tts\n"
        "    path: ../outside.bin\n    size_bytes: 1\n"
        f"    sha256: {digest}\n    required: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        verify_manifest(manifest)


def test_manifest_reports_a_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model manifest is missing"):
        verify_manifest(tmp_path / "manifest.yaml")


def test_manifest_allows_missing_optional_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "version: 1\nartifacts:\n  - name: optional\n    component: rvc\n"
        "    path: missing.bin\n    size_bytes: 1\n"
        f"    sha256: {'f' * 64}\n    required: false\n",
        encoding="utf-8",
    )
    verify_manifest(manifest)


def test_manifest_can_verify_a_separate_artifact_root(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "model.bin"
    artifact.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    manifest = config / "manifest.yaml"
    manifest.write_text(
        f"version: 1\nartifacts:\n  - name: model\n    component: villager\n"
        f"    path: model.bin\n    size_bytes: {artifact.stat().st_size}\n"
        f"    sha256: {digest}\n    required: true\n",
        encoding="utf-8",
    )

    verify_manifest(manifest, artifact_root=models)
