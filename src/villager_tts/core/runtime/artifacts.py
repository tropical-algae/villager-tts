"""Model artifact integrity verification."""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Artifact(BaseModel):
    """One immutable model artifact."""

    model_config = ConfigDict(extra="forbid")

    name: str
    component: Literal["tts", "rvc", "villager"]
    path: Path
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    required: bool = True


class HuggingFaceSource(BaseModel):
    """Default private repository used by Hugging Face file downloads."""

    model_config = ConfigDict(extra="forbid")

    repository: str
    revision: str


class ModelManifest(BaseModel):
    """Versioned list of trusted model files."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    huggingface: HuggingFaceSource | None = None
    artifacts: list[Artifact]

    @model_validator(mode="after")
    def validate_references(self) -> "ModelManifest":
        """Reject ambiguous artifact names and destinations."""
        artifact_names = {artifact.name for artifact in self.artifacts}
        if len(artifact_names) != len(self.artifacts):
            msg = "model artifact names must be unique"
            raise ValueError(msg)
        artifact_paths = {artifact.path for artifact in self.artifacts}
        if len(artifact_paths) != len(self.artifacts):
            msg = "model artifact paths must be unique"
            raise ValueError(msg)
        return self


def read_manifest(path: Path) -> ModelManifest:
    """Parse a model manifest without executing arbitrary content."""
    if not path.is_file():
        msg = f"model manifest is missing: {path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelManifest.model_validate(raw)


def verify_manifest(path: Path, *, artifact_root: Path | None = None) -> None:
    """Verify every required artifact under an explicit or manifest-relative root."""
    manifest = read_manifest(path)
    base_dir = (artifact_root or path.parent).resolve()
    for artifact in manifest.artifacts:
        artifact_path = (base_dir / artifact.path).resolve()
        if not artifact_path.is_relative_to(base_dir):
            msg = f"artifact escapes model root: {artifact.path}"
            raise ValueError(msg)
        if not artifact_path.is_file():
            if artifact.required:
                msg = f"required model artifact is missing: {artifact_path}"
                raise FileNotFoundError(msg)
            continue
        actual_size = artifact_path.stat().st_size
        if actual_size != artifact.size_bytes:
            msg = (
                f"model artifact size mismatch: {artifact.name}; "
                f"expected {artifact.size_bytes}, got {actual_size}"
            )
            raise ValueError(msg)
        digest = _sha256(artifact_path)
        if digest != artifact.sha256:
            msg = f"model artifact checksum mismatch: {artifact.name}"
            raise ValueError(msg)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
