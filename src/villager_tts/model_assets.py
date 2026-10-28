"""Download and install the pinned model artifacts used by the service."""

import argparse
import hashlib
import os
import shutil
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from villager_tts.core.runtime.artifacts import read_manifest


@dataclass(frozen=True, slots=True)
class DownloadAsset:
    """A remotely hosted file with a pinned content digest."""

    name: str
    url: str
    size_bytes: int
    sha256: str


_HF_REPOSITORY_COMPONENTS = 2
_MODEL_COMPONENTS = ("tts", "rvc", "villager")


def download_asset(
    asset: DownloadAsset,
    destination: Path,
    *,
    bearer_token: str | None = None,
) -> None:
    """Download one asset atomically and reject unexpected content."""
    if destination.is_file() and (
        destination.stat().st_size == asset.size_bytes and sha256_file(destination) == asset.sha256
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(  # noqa: S310 - trusted, pinned asset URLs
        asset.url,
        headers={"User-Agent": "villager-tts-model-installer/1"},
    )
    if bearer_token is not None:
        # Hugging Face returns a signed CDN redirect. Do not forward the Hub token
        # to a different origin; the signed URL is sufficient for the redirected request.
        request.add_unredirected_header("Authorization", f"Bearer {bearer_token}")
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,  # noqa: S310
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual_size = partial.stat().st_size
        if actual_size != asset.size_bytes:
            msg = f"size mismatch for {asset.name}: expected {asset.size_bytes}, got {actual_size}"
            raise ValueError(msg)
        actual = sha256_file(partial)
        if actual != asset.sha256:
            msg = f"checksum mismatch for {asset.name}: expected {asset.sha256}, got {actual}"
            raise ValueError(msg)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def install_models(
    models_dir: Path,
    components: Sequence[str],
    *,
    manifest_path: Path = Path("models/manifest.yaml"),
    hf_token: str | None = None,
) -> None:
    """Install selected model groups under a model root."""
    selected = set(components)
    unknown = selected - {"tts", "rvc", "villager"}
    if unknown:
        msg = f"unknown model components: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    manifest = read_manifest(manifest_path)
    artifacts = [artifact for artifact in manifest.artifacts if artifact.component in selected]
    if artifacts and not hf_token:
        msg = "HF_TOKEN is required to download private Hugging Face model artifacts"
        raise ValueError(msg)
    source = manifest.huggingface
    if artifacts and source is None:
        msg = "model manifest does not define a Hugging Face source"
        raise ValueError(msg)

    if source is None:
        return
    for artifact in artifacts:
        remote_path = artifact.path.as_posix()
        download_asset(
            DownloadAsset(
                artifact.path.name,
                _hugging_face_url(source.repository, source.revision, remote_path),
                artifact.size_bytes,
                artifact.sha256,
            ),
            _model_destination(models_dir, artifact.path),
            bearer_token=hf_token,
        )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _hugging_face_url(repository: str, revision: str, remote_path: str) -> str:
    repository_parts = repository.split("/")
    if len(repository_parts) != _HF_REPOSITORY_COMPONENTS or not all(repository_parts):
        msg = "Hugging Face repository must use the 'owner/name' format"
        raise ValueError(msg)
    if not revision:
        msg = "Hugging Face revision must not be empty"
        raise ValueError(msg)
    repository_path = "/".join(quote(part, safe="-._") for part in repository_parts)
    revision_path = quote(revision, safe="-._")
    path = PurePosixPath(remote_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        msg = f"invalid Hugging Face model path: {remote_path}"
        raise ValueError(msg)
    return (
        f"https://huggingface.co/{repository_path}/resolve/{revision_path}/"
        f"{quote(remote_path, safe='-._/')}?download=true"
    )


def _model_destination(models_dir: Path, relative_path: Path) -> Path:
    root = models_dir.resolve()
    destination = (root / relative_path).resolve()
    if not destination.is_relative_to(root):
        msg = f"model destination escapes model root: {relative_path}"
        raise ValueError(msg)
    return destination


def _read_hf_token(token_file: Path | None) -> str | None:
    if token_file is not None:
        try:
            token = token_file.read_text(encoding="utf-8").strip()
        except OSError as error:
            msg = f"unable to read Hugging Face token secret: {token_file}"
            raise RuntimeError(msg) from error
    else:
        token = os.environ.get("HF_TOKEN", "").strip()
    return token or None


def _model_component(value: str) -> str:
    if value not in _MODEL_COMPONENTS:
        choices = ", ".join(_MODEL_COMPONENTS)
        msg = f"invalid component {value!r}; choose one of: {choices}"
        raise argparse.ArgumentTypeError(msg)
    return value


def main() -> None:
    """Run the command-line model installer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "components",
        nargs="*",
        type=_model_component,
        metavar="{tts,rvc,villager}",
        help="model groups to install (default: tts rvc villager)",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--manifest-path", type=Path, default=Path("models/manifest.yaml"))
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        help="BuildKit secret containing a Hugging Face access token; defaults to HF_TOKEN",
    )
    arguments = parser.parse_args()
    components = arguments.components or _MODEL_COMPONENTS
    install_models(
        arguments.models_dir,
        components,
        manifest_path=arguments.manifest_path,
        hf_token=_read_hf_token(arguments.hf_token_file),
    )


if __name__ == "__main__":
    main()
