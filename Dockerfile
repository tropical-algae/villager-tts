ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.21@sha256:ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY packages/rvc-protocol ./packages/rvc-protocol
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM builder AS models-builder
COPY models/manifest.yaml ./config/models-manifest.yaml
RUN --mount=type=secret,id=HF_TOKEN,required=true \
    /app/.venv/bin/villager-tts-download-models \
        --models-dir /app/models \
        --manifest-path /app/config/models-manifest.yaml \
        --hf-token-file /run/secrets/HF_TOKEN \
    && /app/.venv/bin/python -c \
        "from pathlib import Path; from villager_tts.core.runtime.artifacts import verify_manifest; verify_manifest(Path('/app/config/models-manifest.yaml'), artifact_root=Path('/app/models'))"

FROM builder AS rvc-base
COPY workers/rvc/pyproject.toml workers/rvc/uv.lock ./workers/rvc/
COPY workers/rvc/src ./workers/rvc/src

FROM rvc-base AS rvc-cpu-builder
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project workers/rvc --frozen --no-dev --no-editable --extra cpu
RUN /app/workers/rvc/.venv/bin/python -c \
    "import torch, torchaudio; assert torch.__version__ == '2.1.2+cpu'; assert torch.version.cuda is None; assert torchaudio.__version__ == '2.1.2+cpu'"

FROM rvc-base AS rvc-cuda121-builder
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project workers/rvc --frozen --no-dev --no-editable --extra cuda121
RUN /app/workers/rvc/.venv/bin/python -c \
    "import torch, torchaudio; assert torch.__version__ == '2.1.2+cu121'; assert torch.version.cuda == '12.1'; assert torchaudio.__version__ == '2.1.2+cu121'"

FROM ${PYTHON_IMAGE} AS runtime
WORKDIR /app
ENV VILLAGER_TTS_RVC_WORKER_COMMAND="/app/workers/rvc/.venv/bin/python -m villager_rvc_worker" \
    VILLAGER_TTS_MODEL_MANIFEST_PATH=/app/config/models-manifest.yaml \
    VILLAGER_TTS_MODEL_ARTIFACT_ROOT=/app/models

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libgomp1 libsndfile1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/src ./src
COPY --from=rvc-base --chown=app:app /app/workers/rvc/src ./workers/rvc/src
COPY --from=models-builder --chown=app:app /app/config/models-manifest.yaml ./config/models-manifest.yaml
COPY --from=models-builder --chown=app:app /app/models ./models

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import os, urllib.request; port=os.getenv('VILLAGER_TTS_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health/live', timeout=2)"]
ENTRYPOINT ["tini", "--"]
CMD ["/app/.venv/bin/villager-tts"]

FROM runtime AS cpu
COPY --from=rvc-cpu-builder --chown=app:app /app/workers/rvc/.venv ./workers/rvc/.venv
ENV VILLAGER_TTS_RVC_DEVICE=cpu

FROM runtime AS cuda
COPY --from=rvc-cuda121-builder --chown=app:app /app/workers/rvc/.venv ./workers/rvc/.venv
ENV VILLAGER_TTS_RVC_DEVICE=cuda

FROM cpu AS production
