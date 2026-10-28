"""Command-line entry point tests."""

from typing import Any

import pytest
import uvicorn

from villager_tts.__main__ import main


def test_main_passes_configured_port_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def run(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setenv("VILLAGER_TTS_PORT", "9123")
    monkeypatch.setattr(uvicorn, "run", run)

    main()

    assert captured["port"] == 9123
    assert captured["host"] == "0.0.0.0"  # noqa: S104
