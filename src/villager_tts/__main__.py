"""Command-line entry point."""

import uvicorn

from villager_tts.config import Settings


def main() -> None:
    """Run the ASGI server."""
    settings = Settings()
    uvicorn.run(
        "villager_tts.app.factory:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104
        port=settings.port,
    )


if __name__ == "__main__":
    main()
