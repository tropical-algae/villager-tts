"""Cooperative cancellation for blocking inference adapters."""

import asyncio


class CancellationToken:
    """A lightweight cancellation signal shared across pipeline stages."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation was requested."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    def raise_if_cancelled(self) -> None:
        """Raise the standard asyncio cancellation exception if requested."""
        if self.is_cancelled:
            raise asyncio.CancelledError

    async def wait(self) -> None:
        """Wait until cancellation is requested."""
        await self._event.wait()
