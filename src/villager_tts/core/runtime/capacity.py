"""Bounded inference admission control."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from villager_tts.common.errors import CapacityExceededError


class CapacityLimiter:
    """Limit expensive concurrent synthesis operations."""

    def __init__(self, capacity: int, acquire_timeout_seconds: float) -> None:
        if capacity < 1:
            msg = "capacity must be at least one"
            raise ValueError(msg)
        self._semaphore = asyncio.Semaphore(capacity)
        self._acquire_timeout_seconds = acquire_timeout_seconds

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None, None]:
        """Acquire a slot or fail when the bounded wait expires."""
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._acquire_timeout_seconds,
            )
        except TimeoutError as error:
            raise CapacityExceededError from error
        try:
            yield
        finally:
            self._semaphore.release()
