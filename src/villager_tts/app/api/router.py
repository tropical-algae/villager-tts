"""Top-level API router."""

from fastapi import APIRouter

from villager_tts.app.api.endpoints import health, speech

router = APIRouter()
router.include_router(health.router)
router.include_router(speech.router, prefix="/v1/audio", tags=["speech"])
