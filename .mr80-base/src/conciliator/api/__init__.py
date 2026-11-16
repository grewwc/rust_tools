from fastapi import APIRouter

from conciliator.api import aeolus, deployments, health, oauth, trace, webhooks

router = APIRouter()
router.include_router(health.router)
router.include_router(oauth.router)
router.include_router(trace.router)
router.include_router(webhooks.router)
router.include_router(aeolus.router)
router.include_router(deployments.router)

__all__ = ["router"]
