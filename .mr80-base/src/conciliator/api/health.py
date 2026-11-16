from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app: str


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", app="conciliator")


@router.get(
    "/readyz", response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
async def readyz(request: Request, response: Response) -> HealthResponse:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        response.status_code = 503
        return HealthResponse(status="degraded", app="conciliator")

    def _ping() -> bool:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    ok = await run_in_threadpool(_ping)
    if not ok:
        response.status_code = 503
    return HealthResponse(
        status="ok" if ok else "degraded", app="conciliator",
    )
