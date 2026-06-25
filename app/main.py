"""
Research Layer FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agents.router import router as agents_router
from app.api.backtests.router import router as backtests_router
from app.api.backtests.sweep_router import router as sweep_router
from app.api.experiments.router import router as experiments_router
from app.api.features.generation_router import router as feature_generation_router
from app.api.features.router import router as features_router
from app.api.models.router import router as models_router
from app.api.models.training_router import router as model_training_router
from app.api.news.router import router as news_router
from app.api.signals.router import router as signals_router
from app.api.strategies.router import router as strategies_router
from app.api.strategies.promotion_router import router as promotion_router
from app.api.tracking.router import router as tracking_router
from app.api.validation.router import router as validation_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Standalone Research Layer service — Quant Research Platform.",
)

"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    """Start Kafka consumer for Market Data Layer events on app boot."""
    from app.events.bus import get_consumer
    from app.events.handlers import dispatch
    consumer = get_consumer()
    consumer.start(
        topics=["market.datasetcreated", "market.datasetupdated"],
        handler=dispatch,
    )


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "service": settings.APP_NAME, "env": settings.APP_ENV}


api_prefix = settings.API_V1_PREFIX

app.include_router(strategies_router,          prefix=api_prefix)
app.include_router(promotion_router,           prefix=api_prefix)
app.include_router(features_router,            prefix=api_prefix)
app.include_router(feature_generation_router,  prefix=api_prefix)
app.include_router(models_router,              prefix=api_prefix)
app.include_router(model_training_router,      prefix=api_prefix)
app.include_router(backtests_router,           prefix=api_prefix)
app.include_router(sweep_router,               prefix=api_prefix)
app.include_router(experiments_router,         prefix=api_prefix)
app.include_router(signals_router,             prefix=api_prefix)
app.include_router(tracking_router,            prefix=api_prefix)
app.include_router(validation_router,          prefix=api_prefix)
app.include_router(news_router,                prefix=api_prefix)
app.include_router(agents_router,              prefix=api_prefix)


@app.get("/api/v1/tasks/{task_id}", tags=["tasks"])
async def get_task_status(task_id: str):
    """Poll the status and result of any Celery background task."""
    from celery.result import AsyncResult
    from app.workers.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)
    response: dict = {"task_id": task_id, "status": result.status}
    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
    return response
