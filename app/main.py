"""
Unified Quant Research Platform FastAPI application entrypoint.

This is a merged service combining:
- Research Layer: Strategy, Feature, Model, Signal, Backtest, Validation
- Data Layer: Market data ingestion and delivery (OHLCV, News, Fundamentals, Macro)

Local backend: SQLite + DiskCache + APScheduler

Run locally with:
    uvicorn app.main:app --reload

Run with Data Layer integration:
    # Ensure environment variables are set:
    # DATA_SERVICE_URL=http://localhost:8001
    # DATA_SERVICE_API_KEY=your-api-key
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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
from app.api.data.router import router as data_router
from app.core.config import get_settings
from app.workers.task_queue import shutdown as shutdown_task_queue, is_ready as task_queue_ready

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan - startup and shutdown events."""
    # Startup: Initialize database tables
    try:
        from app.db.session import init_db
        await init_db()
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")

    # Startup: Initialize data layer providers if available
    try:
        from app.data.ingestion.providers import registry as provider_registry
        provider_registry.bootstrap()
    except ImportError:
        pass  # Data layer not fully initialized

    yield

    # Shutdown: cleanup task queue
    shutdown_task_queue()


app = FastAPI(
    title="Unified Quant Research Platform",
    version="1.0.0",
    description="""Unified Quant Research Platform combining:
    - Research Layer: Strategy development, feature engineering, model training, backtesting, validation
    - Data Layer: Market data ingestion and delivery (OHLCV, News, Fundamentals, Macro)
    """,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Unified Quant Research Platform",
        "version": "1.0.0",
        "backend": "local (SQLite + DiskCache)",
        "components": {
            "research_layer": "operational",
            "data_layer": "operational" if settings.DATA_SERVICE_URL else "not_configured",
            "task_queue": "operational" if task_queue_ready() else "not_ready"
        }
    }


api_prefix = settings.API_V1_PREFIX

# Research Layer routers
app.include_router(strategies_router,          prefix=api_prefix, tags=["strategies"])
app.include_router(promotion_router,           prefix=api_prefix, tags=["strategies"])
app.include_router(features_router,            prefix=api_prefix, tags=["features"])
app.include_router(feature_generation_router,  prefix=api_prefix, tags=["features"])
app.include_router(models_router,              prefix=api_prefix, tags=["models"])
app.include_router(model_training_router,      prefix=api_prefix, tags=["models"])
app.include_router(backtests_router,           prefix=api_prefix, tags=["backtests"])
app.include_router(sweep_router,              prefix=api_prefix, tags=["backtests"])
app.include_router(experiments_router,         prefix=api_prefix, tags=["experiments"])
app.include_router(signals_router,             prefix=api_prefix, tags=["signals"])
app.include_router(tracking_router,            prefix=api_prefix, tags=["tracking"])
app.include_router(validation_router,          prefix=api_prefix, tags=["validation"])
app.include_router(news_router,                prefix=api_prefix, tags=["news"])
app.include_router(agents_router,              prefix=api_prefix, tags=["agents"])

# Data Layer routers (merged into main app)
app.include_router(data_router,                prefix=api_prefix, tags=["data"])


@app.get("/api/v1/tasks/{task_id}", tags=["tasks"])
async def get_task_status(task_id: str):
    """Poll the status and result of any background task."""
    from app.workers.task_queue import get_task_status as get_local_task_status

    status = get_local_task_status(task_id)
    if status is None:
        return {"task_id": task_id, "status": "NOT_FOUND", "error": "Task not found"}
    
    response = {
        "task_id": task_id,
        "status": status.get("status", "UNKNOWN"),
    }
    
    if status.get("error"):
        response["error"] = status["error"]
    elif status.get("result") is not None:
        response["result"] = status["result"]
    
    return response
