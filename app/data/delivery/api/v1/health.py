"""
GET /api/v1/health  — liveness and readiness probe.

Checks connectivity to infrastructure dependencies:
  database  — SQLAlchemy async ping (SQLite)
  cache     — DiskCache health check
  storage   — MinIO/S3 (optional)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from sqlalchemy import text

from delivery.cache.disk_cache import dataset_cache
from shared.db.session import AsyncSessionLocal
from shared.models.responses import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def _check_database() -> str:
    """Check SQLite database connectivity."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        logger.warning("Health: database check failed: %s", exc)
        return f"error: {exc}"


async def _check_cache() -> str:
    """Check DiskCache connectivity."""
    try:
        ok = await dataset_cache.ping()
        return "ok" if ok else "error: unreachable"
    except Exception as exc:
        return f"error: {exc}"


async def _check_storage() -> str:
    """Check MinIO/S3 connectivity (optional)."""
    import asyncio
    from shared.config import settings

    # If MinIO is not configured, return "skipped"
    if not hasattr(settings, 'MINIO_ENDPOINT') or not settings.MINIO_ENDPOINT:
        return "skipped"

    def _sync() -> str:
        try:
            from ingestion.storage.parquet_store import _s3_client
            client = _s3_client()
            client.head_bucket(Bucket=settings.MINIO_BUCKET)
            return "ok"
        except Exception as exc:
            return f"error: {exc}"

    try:
        return await asyncio.to_thread(_sync)
    except Exception as exc:
        return f"error: {exc}"


@router.get(
    "/health",
    summary="Service health check",
    response_model=HealthResponse,
    tags=["management"],
)
async def health_check() -> HealthResponse:
    database = await _check_database()
    cache = await _check_cache()
    storage = await _check_storage()

    # Consider "skipped" as "ok" for optional services
    statuses = [database, cache]
    if storage != "skipped":
        statuses.append(storage)
    
    all_ok = all(s == "ok" for s in statuses)
    degraded = not all_ok and any(s == "ok" for s in statuses)

    return HealthResponse(
        status="healthy" if all_ok else ("degraded" if degraded else "unhealthy"),
        database=database,
        cache=cache,
        storage=storage,
    )
