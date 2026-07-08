"""
Cache management endpoints.

GET  /api/v1/cache/stats              — cache key counts, size usage
POST /api/v1/cache/refresh            — force re-ingestion by invalidating a hash
DELETE /api/v1/cache/invalidate       — remove a specific hash from cache
DELETE /api/v1/cache/flush/{pattern}  — flush all keys matching a pattern

These are management endpoints — they use the APIResponse envelope and
require the same API key as the data endpoints.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import diskcache
from fastapi import APIRouter, Depends, HTTPException, Query, status

from delivery.cache.disk_cache import dataset_cache
from ingestion.pipeline import compute_hash
from shared.auth.dependencies import verify_api_key
from shared.config import settings
from shared.models.responses import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", dependencies=[Depends(verify_api_key)])

_cache = diskcache.Cache(settings.CACHE_DIR, tag_index=True)


def _get_cache_size() -> int:
    """Get total size of cache directory in bytes."""
    total = 0
    cache_dir = settings.CACHE_DIR
    if os.path.exists(cache_dir):
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
    return total


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


# ── Stats ─────────────────────────────────────────────────────────────────────


@router.get("/stats", response_model=APIResponse, summary="Cache statistics")
async def cache_stats() -> APIResponse:
    """
    Returns the number of cached dataset keys and DiskCache size.
    """
    t = time.perf_counter()
    try:
        # Count keys with ds: prefix
        cached_datasets = sum(1 for key in _cache.iterkeys() if str(key).startswith("ds:"))
        
        # Get cache size
        size_bytes = _get_cache_size()
        size_human = _format_size(size_bytes)

        return APIResponse(
            status="success",
            message="Cache statistics retrieved.",
            data={
                "cached_datasets": cached_datasets,
                "cache_size_human": size_human,
                "cache_size_bytes": size_bytes,
                "backend": "diskcache",
            },
            execution_time=round(time.perf_counter() - t, 4),
        )
    except Exception as exc:
        logger.error("cache_stats failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cache unavailable: {exc}",
        )


# ── Refresh (force re-ingestion) ──────────────────────────────────────────────


@router.post("/refresh", response_model=APIResponse, summary="Force re-ingestion")
async def cache_refresh(
    data_type: str = Query(..., description="ohlcv | news | fundamentals | macro"),
    symbol: str | None = Query(None),
    timeframe: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    series: str | None = Query(None),
    provider: str = Query("yahoo"),
) -> APIResponse:
    """
    Invalidates the cache for a specific dataset so the next request
    triggers a fresh download from the upstream provider.

    Use this when:
    - Corporate action adjustments have been revised retroactively.
    - Provider data has been corrected.
    - You want to force an update before the TTL expires.
    """
    t = time.perf_counter()

    params: dict[str, Any] = {"provider": provider}
    if symbol:
        params["symbol"] = symbol.upper()
    if timeframe:
        params["timeframe"] = timeframe
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if series:
        params["series"] = series.upper()

    hash_val = compute_hash(data_type, params)
    await dataset_cache.invalidate(hash_val)

    logger.info("Cache refresh: data_type=%s hash=%s", data_type, hash_val[:12])

    return APIResponse(
        status="success",
        message=f"Cache invalidated for {data_type}. Next request will re-ingest.",
        data={"hash": hash_val, "data_type": data_type, "params": params},
        execution_time=round(time.perf_counter() - t, 4),
    )


# ── Invalidate by hash ────────────────────────────────────────────────────────


@router.delete("/invalidate", response_model=APIResponse, summary="Invalidate by hash")
async def cache_invalidate(
    hash_val: str = Query(..., description="SHA-256 hash of the dataset to invalidate"),
) -> APIResponse:
    """
    Directly invalidates a cache entry by its SHA-256 hash.
    Use /cache/stats to discover active hashes.
    """
    t = time.perf_counter()
    await dataset_cache.invalidate(hash_val)

    return APIResponse(
        status="success",
        message=f"Cache key invalidated: {hash_val[:12]}...",
        data={"hash": hash_val},
        execution_time=round(time.perf_counter() - t, 4),
    )


# ── Flush by symbol (bulk invalidation) ──────────────────────────────────────


@router.delete("/flush", response_model=APIResponse, summary="Flush all cache for a symbol")
async def cache_flush_symbol(
    symbol: str = Query(..., description="Symbol to flush all cached datasets for"),
) -> APIResponse:
    """
    Flushes ALL cached datasets for a given symbol (ohlcv, news, fundamentals).

    Useful after a ticker rename, merger, or data correction.
    """
    t = time.perf_counter()
    symbol = symbol.upper()
    deleted = 0

    try:
        # Iterate and delete keys containing the symbol
        keys_to_delete = []
        for key in _cache.iterkeys():
            key_str = str(key)
            if key_str.startswith("ds:"):
                val = _cache.get(key)
                if val and f"/{symbol}/" in str(val):
                    keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del _cache[key]
            deleted += 1

        logger.info("Cache flush: symbol=%s deleted=%d", symbol, deleted)

        return APIResponse(
            status="success",
            message=f"Flushed {deleted} cache entries for {symbol}.",
            data={"symbol": symbol, "deleted": deleted},
            execution_time=round(time.perf_counter() - t, 4),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cache error: {exc}",
        )
