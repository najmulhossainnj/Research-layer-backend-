"""
DatasetCache — DiskCache-based local cache for dataset storage URIs.

Key pattern: ds:{sha256_hash}
Value:        s3://bucket/path/to/file.parquet

This is a pure cache — it stores nothing about the data itself, only the
mapping from a deterministic request hash to a Parquet file location.

This replaces the previous Redis-based cache with a local disk-based cache
using the DiskCache library.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from diskcache import Cache

logger = logging.getLogger(__name__)

_KEY_PREFIX = "ds:"


def _get_cache_dir() -> str:
    """Get the cache directory for dataset storage."""
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    cache_dir = os.path.join(app_dir, "cache", "dataset")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class DatasetCache:
    """Local disk-based cache for dataset storage URIs."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._cache_dir = cache_dir or _get_cache_dir()
        self._cache: Optional[Cache] = None

    def _get_cache(self) -> Cache:
        if self._cache is None:
            self._cache = Cache(self._cache_dir, disk_pickle_protocol=4)
        return self._cache

    async def get(self, hash_val: str) -> Optional[str]:
        """Return the cached storage_uri, or None on miss."""
        try:
            cache = self._get_cache()
            val = cache.get(f"{_KEY_PREFIX}{hash_val}")
            if val:
                logger.debug("Cache HIT: hash=%s", hash_val[:12])
            else:
                logger.debug("Cache MISS: hash=%s", hash_val[:12])
            return val
        except Exception as exc:
            logger.warning("DatasetCache.get error: %s", exc)
            return None

    async def set(self, hash_val: str, uri: str, ttl: int) -> None:
        """Store hash → URI with a TTL in seconds."""
        try:
            cache = self._get_cache()
            cache.set(f"{_KEY_PREFIX}{hash_val}", uri, expire=ttl)
            logger.debug("Cache SET: hash=%s ttl=%ds", hash_val[:12], ttl)
        except Exception as exc:
            logger.warning("DatasetCache.set error: %s", exc)

    async def invalidate(self, hash_val: str) -> None:
        """Explicit invalidation — used when upstream data is retroactively revised."""
        try:
            cache = self._get_cache()
            key = f"{_KEY_PREFIX}{hash_val}"
            if key in cache:
                del cache[key]
                logger.info("Cache INVALIDATED: hash=%s", hash_val[:12])
        except Exception as exc:
            logger.warning("DatasetCache.invalidate error: %s", exc)

    async def ping(self) -> bool:
        """Health check — True if cache is accessible."""
        try:
            cache = self._get_cache()
            cache.stats()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the cache."""
        if self._cache:
            self._cache.close()
            self._cache = None


# Module-level singleton used by the delivery layer
dataset_cache = DatasetCache()
