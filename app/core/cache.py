"""
DiskCache-based client wrapper for local caching.

Used by the Feature Engine as a fast, short-TTL cache in front of the
(slower, durable) Feature Store object storage — repeated requests for the
same (feature, dataset_version) within a research session avoid recompute
and avoid round-tripping to S3/MinIO.

This replaces the previous Redis-based cache with a local disk-based cache
using the DiskCache library.
"""
from functools import lru_cache
from typing import Optional

import diskcache

from app.core.config import get_settings


@lru_cache
def get_cache() -> diskcache.Cache:
    settings = get_settings()
    return diskcache.Cache(settings.CACHE_DIR, disk_pickle_protocol=4)


class FeatureCache:
    """Namespaced get/set helpers over the shared DiskCache client."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache = get_cache()
        self._ttl = ttl_seconds

    @staticmethod
    def _key(namespace: str, cache_key: str) -> str:
        return f"feature_cache:{namespace}:{cache_key}"

    def get(self, namespace: str, cache_key: str) -> Optional[bytes]:
        try:
            return self._cache.get(self._key(namespace, cache_key))
        except Exception:
            return None

    def set(self, namespace: str, cache_key: str, data: bytes) -> None:
        try:
            self._cache.set(self._key(namespace, cache_key), data, expire=self._ttl)
        except Exception:
            pass

    def invalidate(self, namespace: str, cache_key: str) -> None:
        try:
            self._cache.delete(self._key(namespace, cache_key))
        except Exception:
            pass
