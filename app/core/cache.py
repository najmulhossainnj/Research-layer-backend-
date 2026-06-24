"""
Redis client wrapper.

Used by the Feature Engine as a fast, short-TTL cache in front of the
(slower, durable) Feature Store object storage — repeated requests for the
same (feature, dataset_version) within a research session avoid recompute
and avoid round-tripping to S3/MinIO.
"""
from functools import lru_cache
from typing import Optional

import redis

from app.core.config import get_settings


@lru_cache
def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.REDIS_URL, decode_responses=False)


class FeatureCache:
    """Namespaced get/set helpers over the shared Redis client."""

    def __init__(self, ttl_seconds: int = 3600):
        self._client = get_redis_client()
        self._ttl = ttl_seconds

    @staticmethod
    def _key(namespace: str, cache_key: str) -> str:
        return f"feature_cache:{namespace}:{cache_key}"

    def get(self, namespace: str, cache_key: str) -> Optional[bytes]:
        return self._client.get(self._key(namespace, cache_key))

    def set(self, namespace: str, cache_key: str, data: bytes) -> None:
        self._client.set(self._key(namespace, cache_key), data, ex=self._ttl)

    def invalidate(self, namespace: str, cache_key: str) -> None:
        self._client.delete(self._key(namespace, cache_key))
