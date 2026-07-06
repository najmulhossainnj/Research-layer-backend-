"""
Celery application.

All long-running Research Layer jobs (model training, feature generation,
backtest runs, Optuna tuning studies) are dispatched as Celery tasks so
the FastAPI process returns immediately with a task ID the client can poll.
Workers are started separately:

    celery -A app.workers.celery_app worker --loglevel=info

The broker and result backend are Redis by default (configured in Settings).
Supports both standard Redis (redis://) and Upstash Redis (rediss:// with TLS).
"""
from celery import Celery
from kombu.utils.url import maybe_sanitize_url

from app.core.config import get_settings

settings = get_settings()


def _parse_redis_url(url: str) -> dict:
    """Parse Redis URL and return connection parameters for Celery."""
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    use_tls = url.startswith("rediss://")
    
    # For standard redis:// URLs, use as-is
    if not use_tls:
        return {"url": url}
    
    # For rediss:// (Upstash TLS), we need to configure SSL
    return {
        "url": url,
        "transport_options": {
            "ssl": True,
            "ssl_cert_reqs": "none",  # For Upstash compatibility
        }
    }


broker_conf = _parse_redis_url(settings.CELERY_BROKER_URL)
backend_conf = _parse_redis_url(settings.CELERY_RESULT_BACKEND)

celery_app = Celery(
    "research_layer",
    broker=broker_conf.get("url") or settings.CELERY_BROKER_URL,
    backend=backend_conf.get("url") or settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.training_tasks",
        "app.workers.feature_tasks",
        "app.workers.signal_tasks",
        "app.workers.backtest_tasks",
        "app.workers.sweep_tasks",
        "app.workers.validation_tasks",
    ],
)

# Apply transport options for TLS connections (Upstash)
if "transport_options" in broker_conf:
    celery_app.conf.broker_transport_options = broker_conf["transport_options"]

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # One task at a time per worker — safe for GPU/memory
    # Suppress pydantic warnings
    task_ignore_result=False,
)
