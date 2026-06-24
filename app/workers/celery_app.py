"""
Celery application.

All long-running Research Layer jobs (model training, feature generation,
backtest runs, Optuna tuning studies) are dispatched as Celery tasks so
the FastAPI process returns immediately with a task ID the client can poll.
Workers are started separately:

    celery -A app.workers.celery_app worker --loglevel=info

The broker and result backend are Redis by default (configured in Settings).
"""
from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "research_layer",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.training_tasks",
        "app.workers.feature_tasks",
        "app.workers.signal_tasks",
        "app.workers.backtest_tasks",
        "app.workers.sweep_tasks",
        "app.workers.validation_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # One task at a time per worker — safe for GPU/memory
)
