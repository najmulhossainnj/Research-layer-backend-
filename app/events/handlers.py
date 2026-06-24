"""
Incoming event handlers.

Processes events consumed from the Market Data Layer:
  DatasetCreated  — new OHLCV/alt-data dataset is available; triggers
                    feature regeneration for any strategy that depends on
                    the updated symbol
  DatasetUpdated  — existing dataset was revised (corporate actions,
                    restatements); same trigger as DatasetCreated
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def handle_dataset_created(event: dict) -> None:
    payload = event.get("payload", {})
    symbol = payload.get("symbol")
    logger.info("DatasetCreated received for symbol=%s — scheduling feature regeneration", symbol)
    _schedule_feature_regeneration(symbol, payload)


def handle_dataset_updated(event: dict) -> None:
    payload = event.get("payload", {})
    symbol = payload.get("symbol")
    logger.info("DatasetUpdated received for symbol=%s — scheduling feature regeneration", symbol)
    _schedule_feature_regeneration(symbol, payload)


def _schedule_feature_regeneration(symbol: str | None, payload: dict) -> None:
    """
    Dispatch a Celery task to regenerate features for the affected symbol.
    The feature content-hash versioning (Phase 2) ensures only features
    whose source data actually changed get a new version.
    """
    if not symbol:
        return
    try:
        from app.workers.feature_tasks import generate_feature_task
        # Best-effort dispatch; worker will no-op if source data unchanged
        generate_feature_task.delay(
            feature_id=payload.get("feature_id", ""),
            symbol=symbol,
            timeframe=payload.get("timeframe", "1d"),
            start_date=payload.get("start_date", ""),
            end_date=payload.get("end_date", ""),
        )
    except Exception as exc:
        logger.error("Failed to schedule feature regeneration: %s", exc)


def dispatch(event: dict) -> None:
    """Route an incoming event to the correct handler."""
    event_type = event.get("event_type", "")
    handlers = {
        "DatasetCreated": handle_dataset_created,
        "DatasetUpdated": handle_dataset_updated,
    }
    handler = handlers.get(event_type)
    if handler:
        handler(event)
    else:
        logger.debug("Unhandled event type: %s", event_type)
