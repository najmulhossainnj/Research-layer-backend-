"""
Event bus backed by in-memory queue (local mode).

Events published:
  DatasetIngested       — a new dataset was written to object storage
  DatasetServedFromCache — a delivery request was fulfilled from cache
  IngestionFailed       — an ingestion pipeline run failed

In local mode, events are logged but not persisted. For production,
configure EVENT_BACKEND=redis for Redis Streams support.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from shared.config import settings

logger = logging.getLogger(__name__)

# In-memory event log for local mode
_event_log: deque[dict[str, Any]] = deque(maxlen=1000)


class EventStream:
    """Local event stream using in-memory deque."""
    
    async def publish(self, event: str, payload: dict[str, Any]) -> None:
        """Log event to in-memory queue. Non-blocking — never raises."""
        try:
            event_record = {
                "event": event,
                "payload": payload,
                "published_at": datetime.now(timezone.utc).isoformat(),
            }
            _event_log.append(event_record)
            logger.debug("Event published: %s", event)
        except Exception as exc:
            logger.warning("EventStream.publish failed: %s", exc)

    async def close(self) -> None:
        """No-op for local mode."""
        pass

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent events from the log."""
        return list(_event_log)[-limit:]


# ── Pre-built event helpers ───────────────────────────────────────────────────


async def publish_dataset_ingested(
    stream: EventStream,
    *,
    data_type: str,
    symbol: str | None,
    timeframe: str | None,
    storage_uri: str,
    hash_val: str,
    rows: int,
) -> None:
    await stream.publish(
        "DatasetIngested",
        {
            "data_type": data_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "storage_uri": storage_uri,
            "hash": hash_val,
            "rows": rows,
        },
    )


async def publish_cache_hit(
    stream: EventStream, *, hash_val: str, symbol: str | None
) -> None:
    await stream.publish(
        "DatasetServedFromCache",
        {"hash": hash_val, "symbol": symbol},
    )


async def publish_ingestion_failed(
    stream: EventStream,
    *,
    data_type: str,
    symbol: str | None,
    error: str,
) -> None:
    await stream.publish(
        "IngestionFailed",
        {"data_type": data_type, "symbol": symbol, "error": error},
    )


# ── Module-level singleton ────────────────────────────────────────────────────

event_stream = EventStream()
