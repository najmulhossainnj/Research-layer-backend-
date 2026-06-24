"""
Event bus.

Publishes and consumes domain events over Kafka (default) or NATS,
with a noop backend for local development and testing.

Events produced by the Research Layer (from spec):
  StrategyCreated, StrategyUpdated, ModelTrained, SignalGenerated,
  BacktestCompleted, StrategyValidated

Events consumed:
  DatasetCreated, DatasetUpdated  (from Market Data Layer)

All events share the envelope schema:
  {
    "event_type": "StrategyValidated",
    "source":     "research-layer",
    "version":    "1.0",
    "timestamp":  "2025-01-01T00:00:00Z",
    "payload":    { ... }
  }
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _envelope(event_type: str, payload: dict) -> dict:
    return {
        "event_type": event_type,
        "source":     "research-layer",
        "version":    "1.0",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "payload":    payload,
    }


# ── Publisher ─────────────────────────────────────────────────────────────

class EventPublisher:
    """Publish domain events to the configured backend."""

    def publish(self, event_type: str, payload: dict) -> None:
        raise NotImplementedError


class KafkaPublisher(EventPublisher):
    def __init__(self, bootstrap_servers: str):
        from kafka import KafkaProducer
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )

    def publish(self, event_type: str, payload: dict) -> None:
        topic = f"research.{event_type.lower()}"
        envelope = _envelope(event_type, payload)
        self._producer.send(topic, value=envelope)
        self._producer.flush()
        logger.info("Published %s to %s", event_type, topic)


class NoopPublisher(EventPublisher):
    """Silent fallback for local dev / test environments."""
    def publish(self, event_type: str, payload: dict) -> None:
        logger.debug("[NOOP] Would publish %s: %s", event_type, payload)


@lru_cache(maxsize=1)
def get_publisher() -> EventPublisher:
    settings = get_settings()
    if settings.EVENT_BACKEND == "kafka":
        try:
            return KafkaPublisher(settings.KAFKA_BOOTSTRAP_SERVERS)
        except Exception as exc:
            logger.warning("Kafka unavailable (%s) — falling back to NoopPublisher", exc)
    return NoopPublisher()


# ── Consumer ──────────────────────────────────────────────────────────────

class EventConsumer:
    """Consume events from the configured backend in a background thread."""

    def start(self, topics: list[str], handler) -> None:
        raise NotImplementedError


class KafkaConsumer(EventConsumer):
    def start(self, topics: list[str], handler) -> None:
        import threading
        from kafka import KafkaConsumer as _KC

        settings = get_settings()

        def _run():
            consumer = _KC(
                *topics,
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                group_id="research-layer",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
            )
            for msg in consumer:
                try:
                    handler(msg.value)
                except Exception as exc:
                    logger.error("Error handling event %s: %s", msg.value, exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        logger.info("Kafka consumer started for topics: %s", topics)


class NoopConsumer(EventConsumer):
    def start(self, topics: list[str], handler) -> None:
        logger.debug("[NOOP] Consumer not started (noop backend)")


@lru_cache(maxsize=1)
def get_consumer() -> EventConsumer:
    settings = get_settings()
    if settings.EVENT_BACKEND == "kafka":
        try:
            return KafkaConsumer()
        except Exception as exc:
            logger.warning("Kafka consumer unavailable (%s) — noop fallback", exc)
    return NoopConsumer()
