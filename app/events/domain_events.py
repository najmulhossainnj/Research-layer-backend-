"""
Domain event definitions.

Each function publishes one typed event and is the single place where the
payload schema is defined.  Callers import only the function they need;
none of the Kafka/NATS plumbing leaks out.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from app.events.bus import get_publisher


def publish_strategy_created(strategy_id: uuid.UUID, name: str, universe: list[str]) -> None:
    get_publisher().publish("StrategyCreated", {
        "strategy_id": str(strategy_id),
        "name":        name,
        "universe":    universe,
    })


def publish_strategy_updated(strategy_id: uuid.UUID, status: str) -> None:
    get_publisher().publish("StrategyUpdated", {
        "strategy_id": str(strategy_id),
        "status":      status,
    })


def publish_model_trained(
    model_id: uuid.UUID,
    strategy_id: uuid.UUID,
    model_type: str,
    metrics: dict,
) -> None:
    get_publisher().publish("ModelTrained", {
        "model_id":    str(model_id),
        "strategy_id": str(strategy_id),
        "model_type":  model_type,
        "metrics":     metrics,
    })


def publish_signal_generated(
    strategy_id: uuid.UUID,
    symbol: str,
    signal_count: int,
    buy_count: int,
    sell_count: int,
) -> None:
    get_publisher().publish("SignalGenerated", {
        "strategy_id":  str(strategy_id),
        "symbol":       symbol,
        "signal_count": signal_count,
        "buy_count":    buy_count,
        "sell_count":   sell_count,
    })


def publish_backtest_completed(
    backtest_id: uuid.UUID,
    strategy_id: uuid.UUID,
    engine: str,
    metrics: dict,
) -> None:
    get_publisher().publish("BacktestCompleted", {
        "backtest_id":  str(backtest_id),
        "strategy_id":  str(strategy_id),
        "engine":       engine,
        "metrics":      metrics,
    })


def publish_strategy_validated(
    strategy_id: uuid.UUID,
    expected_return: float,
    confidence: float,
    signal_model: str,
    sharpe: float,
    max_drawdown: float,
    turnover: float,
) -> None:
    """
    Published when a strategy passes all validation gates and is ready for
    promotion to the Portfolio Construction Layer.

    Payload matches the spec interface contract:
      {
        "strategy_id":     "uuid",
        "expected_return": 0.02,
        "confidence":      0.85,
        "signal_model":    "ml.xgboost_v5",
        ...risk metrics for the Risk Layer...
      }
    """
    get_publisher().publish("StrategyValidated", {
        "strategy_id":     str(strategy_id),
        "expected_return": expected_return,
        "confidence":      confidence,
        "signal_model":    signal_model,
        "sharpe":          sharpe,
        "max_drawdown":    max_drawdown,
        "turnover":        turnover,
    })
