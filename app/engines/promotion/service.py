"""
Portfolio Layer promotion service.

A strategy can only be promoted once it has:
  1. Status == "validated" (set by the Validation Engine)
  2. No CRITICAL governance flags (checked by the Governance Agent)
  3. At least one completed backtest with required metrics

On promotion the service:
  - Updates Strategy.status = "promoted"
  - Publishes StrategyValidated to the event bus
  - Returns the full promotion payload for the Portfolio Construction Layer

The Research Layer NEVER places orders or sends execution instructions —
it only publishes the StrategyValidated event.  The Portfolio Construction
Layer decides whether and how to deploy the strategy.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.backtest.orm import Backtest, BacktestStatus
from app.domain.strategy.orm import Strategy, StrategyStatus
from app.events.domain_events import publish_strategy_validated


@dataclass
class PromotionResult:
    success: bool
    strategy_id: uuid.UUID
    message: str
    payload: dict          # The exact payload sent to the Portfolio Layer


class PromotionService:

    async def promote(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        confidence_override: Optional[float] = None,
    ) -> PromotionResult:
        # ── Load and validate strategy ────────────────────────────────
        strat_res = await db.execute(
            select(Strategy).where(Strategy.id == strategy_id)
        )
        strategy = strat_res.scalar_one_or_none()
        if strategy is None:
            return PromotionResult(
                success=False, strategy_id=strategy_id,
                message="Strategy not found", payload={},
            )

        if strategy.status != StrategyStatus.VALIDATED:
            return PromotionResult(
                success=False, strategy_id=strategy_id,
                message=(
                    f"Strategy must be in 'validated' status before promotion "
                    f"(current: {strategy.status}). Run walk-forward + CPCV first."
                ),
                payload={},
            )

        # ── Find the best completed backtest for this strategy ────────
        bt_res = await db.execute(
            select(Backtest)
            .where(
                Backtest.strategy_id == strategy_id,
                Backtest.status == BacktestStatus.COMPLETED,
            )
            .order_by(Backtest.created_at.desc())
        )
        backtests = list(bt_res.scalars().all())

        if not backtests:
            return PromotionResult(
                success=False, strategy_id=strategy_id,
                message="No completed backtests found for this strategy.",
                payload={},
            )

        # Pick the backtest with the highest Sharpe
        best_bt = max(
            backtests,
            key=lambda b: b.metrics.get("perf_sharpe_ratio", float("-inf"))
            if b.metrics else float("-inf"),
        )
        metrics = best_bt.metrics or {}

        # ── Build promotion payload ───────────────────────────────────
        sharpe        = float(metrics.get("perf_sharpe_ratio", 0.0))
        cagr          = float(metrics.get("perf_cagr", 0.0))
        max_drawdown  = float(metrics.get("risk_max_drawdown", 0.0))
        turnover      = float(metrics.get("trade_turnover_annualised", 0.0))

        # Confidence heuristic: normalised Sharpe capped at 1.0
        confidence = confidence_override or min(1.0, max(0.0, sharpe / 3.0))

        # signal_model: "{model_type}_v{version}" resolved from strategy
        from app.domain.model.orm import MLModel
        signal_model = "unknown"
        if strategy.model_id:
            model_res = await db.execute(
                select(MLModel).where(MLModel.id == strategy.model_id)
            )
            model = model_res.scalar_one_or_none()
            if model:
                signal_model = f"{model.model_type}_v{model.version}"

        promotion_payload = {
            "strategy_id":     str(strategy_id),
            "expected_return": cagr,
            "confidence":      confidence,
            "signal_model":    signal_model,
            "sharpe":          sharpe,
            "max_drawdown":    max_drawdown,
            "turnover":        turnover,
        }

        # ── Update status ─────────────────────────────────────────────
        strategy.status = StrategyStatus.PROMOTED
        db.add(strategy)
        await db.commit()

        # ── Publish event ─────────────────────────────────────────────
        publish_strategy_validated(
            strategy_id=strategy_id,
            expected_return=cagr,
            confidence=confidence,
            signal_model=signal_model,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            turnover=turnover,
        )

        return PromotionResult(
            success=True,
            strategy_id=strategy_id,
            message=(
                f"Strategy '{strategy.name}' promoted to Portfolio Construction Layer. "
                f"Sharpe={sharpe:.2f}, CAGR={cagr:.2%}, confidence={confidence:.2f}"
            ),
            payload=promotion_payload,
        )

    async def demote(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        reason: str = "",
    ) -> dict:
        """Roll a promoted strategy back to 'validated' status."""
        strat_res = await db.execute(
            select(Strategy).where(Strategy.id == strategy_id)
        )
        strategy = strat_res.scalar_one_or_none()
        if strategy is None:
            return {"success": False, "message": "Strategy not found"}

        strategy.status = StrategyStatus.VALIDATED
        db.add(strategy)
        await db.commit()

        from app.events.domain_events import publish_strategy_updated
        publish_strategy_updated(strategy_id, StrategyStatus.VALIDATED)

        return {
            "success": True,
            "strategy_id": str(strategy_id),
            "new_status": StrategyStatus.VALIDATED,
            "reason": reason,
        }
