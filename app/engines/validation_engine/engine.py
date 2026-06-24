"""
Validation Engine.

Sits between the Research Layer and the Portfolio Construction Layer:
before a strategy can be promoted it must pass validation. The engine
runs walk-forward analysis (Phase 7) and will run CPCV (Phase 8) on the
assembled training dataset, logs every fold result to MLflow, records an
`Experiment` row, and — if the strategy passes the configured thresholds
— updates its status to `validated`.

Promotion gate (configurable via ValidationConfig):
  - OOS Sharpe mean > min_sharpe
  - OOS max drawdown < max_drawdown
  - n_profitable_folds / n_folds >= profitable_fold_ratio
  - overfitting_score < max_overfit_ratio
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING
import uuid

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.strategy.orm import Strategy, StrategyStatus
from app.engines.model_training_engine.dataset_assembler import assemble_training_data
from app.engines.tracking.service import ExperimentTracker
from app.engines.validation_engine.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
)


@dataclass
class ValidationConfig:
    # Walk-forward settings
    wf: WalkForwardConfig = None

    # Promotion gate thresholds
    min_sharpe: float = 0.3
    max_drawdown: float = 0.25
    profitable_fold_ratio: float = 0.5
    max_overfit_ratio: float = 3.0

    def __post_init__(self):
        if self.wf is None:
            self.wf = WalkForwardConfig()


@dataclass
class ValidationResult:
    walk_forward: WalkForwardResult
    passed: bool
    gate_results: dict        # {gate_name: {"passed": bool, "value": float, "threshold": float}}
    mlflow_run_id: Optional[str] = None
    experiment_id: Optional[uuid.UUID] = None


class ValidationEngine:

    def __init__(self):
        self._wf_engine = WalkForwardEngine()
        self._tracker = ExperimentTracker()

    async def validate_strategy(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        plugin_key: str,
        model_params: dict,
        feature_ids: list[uuid.UUID],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        target_horizon: int = 1,
        config: ValidationConfig | None = None,
        bars_per_year: int = 252,
    ) -> ValidationResult:
        config = config or ValidationConfig()

        # ── Load strategy ────────────────────────────────────────────
        strat_res = await db.execute(
            select(Strategy).where(Strategy.id == strategy_id)
        )
        strategy = strat_res.scalar_one_or_none()
        if strategy is None:
            raise ValueError(f"Strategy {strategy_id} not found")

        # ── Assemble training data ────────────────────────────────────
        from app.domain.feature.orm import Feature
        feat_res = await db.execute(
            select(Feature).where(Feature.id.in_(feature_ids))
        )
        features = list(feat_res.scalars().all())

        training_data = await assemble_training_data(
            db=db,
            features=features,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            target_horizon=target_horizon,
        )

        # ── Walk-forward analysis ─────────────────────────────────────
        wf_result = self._wf_engine.run(
            X=training_data.X,
            y=training_data.y,
            plugin_key=plugin_key,
            params=model_params,
            config=config.wf,
            bars_per_year=bars_per_year,
        )

        # ── Promotion gate ────────────────────────────────────────────
        agg = wf_result.aggregate
        n_folds = agg.get("n_folds", 1) or 1
        n_profitable = agg.get("n_profitable_folds", 0)

        gates = {
            "oos_sharpe": {
                "passed": agg.get("mean_oos_sharpe", -99) >= config.min_sharpe,
                "value": agg.get("mean_oos_sharpe", 0.0),
                "threshold": config.min_sharpe,
            },
            "oos_max_drawdown": {
                "passed": agg.get("mean_oos_max_drawdown", 99) <= config.max_drawdown,
                "value": agg.get("mean_oos_max_drawdown", 0.0),
                "threshold": config.max_drawdown,
            },
            "profitable_folds": {
                "passed": (n_profitable / n_folds) >= config.profitable_fold_ratio,
                "value": round(n_profitable / n_folds, 3),
                "threshold": config.profitable_fold_ratio,
            },
            "overfit_ratio": {
                "passed": wf_result.overfitting_score <= config.max_overfit_ratio,
                "value": wf_result.overfitting_score,
                "threshold": config.max_overfit_ratio,
            },
        }
        passed = all(g["passed"] for g in gates.values())

        # ── Update strategy status ────────────────────────────────────
        if passed:
            strategy.status = StrategyStatus.VALIDATED
        db.add(strategy)
        await db.commit()

        # ── Log to MLflow ─────────────────────────────────────────────
        agg_metrics = {k: float(v) for k, v in agg.items() if isinstance(v, (int, float))}
        agg_metrics["validation_passed"] = float(passed)
        agg_metrics["overfitting_score"] = wf_result.overfitting_score

        fold_metrics_list = [
            {**f.oos_metrics, "fold_idx": f.fold_idx}
            for f in wf_result.folds
        ]

        run_id = await self._tracker.log_validation_run(
            db=db,
            strategy_id=strategy_id,
            validation_type=config.wf.method,
            fold_metrics=fold_metrics_list,
            aggregate_metrics=agg_metrics,
            params={
                "plugin_key": plugin_key,
                "n_splits": config.wf.n_splits,
                "test_size": config.wf.test_size,
                "min_train_size": config.wf.min_train_size,
                "gap_bars": config.wf.gap_bars,
                "refit": config.wf.refit,
                **model_params,
            },
        )

        return ValidationResult(
            walk_forward=wf_result,
            passed=passed,
            gate_results=gates,
            mlflow_run_id=run_id,
        )


# ── CPCV integration ───────────────────────────────────────────────────────

@dataclass
class CPCVValidationResult:
    cpcv: "CPCVEvalResult"
    passed: bool
    gate_results: dict
    mlflow_run_id: Optional[str] = None


class CPCVValidationEngine:
    """
    Runs Combinatorial Purged Cross-Validation and applies the same
    promotion gate logic as the WF ValidationEngine.  Intended to run
    *after* walk-forward passes to provide a second, stricter leakage
    check via PBO and the deflated Sharpe.
    """

    def __init__(self):
        from app.engines.validation_engine.cpcv_engine import CPCVEngine
        self._engine = CPCVEngine()
        self._tracker = ExperimentTracker()

    async def validate(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        plugin_key: str,
        model_params: dict,
        feature_ids: list[uuid.UUID],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        target_horizon: int = 1,
        cpcv_config=None,         # CPCVConfig
        max_pbo: float = 0.6,
        min_deflated_sharpe: float = 0.1,
        min_sharpe: float = 0.2,
        bars_per_year: int = 252,
    ) -> "CPCVValidationResult":
        from app.engines.validation_engine.cpcv import CPCVConfig

        cfg = cpcv_config or CPCVConfig()
        cfg.target_horizon = target_horizon

        # Assemble data
        from app.domain.feature.orm import Feature
        from sqlalchemy import select

        feat_res = await db.execute(
            select(Feature).where(Feature.id.in_(feature_ids))
        )
        features = list(feat_res.scalars().all())
        training_data = await assemble_training_data(
            db=db, features=features, symbol=symbol,
            timeframe=timeframe, start_date=start_date,
            end_date=end_date, target_horizon=target_horizon,
        )

        # Run CPCV
        cpcv_result = self._engine.run(
            X=training_data.X,
            y=training_data.y,
            plugin_key=plugin_key,
            params=model_params,
            config=cfg,
            bars_per_year=bars_per_year,
        )

        agg = cpcv_result.aggregate
        gates = {
            "mean_oos_sharpe": {
                "passed": agg.get("mean_oos_sharpe", -99) >= min_sharpe,
                "value": agg.get("mean_oos_sharpe", 0.0),
                "threshold": min_sharpe,
            },
            "pbo": {
                "passed": cpcv_result.pbo <= max_pbo,
                "value": cpcv_result.pbo,
                "threshold": max_pbo,
            },
            "deflated_sharpe": {
                "passed": cpcv_result.deflated_sharpe >= min_deflated_sharpe,
                "value": cpcv_result.deflated_sharpe,
                "threshold": min_deflated_sharpe,
            },
        }
        passed = all(g["passed"] for g in gates.values())

        # Update strategy status if fully passed
        strat_res = await db.execute(
            select(Strategy).where(Strategy.id == strategy_id)
        )
        strategy = strat_res.scalar_one_or_none()
        if strategy and passed:
            strategy.status = StrategyStatus.VALIDATED
            db.add(strategy)
            await db.commit()

        # Log to MLflow
        agg_metrics = {k: float(v) for k, v in agg.items() if isinstance(v, (int, float))}
        agg_metrics["pbo"] = cpcv_result.pbo
        agg_metrics["deflated_sharpe"] = cpcv_result.deflated_sharpe
        agg_metrics["validation_passed"] = float(passed)

        run_id = await self._tracker.log_validation_run(
            db=db,
            strategy_id=strategy_id,
            validation_type="cpcv",
            fold_metrics=[p.__dict__ for p in cpcv_result.paths],
            aggregate_metrics=agg_metrics,
            params={
                "plugin_key": plugin_key,
                "n_splits": cfg.n_splits,
                "n_test_splits": cfg.n_test_splits,
                "embargo_pct": cfg.embargo_pct,
                "purge": cfg.purge,
                **model_params,
            },
        )

        from app.engines.validation_engine.cpcv_engine import CPCVEvalResult
        return CPCVValidationResult(
            cpcv=cpcv_result, passed=passed,
            gate_results=gates, mlflow_run_id=run_id,
        )
