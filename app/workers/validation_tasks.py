"""
Async walk-forward validation task.

Walk-forward analysis on large datasets (5+ year, daily, 10 folds,
refit=True) takes minutes — long enough to warrant Celery dispatch.
"""
import asyncio
import uuid
from datetime import datetime

from app.workers.celery_app import celery_app


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="validation.walk_forward")
def walk_forward_task(self, payload: dict):
    async def _inner():
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.core.config import get_settings
        from app.engines.validation_engine.engine import ValidationConfig, ValidationEngine
        from app.engines.validation_engine.walk_forward import WalkForwardConfig

        settings = get_settings()
        engine = create_async_engine(settings.DATABASE_URL)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        wf_cfg_data = payload.get("config", {}).get("wf", {})
        val_cfg_data = payload.get("config", {})

        wf_cfg = WalkForwardConfig(
            method=wf_cfg_data.get("method", "rolling"),
            n_splits=wf_cfg_data.get("n_splits", 5),
            test_size=wf_cfg_data.get("test_size", 0.2),
            min_train_size=wf_cfg_data.get("min_train_size", 0.3),
            gap_bars=wf_cfg_data.get("gap_bars", 0),
            refit=wf_cfg_data.get("refit", True),
        )
        val_cfg = ValidationConfig(
            wf=wf_cfg,
            min_sharpe=val_cfg_data.get("min_sharpe", 0.3),
            max_drawdown=val_cfg_data.get("max_drawdown", 0.25),
            profitable_fold_ratio=val_cfg_data.get("profitable_fold_ratio", 0.5),
            max_overfit_ratio=val_cfg_data.get("max_overfit_ratio", 3.0),
        )

        async with Session() as db:
            ve = ValidationEngine()
            result = await ve.validate_strategy(
                db=db,
                strategy_id=uuid.UUID(payload["strategy_id"]),
                plugin_key=payload["plugin_key"],
                model_params=payload.get("model_params", {}),
                feature_ids=[uuid.UUID(fid) for fid in payload["feature_ids"]],
                symbol=payload["symbol"],
                timeframe=payload.get("timeframe", "1d"),
                start_date=datetime.fromisoformat(payload["start_date"]),
                end_date=datetime.fromisoformat(payload["end_date"]),
                target_horizon=payload.get("target_horizon", 1),
                config=val_cfg,
                bars_per_year=payload.get("bars_per_year", 252),
            )

        return {
            "strategy_id": payload["strategy_id"],
            "passed": result.passed,
            "gate_results": result.gate_results,
            "aggregate": result.walk_forward.aggregate,
            "overfitting_score": result.walk_forward.overfitting_score,
            "mlflow_run_id": result.mlflow_run_id,
        }

    self.update_state(
        state="STARTED",
        meta={"strategy_id": payload.get("strategy_id"), "symbol": payload.get("symbol")},
    )
    return _run_async(_inner())


@celery_app.task(bind=True, name="validation.cpcv")
def cpcv_task(self, payload: dict):
    """Async CPCV validation."""
    async def _inner():
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        from app.core.config import get_settings
        from app.engines.validation_engine.cpcv import CPCVConfig
        from app.engines.validation_engine.engine import CPCVValidationEngine

        settings = get_settings()
        engine = create_async_engine(settings.DATABASE_URL)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        cpcv_data = payload.get("cpcv", {})
        cfg = CPCVConfig(
            n_splits=cpcv_data.get("n_splits", 6),
            n_test_splits=cpcv_data.get("n_test_splits", 2),
            embargo_pct=cpcv_data.get("embargo_pct", 0.01),
            purge=cpcv_data.get("purge", True),
            target_horizon=payload.get("target_horizon", 1),
        )

        async with Session() as db:
            ve = CPCVValidationEngine()
            result = await ve.validate(
                db=db,
                strategy_id=uuid.UUID(payload["strategy_id"]),
                plugin_key=payload["plugin_key"],
                model_params=payload.get("model_params", {}),
                feature_ids=[uuid.UUID(f) for f in payload["feature_ids"]],
                symbol=payload["symbol"],
                timeframe=payload.get("timeframe", "1d"),
                start_date=datetime.fromisoformat(payload["start_date"]),
                end_date=datetime.fromisoformat(payload["end_date"]),
                target_horizon=payload.get("target_horizon", 1),
                cpcv_config=cfg,
                max_pbo=payload.get("max_pbo", 0.6),
                min_deflated_sharpe=payload.get("min_deflated_sharpe", 0.1),
                min_sharpe=payload.get("min_sharpe", 0.2),
                bars_per_year=payload.get("bars_per_year", 252),
            )
        return {
            "strategy_id": payload["strategy_id"],
            "passed": result.passed,
            "pbo": result.cpcv.pbo,
            "deflated_sharpe": result.cpcv.deflated_sharpe,
            "gate_results": result.gate_results,
            "mlflow_run_id": result.mlflow_run_id,
        }

    self.update_state(state="STARTED", meta={"strategy_id": payload.get("strategy_id")})
    return _run_async(_inner())
