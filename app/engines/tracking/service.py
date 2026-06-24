"""
Experiment tracking service.

Called by the Model Trainer, Hyperparameter Tuner, and Backtest Pipeline
to record every run in MLflow with full lineage. Also creates/updates the
platform's own `Experiment` row so the Experiment Tracker UI can query
without hitting MLflow directly.

Versioning convention
---------------------
  dataset_version  : SHA-256 of (symbol, timeframe, start, end) — stable identifier
  feature_version  : comma-joined sorted version_hashes of all FeatureDataset rows used
  model_version    : str(MLModel.version)
  strategy_version : str(Strategy.version)
"""
import hashlib
import io
import json
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.mlflow_client import (
    log_artifact_bytes,
    log_metrics,
    log_params,
    mlflow_run,
)
from app.domain.model.experiment_orm import Experiment
from app.engines.backtest_engine.metrics import BacktestMetrics
from app.engines.backtest_engine.result import BacktestResult
from app.engines.model_training_engine.evaluation import CVResult


def _dataset_version(symbol: str, timeframe: str, start: datetime, end: datetime) -> str:
    raw = f"{symbol}:{timeframe}:{start.isoformat()}:{end.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _feature_version(version_hashes: list[str]) -> str:
    combined = ",".join(sorted(version_hashes))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class ExperimentTracker:

    # ── Model training ─────────────────────────────────────────────────

    async def log_training_run(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        model,                    # MLModel ORM instance
        cv_result: CVResult,
        feature_version_hashes: list[str],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        artifact_bytes: Optional[bytes] = None,
    ) -> str:
        """Log a model training run to MLflow. Returns the mlflow run_id."""
        ds_ver = _dataset_version(symbol, timeframe, start_date, end_date)
        feat_ver = _feature_version(feature_version_hashes) if feature_version_hashes else "none"
        model_ver = str(model.version)

        with mlflow_run(
            run_name=f"train:{model.name}",
            run_type="training",
            strategy_id=str(strategy_id),
            dataset_version=ds_ver,
            feature_version=feat_ver,
            model_version=model_ver,
        ) as run:
            # Hyperparameters
            log_params({
                "model_type": model.model_type,
                "model_family": model.family,
                **model.parameters,
            })

            # CV metrics
            log_metrics({
                "cv_mean_mse": cv_result.mean_mse,
                "cv_mean_mae": cv_result.mean_mae,
                "cv_directional_accuracy": cv_result.mean_directional_accuracy,
                "cv_n_folds": len(cv_result.folds),
            })

            # Per-fold metrics as steps
            for fold in cv_result.folds:
                log_metrics(
                    {"fold_mse": fold.mse, "fold_mae": fold.mae,
                     "fold_dir_acc": fold.directional_accuracy},
                    step=fold.fold,
                )

            # Model artifact
            if artifact_bytes:
                log_artifact_bytes(artifact_bytes, "model", f"{model.model_type}.model")

            run_id = run.info.run_id

        # Persist to the platform Experiment table
        await self._upsert_experiment(
            db=db,
            strategy_id=strategy_id,
            mlflow_run_id=run_id,
            dataset_version=ds_ver,
            feature_version=feat_ver,
            model_version=model_ver,
            parameters=model.parameters,
            metrics={
                "cv_mean_mse": cv_result.mean_mse,
                "cv_mean_mae": cv_result.mean_mae,
                "cv_directional_accuracy": cv_result.mean_directional_accuracy,
            },
        )

        # Write run_id back to the model row
        model.mlflow_run_id = run_id
        db.add(model)
        await db.commit()

        return run_id

    # ── Hyperparameter tuning ──────────────────────────────────────────

    async def log_tuning_run(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        plugin_key: str,
        best_params: dict,
        best_score: float,
        study_summary: list[dict],
        metric: str,
        feature_version_hashes: list[str],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> str:
        ds_ver = _dataset_version(symbol, timeframe, start_date, end_date)
        feat_ver = _feature_version(feature_version_hashes) if feature_version_hashes else "none"

        with mlflow_run(
            run_name=f"tune:{plugin_key}",
            run_type="tuning",
            strategy_id=str(strategy_id),
            dataset_version=ds_ver,
            feature_version=feat_ver,
        ) as run:
            log_params({"plugin_key": plugin_key, "optimize_metric": metric, **best_params})
            log_metrics({"best_score": best_score, "n_trials": len(study_summary)})

            # Log full study as an artifact for analysis
            study_bytes = json.dumps(study_summary, default=str).encode()
            log_artifact_bytes(study_bytes, "tuning", "study_summary.json")

            run_id = run.info.run_id

        await self._upsert_experiment(
            db=db,
            strategy_id=strategy_id,
            mlflow_run_id=run_id,
            dataset_version=ds_ver,
            feature_version=feat_ver,
            parameters={"plugin_key": plugin_key, **best_params},
            metrics={"best_score": best_score},
        )
        return run_id

    # ── Backtest ───────────────────────────────────────────────────────

    async def log_backtest_run(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        backtest,            # Backtest ORM instance
        result: BacktestResult,
        feature_version_hashes: list[str],
        model_version: Optional[str] = None,
    ) -> str:
        ds_ver = _dataset_version(
            backtest.config.get("symbol", ""),
            backtest.config.get("timeframe", "1d"),
            datetime.fromisoformat(backtest.config.get("start_date", "2000-01-01")),
            datetime.fromisoformat(backtest.config.get("end_date", "2000-01-02")),
        )
        feat_ver = _feature_version(feature_version_hashes) if feature_version_hashes else "none"

        with mlflow_run(
            run_name=f"backtest:{backtest.engine}",
            run_type="backtest",
            strategy_id=str(strategy_id),
            dataset_version=ds_ver,
            feature_version=feat_ver,
            model_version=model_version or "none",
        ) as run:
            log_params({
                "engine": backtest.engine,
                "initial_capital": str(backtest.initial_capital),
                "commission": str(backtest.commission),
                "slippage": str(backtest.slippage),
            })

            flat = result.metrics.to_flat_dict()
            numeric_metrics = {k: v for k, v in flat.items()
                               if isinstance(v, (int, float)) and k != "engine_stats"}
            log_metrics(numeric_metrics)

            # Equity curve artifact
            eq_buf = io.BytesIO()
            result.equity_curve.to_frame("equity").to_parquet(eq_buf, index=True)
            log_artifact_bytes(eq_buf.getvalue(), "backtest", "equity_curve.parquet")

            # Trades artifact
            tr_buf = io.BytesIO()
            result.trades.to_parquet(tr_buf, index=False)
            log_artifact_bytes(tr_buf.getvalue(), "backtest", "trades.parquet")

            run_id = run.info.run_id

        await self._upsert_experiment(
            db=db,
            strategy_id=strategy_id,
            mlflow_run_id=run_id,
            dataset_version=ds_ver,
            feature_version=feat_ver,
            model_version=model_version,
            parameters={"engine": backtest.engine},
            metrics=numeric_metrics,
            artifacts={
                "equity_curve_uri": backtest.equity_curve_uri or "",
                "trades_uri": backtest.trades_uri or "",
            },
        )
        return run_id

    # ── Walk-forward / validation ──────────────────────────────────────

    async def log_validation_run(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        validation_type: str,
        fold_metrics: list[dict],
        aggregate_metrics: dict,
        params: dict,
    ) -> str:
        with mlflow_run(
            run_name=f"validate:{validation_type}",
            run_type="validation",
            strategy_id=str(strategy_id),
        ) as run:
            log_params({"validation_type": validation_type, **params})
            log_metrics(aggregate_metrics)

            for i, fm in enumerate(fold_metrics):
                numeric = {k: v for k, v in fm.items() if isinstance(v, (int, float))}
                log_metrics(numeric, step=i)

            run_id = run.info.run_id

        await self._upsert_experiment(
            db=db,
            strategy_id=strategy_id,
            mlflow_run_id=run_id,
            parameters={"validation_type": validation_type, **params},
            metrics=aggregate_metrics,
        )
        return run_id

    # ── Internal ───────────────────────────────────────────────────────

    async def _upsert_experiment(
        self,
        db: AsyncSession,
        strategy_id: uuid.UUID,
        mlflow_run_id: str,
        dataset_version: Optional[str] = None,
        feature_version: Optional[str] = None,
        model_version: Optional[str] = None,
        parameters: Optional[dict] = None,
        metrics: Optional[dict] = None,
        artifacts: Optional[dict] = None,
    ) -> Experiment:
        exp = Experiment(
            strategy_id=strategy_id,
            mlflow_run_id=mlflow_run_id,
            dataset_version=dataset_version,
            feature_version=feature_version,
            model_version=model_version,
            parameters=parameters or {},
            metrics=metrics or {},
            artifacts=artifacts or {},
        )
        db.add(exp)
        await db.commit()
        await db.refresh(exp)
        return exp
