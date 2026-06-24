"""
BacktestPipeline.

Orchestrates the complete backtest execution chain:

  1. Load strategy → feature definitions + model + signal logic
  2. Fetch OHLCV from Market Data Layer
  3. Compute features via FeaturePipeline
  4. Generate signals via SignalPipeline
  5. Run the chosen backtest engine
  6. Persist results (equity curve, trades, metrics) via result storage
  7. Update the Backtest row status and return the result

Callers (API handler or Celery task) only interact with this class;
neither knows which engine was used.
"""
import uuid

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.backtest.orm import Backtest, BacktestStatus
from app.domain.feature.orm import Feature
from app.domain.model.orm import MLModel
from app.domain.signal.orm import SignalLogic
from app.domain.strategy.orm import Strategy
from app.engines.backtest_engine.engine_registry import get_engine
from app.engines.backtest_engine.result import BacktestResult
from app.engines.backtest_engine.storage import persist_backtest_result
from app.engines.feature_engine.market_data_client import get_market_data_client
from app.engines.feature_engine.pipeline import FeaturePipeline
from app.engines.signal_engine.pipeline import SignalPipeline
from app.engines.tracking.service import ExperimentTracker


class BacktestPipeline:

    async def execute(
        self,
        db: AsyncSession,
        backtest: Backtest,
    ) -> tuple[Backtest, BacktestResult]:
        """
        Run a full backtest from the persisted `Backtest` config row.

        The `Backtest` row must already exist (created via the CRUD endpoint)
        with at least `strategy_id`, `engine`, `initial_capital`,
        `commission`, `slippage`, and optionally fields from `config` like
        `start_date`, `end_date`, `symbol`, `timeframe`, `feature_ids`,
        `signal_logic_id`, and `plugin_key`.
        """
        # Mark as running so the UI can show progress
        backtest.status = BacktestStatus.RUNNING
        db.add(backtest)
        await db.commit()

        try:
            result = await self._run(db, backtest)
            backtest = await persist_backtest_result(db, backtest, result)

            # Log to MLflow
            tracker = ExperimentTracker()
            await tracker.log_backtest_run(
                db=db,
                strategy_id=backtest.strategy_id,
                backtest=backtest,
                result=result,
                feature_version_hashes=[],   # populated by caller when known
            )

            return backtest, result

        except Exception as exc:
            backtest.status = BacktestStatus.FAILED
            backtest.metrics = {"error": str(exc)}
            db.add(backtest)
            await db.commit()
            raise

    async def _run(self, db: AsyncSession, backtest: Backtest) -> BacktestResult:
        cfg = backtest.config or {}

        # ── Resolve symbol / date range / timeframe ───────────────────
        symbol = cfg.get("symbol")
        timeframe = cfg.get("timeframe", "1d")
        start_date = pd.Timestamp(cfg["start_date"])
        end_date = pd.Timestamp(cfg["end_date"])

        if not symbol:
            # Fall back to the first ticker in the strategy universe
            strategy_result = await db.execute(
                select(Strategy).where(Strategy.id == backtest.strategy_id)
            )
            strategy = strategy_result.scalar_one_or_none()
            if strategy is None or not strategy.universe:
                raise ValueError("Cannot determine symbol: set config.symbol or strategy.universe")
            symbol = strategy.universe[0]

        # ── Fetch OHLCV ───────────────────────────────────────────────
        market_data_client = get_market_data_client()
        ohlcv = await market_data_client.get_ohlcv(symbol, timeframe, start_date, end_date)
        if ohlcv.empty:
            raise ValueError(f"Market Data Layer returned no OHLCV for {symbol} {start_date}–{end_date}")

        # ── Resolve features ──────────────────────────────────────────
        feature_ids = [uuid.UUID(fid) for fid in cfg.get("feature_ids", [])]
        if not feature_ids:
            # Pull from the linked strategy
            strat_res = await db.execute(
                select(Strategy).where(Strategy.id == backtest.strategy_id)
            )
            strat = strat_res.scalar_one_or_none()
            feature_ids = list(strat.feature_ids) if strat else []

        features: list[Feature] = []
        if feature_ids:
            feat_res = await db.execute(
                select(Feature).where(Feature.id.in_(feature_ids))
            )
            features = list(feat_res.scalars().all())

        # ── Compute features ──────────────────────────────────────────
        if features:
            feature_pipeline = FeaturePipeline()
            feature_df = await feature_pipeline.run_many(
                db=db,
                features=features,
                market_data=ohlcv,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                join=True,
            )
        else:
            feature_df = ohlcv.copy()

        # ── Resolve model ─────────────────────────────────────────────
        model_id = cfg.get("model_id") or (
            str(backtest.strategy_id)
            if not cfg.get("model_id")
            else cfg["model_id"]
        )
        model = None
        if cfg.get("model_id"):
            model_res = await db.execute(
                select(MLModel).where(MLModel.id == uuid.UUID(cfg["model_id"]))
            )
            model = model_res.scalar_one_or_none()

        if model is None:
            # No model → pass through raw features as predictions (e.g. rule-only strategies)
            strat_res = await db.execute(
                select(Strategy).where(Strategy.id == backtest.strategy_id)
            )
            strat = strat_res.scalar_one_or_none()
            if strat and strat.model_id:
                model_res = await db.execute(
                    select(MLModel).where(MLModel.id == strat.model_id)
                )
                model = model_res.scalar_one_or_none()

        # ── Generate signals ──────────────────────────────────────────
        signal_pipeline = SignalPipeline()
        signal_logic_id = cfg.get("signal_logic_id")
        plugin_key = cfg.get("signal_plugin_key")
        plugin_params = cfg.get("signal_plugin_params", {})

        if model is None:
            # No model at all — generate a flat BUY signal (benchmark / rule-only mode)
            signals = pd.Series("BUY", index=feature_df.index, name="signal")
        elif signal_logic_id:
            signal_result = await signal_pipeline.generate_from_rule_tree(
                db=db,
                signal_logic_id=uuid.UUID(signal_logic_id),
                model=model,
                feature_data=feature_df,
            )
            signals = signal_result.signals
        elif plugin_key:
            signal_result = await signal_pipeline.generate_from_plugin(
                model=model,
                feature_data=feature_df,
                plugin_key=plugin_key,
                plugin_params=plugin_params,
            )
            signals = signal_result.signals
        else:
            # Default: threshold plugin with sensible defaults
            signal_result = await signal_pipeline.generate_from_plugin(
                model=model,
                feature_data=feature_df,
                plugin_key="signal.threshold",
                plugin_params={"buy_threshold": 0.0, "sell_threshold": -0.5},
            )
            signals = signal_result.signals

        # ── Build engine config ───────────────────────────────────────
        engine_config = {
            "initial_capital": float(backtest.initial_capital or cfg.get("initial_capital", 100_000)),
            "commission": float(backtest.commission or cfg.get("commission", 0.0005)),
            "slippage": float(backtest.slippage or cfg.get("slippage", 0.0005)),
            "bars_per_year": cfg.get("bars_per_year", 252),
            "risk_free_rate": cfg.get("risk_free_rate", 0.0),
            "size_type": cfg.get("size_type", "percent"),
        }

        # ── Run engine ────────────────────────────────────────────────
        engine = get_engine(backtest.engine, engine_config)
        result = engine.run(ohlcv, signals)
        return result
