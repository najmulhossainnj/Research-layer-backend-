"""
Backtest Agent.

Responsibilities (from the spec):
  - Execute parameter sweeps across backtest configurations
  - Train the tuned model on the full dataset first
  - Run a baseline backtest and store results in ctx
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentContext, AgentResult, AgentRole, BaseAgent


class BacktestAgent(BaseAgent):
    role = AgentRole.BACKTEST

    def __init__(self, db: AsyncSession):
        self._db = db

    async def run(
        self,
        ctx: AgentContext,
        engine: str = "vectorbt",
        initial_capital: float = 100_000,
        commission: float = 0.0005,
        slippage: float = 0.0005,
        **kwargs,
    ) -> AgentResult:
        if not ctx.strategy_id or not ctx.model_id:
            return self._fail(
                "Missing strategy_id or model_id in context",
                ["Run FeatureDiscovery and ModelDiscovery agents first"],
            )

        try:
            # ── Train model on full dataset ───────────────────────────
            await self._train_model(ctx)

            # ── Create and execute backtest ───────────────────────────
            backtest_id = await self._create_and_run_backtest(
                ctx, engine, initial_capital, commission, slippage
            )

            return self._ok(
                summary=f"Backtest complete (engine={engine}). ID: {backtest_id}",
                details={"backtest_id": str(backtest_id), "engine": engine},
                ctx_updates={"backtest_ids": ctx.backtest_ids + [backtest_id]},
            )
        except Exception as exc:
            return self._fail(f"Backtest failed: {exc}", [str(exc)])

    async def _train_model(self, ctx: AgentContext) -> None:
        from app.domain.feature.orm import Feature
        from app.domain.model.orm import MLModel
        from sqlalchemy import select
        from app.engines.model_training_engine.dataset_assembler import assemble_training_data
        from app.engines.model_training_engine.trainer import ModelTrainer
        from app.engines.model_training_engine.cross_validation import CVConfig

        feat_res = await self._db.execute(
            select(Feature).where(Feature.id.in_(ctx.feature_ids))
        )
        features = list(feat_res.scalars().all())

        model_res = await self._db.execute(
            select(MLModel).where(MLModel.id == ctx.model_id)
        )
        model = model_res.scalar_one_or_none()
        if model is None:
            raise ValueError(f"MLModel {ctx.model_id} not found")

        training_data = await assemble_training_data(
            db=self._db, features=features,
            symbol=ctx.symbols[0], timeframe=ctx.timeframe,
            start_date=ctx.start_date, end_date=ctx.end_date,
        )

        trainer = ModelTrainer()
        await trainer.train(
            db=self._db, model=model,
            X=training_data.X, y=training_data.y,
            cv_config=CVConfig(n_splits=3),
            strategy_id=ctx.strategy_id,
            feature_version_hashes=[],
            symbol=ctx.symbols[0], timeframe=ctx.timeframe,
            start_date=ctx.start_date, end_date=ctx.end_date,
        )

    async def _create_and_run_backtest(
        self, ctx: AgentContext, engine: str,
        initial_capital: float, commission: float, slippage: float,
    ) -> uuid.UUID:
        from app.domain.backtest.orm import Backtest
        from app.domain.backtest.schemas import BacktestCreate
        from app.engines.backtest_engine.pipeline import BacktestPipeline

        config = {
            "symbol": ctx.symbols[0],
            "timeframe": ctx.timeframe,
            "start_date": ctx.start_date.isoformat(),
            "end_date": ctx.end_date.isoformat(),
            "feature_ids": [str(fid) for fid in ctx.feature_ids],
            "model_id": str(ctx.model_id),
            "signal_logic_id": str(ctx.signal_logic_id) if ctx.signal_logic_id else None,
        }

        bt = Backtest(
            strategy_id=ctx.strategy_id,
            engine=engine,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            config=config,
        )
        self._db.add(bt)
        await self._db.commit()
        await self._db.refresh(bt)

        pipeline = BacktestPipeline()
        updated_bt, _ = await pipeline.execute(self._db, bt)
        return updated_bt.id
