"""
Hyperparameter Agent.

Responsibilities (from the spec):
  - Run an Optuna study for the model plugin chosen by Model Discovery
  - Use the default search space from `search_spaces.py` unless the user
    supplied overrides
  - Write best_params back into ctx and update the MLModel row
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentContext, AgentResult, AgentRole, BaseAgent
from app.engines.model_training_engine.cross_validation import CVConfig
from app.engines.model_training_engine.search_spaces import DEFAULT_SEARCH_SPACES


class HyperparameterAgent(BaseAgent):
    role = AgentRole.HYPERPARAMETER

    def __init__(self, db: AsyncSession):
        self._db = db

    async def run(
        self,
        ctx: AgentContext,
        n_trials: int = 20,
        **kwargs,
    ) -> AgentResult:
        plugin_key = ctx.best_model_plugin
        if not plugin_key:
            return self._fail("No model plugin selected yet", ["Run ModelDiscoveryAgent first"])

        space = DEFAULT_SEARCH_SPACES.get(plugin_key)
        if not space:
            # No search space defined — skip tuning, keep current params
            return self._ok(
                summary=f"No search space for {plugin_key}; keeping default params.",
                ctx_updates={"tuning_results": {"skipped": True}},
            )

        if not ctx.feature_ids or not ctx.symbols:
            return self._ok(
                summary="Insufficient context for tuning; using default params.",
                ctx_updates={"tuning_results": {"skipped": True}},
            )

        try:
            from app.domain.feature.orm import Feature
            from sqlalchemy import select
            from app.engines.model_training_engine.dataset_assembler import assemble_training_data
            from app.engines.model_training_engine.tuning import tune_hyperparameters

            feat_res = await self._db.execute(
                select(Feature).where(Feature.id.in_(ctx.feature_ids))
            )
            features = list(feat_res.scalars().all())

            training_data = await assemble_training_data(
                db=self._db, features=features,
                symbol=ctx.symbols[0], timeframe=ctx.timeframe,
                start_date=ctx.start_date, end_date=ctx.end_date,
            )

            tuning = tune_hyperparameters(
                plugin_key=plugin_key,
                X=training_data.X,
                y=training_data.y,
                param_space=space,
                n_trials=n_trials,
                cv_config=CVConfig(n_splits=3, test_size=0.15, min_train_size=0.3),
            )

            # Update the MLModel row with tuned params
            if ctx.model_id:
                from app.domain.model.orm import MLModel
                from sqlalchemy import select as sa_select
                res = await self._db.execute(
                    sa_select(MLModel).where(MLModel.id == ctx.model_id)
                )
                model = res.scalar_one_or_none()
                if model:
                    model.parameters = tuning.best_params
                    self._db.add(model)
                    await self._db.commit()

            return self._ok(
                summary=(
                    f"Optuna tuning complete: {tuning.n_trials} trials for {plugin_key}. "
                    f"Best score: {tuning.best_score:.4f}"
                ),
                details={
                    "best_params": tuning.best_params,
                    "best_score": tuning.best_score,
                    "n_trials": tuning.n_trials,
                },
                ctx_updates={
                    "best_model_params": tuning.best_params,
                    "tuning_results": {
                        "best_params": tuning.best_params,
                        "best_score": tuning.best_score,
                    },
                },
            )

        except Exception as exc:
            return self._fail(f"Tuning failed: {exc}", [str(exc)])
