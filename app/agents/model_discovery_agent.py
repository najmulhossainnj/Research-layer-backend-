"""
Model Discovery Agent.

Responsibilities (from the spec):
  - Compare model families (statistical, ML, DL) via AutoML
  - Parse user query for preferred model type hints
  - Select the best plugin_key and baseline params
  - Persist the winning model as an `MLModel` row
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentContext, AgentResult, AgentRole, BaseAgent
from app.plugins.models import model_registry


_QUERY_HINTS: dict[str, str] = {
    "xgboost":        "ml.xgboost",
    "xgb":            "ml.xgboost",
    "lightgbm":       "ml.lightgbm",
    "lgbm":           "ml.lightgbm",
    "catboost":       "ml.catboost",
    "random forest":  "ml.random_forest",
    "rf":             "ml.random_forest",
    "lstm":           "dl.lstm",
    "deep":           "dl.lstm",
    "neural":         "dl.lstm",
}

_DEFAULT_CANDIDATES: dict[str, dict] = {
    "ml.xgboost":      {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 200},
    "ml.lightgbm":     {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200},
    "ml.random_forest": {"n_estimators": 200, "max_depth": 8},
}


class ModelDiscoveryAgent(BaseAgent):
    role = AgentRole.MODEL_DISCOVERY

    def __init__(self, db: AsyncSession):
        self._db = db

    async def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        query = ctx.user_query.lower()

        # Check for explicit model hint in user query
        forced_key: str | None = None
        for hint, key in _QUERY_HINTS.items():
            if hint in query:
                forced_key = key
                break

        # Build candidate set
        if forced_key:
            candidates = {forced_key: _DEFAULT_CANDIDATES.get(forced_key, {})}
        else:
            candidates = _DEFAULT_CANDIDATES.copy()

        # ── Run AutoML if we have assembled data ──────────────────────
        leaderboard: list[dict] = []
        best_plugin = forced_key or "ml.xgboost"
        best_params = candidates.get(best_plugin, {})

        if ctx.feature_ids and ctx.symbols and ctx.start_date and ctx.end_date:
            try:
                leaderboard, best_plugin, best_params = await self._run_automl(
                    ctx, candidates
                )
            except Exception as exc:
                return self._fail(
                    f"AutoML failed: {exc}", [str(exc)]
                )
        else:
            leaderboard = [
                {"plugin_key": k, "params": v, "score": None}
                for k, v in candidates.items()
            ]

        # ── Persist winning MLModel row ────────────────────────────────
        from app.domain.model.orm import MLModel, ModelFamily

        family_map = {
            "ml.": ModelFamily.MACHINE_LEARNING,
            "dl.": ModelFamily.DEEP_LEARNING,
            "stat.": ModelFamily.STATISTICAL,
        }
        family = next(
            (v for k, v in family_map.items() if best_plugin.startswith(k)),
            ModelFamily.MACHINE_LEARNING,
        )

        model = MLModel(
            name=f"agent:{best_plugin}",
            model_type=best_plugin,
            family=family,
            parameters=best_params,
        )
        self._db.add(model)
        await self._db.commit()
        await self._db.refresh(model)

        summary = (
            f"Selected {best_plugin} as the best model "
            f"(tested {len(leaderboard)} candidates). "
            f"MLModel ID: {model.id}"
        )
        return self._ok(
            summary=summary,
            details={"leaderboard": leaderboard, "selected": best_plugin},
            ctx_updates={
                "model_id": model.id,
                "best_model_plugin": best_plugin,
                "best_model_params": best_params,
                "candidate_models": leaderboard,
            },
        )

    async def _run_automl(
        self, ctx: AgentContext, candidates: dict
    ) -> tuple[list[dict], str, dict]:
        from app.domain.feature.orm import Feature
        from sqlalchemy import select
        from app.engines.model_training_engine.dataset_assembler import assemble_training_data
        from app.engines.model_training_engine.automl import run_automl
        from app.engines.model_training_engine.cross_validation import CVConfig

        feat_res = await self._db.execute(
            select(Feature).where(Feature.id.in_(ctx.feature_ids))
        )
        features = list(feat_res.scalars().all())

        training_data = await assemble_training_data(
            db=self._db, features=features,
            symbol=ctx.symbols[0], timeframe=ctx.timeframe,
            start_date=ctx.start_date, end_date=ctx.end_date,
        )

        result = run_automl(
            X=training_data.X, y=training_data.y,
            candidates=candidates,
            cv_config=CVConfig(n_splits=3, test_size=0.15, min_train_size=0.3),
        )

        leaderboard = [
            {"plugin_key": c.plugin_key, "params": c.params,
             "score": c.score, "metrics": c.metrics}
            for c in result.leaderboard
        ]
        best = result.best
        return leaderboard, best.plugin_key, best.params
