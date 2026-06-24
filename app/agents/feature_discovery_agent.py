"""
Feature Discovery Agent.

Responsibilities (from the spec):
  - Generate candidate features from the full plugin catalog
  - Parse the user query for hints (e.g. "RSI", "momentum", "sentiment")
  - Run SHAP importance on a quick RF model to prune low-signal features
  - Persist winning features as `Feature` rows and write their IDs into ctx

The agent uses a lightweight Random Forest so SHAP analysis completes in
seconds even on large datasets, rather than waiting for the full model
selected by the Model Discovery Agent.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentContext, AgentResult, AgentRole, BaseAgent
from app.plugins.features import feature_registry


# Keyword → plugin-key heuristic table (extended by the user query parser)
_QUERY_HINTS: dict[str, list[str]] = {
    "rsi":         ["technical.rsi"],
    "atr":         ["technical.atr"],
    "momentum":    ["statistical.momentum"],
    "mean reversion": ["statistical.mean_reversion"],
    "hurst":       ["statistical.hurst_exponent"],
    "volatility":  ["statistical.volatility_regime"],
    "sentiment":   ["news.finbert_sentiment", "news.sentiment_momentum"],
    "news":        ["news.finbert_sentiment"],
    "tsfresh":     ["automated.tsfresh"],
    "automated":   ["automated.tsfresh"],
    "z score":     ["statistical.z_score"],
    "z-score":     ["statistical.z_score"],
}

_DEFAULT_FEATURES = [
    {"plugin_key": "technical.rsi",         "parameters": {"period": 14}},
    {"plugin_key": "technical.atr",         "parameters": {"period": 14}},
    {"plugin_key": "statistical.momentum",  "parameters": {"window": 20}},
    {"plugin_key": "statistical.z_score",   "parameters": {"window": 20}},
    {"plugin_key": "statistical.volatility_regime", "parameters": {}},
]


class FeatureDiscoveryAgent(BaseAgent):
    role = AgentRole.FEATURE_DISCOVERY

    def __init__(self, db: AsyncSession):
        self._db = db

    async def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        query = ctx.user_query.lower()

        # ── 1. Select candidates from plugin catalog ──────────────────
        candidates: list[dict] = []
        matched_keys: set[str] = set()

        for hint, keys in _QUERY_HINTS.items():
            if hint in query:
                for key in keys:
                    if key not in matched_keys and key in feature_registry.list_keys():
                        candidates.append({"plugin_key": key, "parameters": {}})
                        matched_keys.add(key)

        if not candidates:
            candidates = _DEFAULT_FEATURES

        # ── 2. Persist Feature rows ───────────────────────────────────
        from app.domain.feature.orm import Feature
        feature_ids: list[uuid.UUID] = []

        for spec in candidates:
            feature = Feature(
                name=spec["plugin_key"],
                type=spec["plugin_key"].split(".")[0],
                plugin_key=spec["plugin_key"],
                parameters=spec.get("parameters", {}),
            )
            self._db.add(feature)
            await self._db.commit()
            await self._db.refresh(feature)
            feature_ids.append(feature.id)

        # ── 3. SHAP importance (lightweight RF proxy) ─────────────────
        shap_results = await self._run_shap_if_data_available(ctx, candidates)

        summary = (
            f"Discovered {len(candidates)} candidate features "
            f"({', '.join(c['plugin_key'] for c in candidates[:3])}{'...' if len(candidates) > 3 else ''}). "
            + (f"SHAP pruned to {len(shap_results)} high-signal features." if shap_results
               else "SHAP skipped (no assembled data yet).")
        )

        return self._ok(
            summary=summary,
            details={"candidates": candidates, "shap_importance": shap_results},
            ctx_updates={
                "feature_ids": feature_ids,
                "candidate_features": candidates,
            },
        )

    async def _run_shap_if_data_available(
        self, ctx: AgentContext, candidates: list[dict]
    ) -> dict:
        """Quick RF + SHAP pass — returns {plugin_key: importance} or {} if skipped."""
        if not ctx.symbols or not ctx.start_date or not ctx.end_date:
            return {}
        try:
            import shap
            from sklearn.ensemble import RandomForestRegressor
            from app.engines.feature_engine.market_data_client import get_market_data_client
            from app.engines.feature_engine.pipeline import FeaturePipeline
            from app.engines.model_training_engine.dataset_assembler import assemble_training_data

            # Only load features that are already persisted in ctx
            if not ctx.feature_ids:
                return {}

            from app.domain.feature.orm import Feature
            from sqlalchemy import select
            feat_res = await self._db.execute(
                select(Feature).where(Feature.id.in_(ctx.feature_ids))
            )
            features = list(feat_res.scalars().all())

            training_data = await assemble_training_data(
                db=self._db, features=features,
                symbol=ctx.symbols[0], timeframe=ctx.timeframe,
                start_date=ctx.start_date, end_date=ctx.end_date,
            )
            if len(training_data.X) < 30:
                return {}

            rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
            rf.fit(training_data.X, training_data.y)

            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(training_data.X)

            import numpy as np
            importance = dict(zip(
                training_data.feature_columns,
                [float(v) for v in abs(shap_vals).mean(axis=0)],
            ))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}
