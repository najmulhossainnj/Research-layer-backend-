"""
FeaturePipeline.

Orchestrates the "generate features" responsibility of the spec: given a
list of Feature definitions and raw market/alt data, runs each feature's
plugin, persists results through the FeatureStore (with caching and
versioning), and returns either the joined wide DataFrame or per-feature
datasets, depending on what the caller needs.

Reproducibility / historical regeneration: because storage keys are
content-hashed (see `versioning.py`), re-running the exact same pipeline
config against the same source data is a no-op (cache hit at the FeatureStore
level) and re-running against revised source data automatically produces a
new, distinct dataset version rather than silently overwriting history.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.feature.dataset_orm import FeatureDataset
from app.domain.feature.orm import Feature
from app.engines.feature_engine.store import FeatureStore
from app.engines.feature_engine.versioning import compute_source_fingerprint
from app.plugins.features import feature_registry


@dataclass
class FeatureRunResult:
    feature: Feature
    dataset: FeatureDataset
    dataframe: pd.DataFrame


class FeaturePipeline:
    def __init__(self, store: Optional[FeatureStore] = None):
        self.store = store or FeatureStore()

    async def run_one(
        self,
        db: AsyncSession,
        feature: Feature,
        market_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> FeatureRunResult:
        """Compute (or fetch a cached/stored version of) a single feature."""
        source_fp = compute_source_fingerprint(
            pd.util.hash_pandas_object(market_data).values.tobytes()
        )

        plugin_cls = feature_registry.get(feature.plugin_key)
        plugin = plugin_cls(**feature.parameters)
        result_df = plugin.compute(market_data)

        dataset = await self.store.write_dataframe(
            db=db,
            feature_id=feature.id,
            plugin_key=feature.plugin_key,
            params=feature.parameters,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            df=result_df,
            source_fingerprint=source_fp,
        )

        # If write_dataframe returned an existing (cached) dataset, prefer
        # its persisted dataframe so callers always see what's on record.
        materialized = self.store.read_dataframe(dataset)
        return FeatureRunResult(feature=feature, dataset=dataset, dataframe=materialized)

    async def run_many(
        self,
        db: AsyncSession,
        features: list[Feature],
        market_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        join: bool = True,
    ) -> pd.DataFrame | list[FeatureRunResult]:
        """Compute a set of features and optionally join them into one wide frame."""
        results: list[FeatureRunResult] = []
        for feature in features:
            result = await self.run_one(
                db, feature, market_data, symbol, timeframe, start_date, end_date
            )
            results.append(result)

        if not join:
            return results

        frames = [r.dataframe for r in results]
        if not frames:
            return pd.DataFrame()

        joined = frames[0]
        for frame in frames[1:]:
            joined = joined.join(frame, how="outer", rsuffix="_dup")
        return joined

    async def regenerate(
        self,
        db: AsyncSession,
        feature: Feature,
        market_data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> FeatureRunResult:
        """Force recomputation against current source data, creating a new
        version if the source has changed (content-hash naturally handles
        this — see module docstring)."""
        return await self.run_one(
            db, feature, market_data, symbol, timeframe, start_date, end_date
        )
