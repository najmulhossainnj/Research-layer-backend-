"""
Feature Store.

Persists generated feature datasets to object storage as parquet, tracks
metadata/lineage in Postgres (`FeatureDataset`), and fronts reads with a
short-TTL Redis cache. This is the component the spec calls out as needing
to "generate features / cache features / version features / store
metadata" with support for "historical regeneration" and "reproducibility".
"""
import io
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import FeatureCache
from app.core.config import get_settings
from app.core.storage import get_storage_client
from app.domain.feature.dataset_orm import FeatureDataset
from app.engines.feature_engine.versioning import compute_version_hash


class FeatureStore:
    def __init__(self):
        self._settings = get_settings()
        self._storage = get_storage_client()
        self._cache = FeatureCache(ttl_seconds=3600)
        self._bucket = self._settings.S3_BUCKET_FEATURES

    # ---- lookup -----------------------------------------------------

    async def find_existing(
        self,
        db: AsyncSession,
        feature_id,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        version_hash: str,
    ) -> Optional[FeatureDataset]:
        result = await db.execute(
            select(FeatureDataset).where(
                FeatureDataset.feature_id == feature_id,
                FeatureDataset.symbol == symbol,
                FeatureDataset.timeframe == timeframe,
                FeatureDataset.start_date == start_date,
                FeatureDataset.end_date == end_date,
                FeatureDataset.version_hash == version_hash,
            )
        )
        return result.scalar_one_or_none()

    # ---- read ---------------------------------------------------------

    def read_dataframe(self, dataset: FeatureDataset) -> pd.DataFrame:
        cached = self._cache.get("dataset", dataset.version_hash)
        if cached is not None:
            return pd.read_parquet(io.BytesIO(cached))

        bucket, key = self._storage.parse_uri(dataset.storage_uri)
        raw = self._storage.get_bytes(bucket, key)
        self._cache.set("dataset", dataset.version_hash, raw)
        return pd.read_parquet(io.BytesIO(raw))

    # ---- write ----------------------------------------------------------

    async def write_dataframe(
        self,
        db: AsyncSession,
        feature_id,
        plugin_key: str,
        params: dict,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        df: pd.DataFrame,
        source_fingerprint: str = "",
    ) -> FeatureDataset:
        version_hash = compute_version_hash(
            plugin_key, params, symbol, timeframe, start_date, end_date, source_fingerprint
        )

        existing = await self.find_existing(
            db, feature_id, symbol, timeframe, start_date, end_date, version_hash
        )
        if existing is not None:
            return existing  # identical inputs already computed -> reuse

        buf = io.BytesIO()
        df.to_parquet(buf, index=True)
        raw = buf.getvalue()

        key = f"{plugin_key}/{symbol}/{timeframe}/{version_hash}.parquet"
        storage_uri = self._storage.put_bytes(
            self._bucket, key, raw, content_type="application/octet-stream"
        )
        self._cache.set("dataset", version_hash, raw)

        dataset = FeatureDataset(
            feature_id=feature_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            version_hash=version_hash,
            storage_uri=storage_uri,
            row_count=len(df),
            columns=list(df.columns),
            source_fingerprint=source_fingerprint,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        return dataset

    # ---- regeneration -----------------------------------------------

    async def list_versions(
        self, db: AsyncSession, feature_id, symbol: str, timeframe: str
    ) -> list[FeatureDataset]:
        """All historical dataset versions for a feature/symbol/timeframe,
        most recent first — used by the regeneration/audit UI."""
        result = await db.execute(
            select(FeatureDataset)
            .where(
                FeatureDataset.feature_id == feature_id,
                FeatureDataset.symbol == symbol,
                FeatureDataset.timeframe == timeframe,
            )
            .order_by(FeatureDataset.created_at.desc())
        )
        return list(result.scalars().all())
