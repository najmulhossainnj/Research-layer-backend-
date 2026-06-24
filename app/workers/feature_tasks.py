"""
Async feature generation tasks.

Dispatched from POST /features/{id}/generate when large date ranges
or expensive tsfresh/FinBERT runs should not block the API thread.
"""
import asyncio
import uuid

from app.workers.celery_app import celery_app


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="features.generate")
def generate_feature_task(
    self,
    feature_id: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
):
    from datetime import datetime

    async def _inner():
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

        from app.core.config import get_settings
        from app.db.crud_base import CRUDRepository
        from app.domain.feature.orm import Feature
        from app.domain.feature.schemas import FeatureCreate, FeatureUpdate
        from app.engines.feature_engine.market_data_client import get_market_data_client
        from app.engines.feature_engine.pipeline import FeaturePipeline

        settings = get_settings()
        async_engine = create_async_engine(settings.DATABASE_URL)
        AsyncSess = async_sessionmaker(async_engine, expire_on_commit=False)

        async with AsyncSess() as db:
            repo = CRUDRepository[Feature, FeatureCreate, FeatureUpdate](Feature)
            feature = await repo.get(db, uuid.UUID(feature_id))
            if feature is None:
                raise ValueError(f"Feature {feature_id} not found")

            ohlcv = await get_market_data_client().get_ohlcv(
                symbol, timeframe,
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date),
            )

            pipeline = FeaturePipeline()
            result = await pipeline.run_one(
                db=db,
                feature=feature,
                market_data=ohlcv,
                symbol=symbol,
                timeframe=timeframe,
                start_date=datetime.fromisoformat(start_date),
                end_date=datetime.fromisoformat(end_date),
            )

            return {
                "dataset_id": str(result.dataset.id),
                "version_hash": result.dataset.version_hash,
                "row_count": result.dataset.row_count,
                "storage_uri": result.dataset.storage_uri,
            }

    self.update_state(state="STARTED", meta={"feature_id": feature_id, "symbol": symbol})
    return _run_async(_inner())
