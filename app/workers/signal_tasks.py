"""
Async signal generation tasks.

Dispatched from POST /signals/generate/async for large universes or
long date ranges that would otherwise block the API thread.
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


@celery_app.task(bind=True, name="signals.generate")
def generate_signals_task(
    self,
    model_id: str,
    feature_ids: list[str],
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    signal_logic_id: str | None = None,
    plugin_key: str | None = None,
    plugin_params: dict | None = None,
    target_horizon: int = 1,
):
    from datetime import datetime

    async def _inner():
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.core.config import get_settings
        from app.domain.feature.orm import Feature
        from app.domain.model.orm import MLModel
        from app.engines.feature_engine.market_data_client import get_market_data_client
        from app.engines.feature_engine.pipeline import FeaturePipeline
        from app.engines.signal_engine.pipeline import SignalPipeline

        settings = get_settings()
        engine = create_async_engine(settings.DATABASE_URL)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        async with Session() as db:
            result = await db.execute(
                select(MLModel).where(MLModel.id == uuid.UUID(model_id))
            )
            model = result.scalar_one_or_none()
            if model is None:
                raise ValueError(f"Model {model_id} not found")

            feat_result = await db.execute(
                select(Feature).where(
                    Feature.id.in_([uuid.UUID(fid) for fid in feature_ids])
                )
            )
            features = list(feat_result.scalars().all())

            ohlcv = await get_market_data_client().get_ohlcv(
                symbol, timeframe,
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date),
            )

            feature_pipeline = FeaturePipeline()
            feature_df = await feature_pipeline.run_many(
                db=db, features=features, market_data=ohlcv,
                symbol=symbol, timeframe=timeframe,
                start_date=datetime.fromisoformat(start_date),
                end_date=datetime.fromisoformat(end_date),
                join=True,
            )

            signal_pipeline = SignalPipeline()
            if signal_logic_id:
                signal_result = await signal_pipeline.generate_from_rule_tree(
                    db=db,
                    signal_logic_id=uuid.UUID(signal_logic_id),
                    model=model,
                    feature_data=feature_df,
                )
            else:
                signal_result = await signal_pipeline.generate_from_plugin(
                    model=model,
                    feature_data=feature_df,
                    plugin_key=plugin_key,
                    plugin_params=plugin_params or {},
                )

            signals = signal_result.signals
            return {
                "total_bars": len(signals),
                "metadata": signal_result.metadata,
                "signal_counts": signals.value_counts().to_dict(),
            }

    self.update_state(state="STARTED", meta={"model_id": model_id, "symbol": symbol})
    return _run_async(_inner())
