"""
Async model training tasks.

The `/models/{id}/train` endpoint can either run synchronously (small
datasets, dev mode) or dispatch here for background execution. The task
runs in its own DB session (sync SQLAlchemy via `AsyncSession` is not
safe across process boundaries, so we use a sync sessionmaker inside the
worker process).
"""
import asyncio
import uuid

from app.workers.celery_app import celery_app


def _run_async(coro):
    """Run an async coroutine from a Celery task (sync context)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="training.train_model")
def train_model_task(
    self,
    model_id: str,
    feature_ids: list[str],
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    target_horizon: int = 1,
    cv_config: dict | None = None,
):
    """Train a model asynchronously.  Dispatched from POST /models/{id}/train
    when `async_mode=true` is passed as a query param (Phase 3+)."""
    from datetime import datetime

    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine

    from app.core.config import get_settings
    from app.db.session import Base
    import app.db.models_registry  # noqa: F401

    settings = get_settings()
    # Use a sync engine inside the worker (asyncpg is not fork-safe).
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    SyncSession = sessionmaker(bind=engine)

    async def _inner():
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

        from app.db.crud_base import CRUDRepository
        from app.domain.model.orm import MLModel
        from app.domain.model.schemas import ModelCreate, ModelUpdate
        from app.domain.feature.orm import Feature
        from app.engines.model_training_engine.cross_validation import CVConfig
        from app.engines.model_training_engine.dataset_assembler import assemble_training_data
        from app.engines.model_training_engine.trainer import ModelTrainer

        async_engine = create_async_engine(settings.DATABASE_URL)
        AsyncSess = async_sessionmaker(async_engine, expire_on_commit=False)

        async with AsyncSess() as db:
            repo = CRUDRepository[MLModel, ModelCreate, ModelUpdate](MLModel)
            model = await repo.get(db, uuid.UUID(model_id))
            if model is None:
                raise ValueError(f"Model {model_id} not found")

            from sqlalchemy import select

            result = await db.execute(
                select(Feature).where(Feature.id.in_([uuid.UUID(fid) for fid in feature_ids]))
            )
            features = list(result.scalars().all())

            training_data = await assemble_training_data(
                db=db,
                features=features,
                symbol=symbol,
                timeframe=timeframe,
                start_date=datetime.fromisoformat(start_date),
                end_date=datetime.fromisoformat(end_date),
                target_horizon=target_horizon,
            )

            cv = CVConfig(**(cv_config or {}))
            trainer = ModelTrainer()
            updated_model, cv_result = await trainer.train(
                db=db, model=model, X=training_data.X, y=training_data.y, cv_config=cv
            )

            return {
                "model_id": str(updated_model.id),
                "artifact_uri": updated_model.artifact_uri,
                "cv_metrics": cv_result.summary(),
            }

    self.update_state(state="STARTED", meta={"model_id": model_id})
    result = _run_async(_inner())
    return result


@celery_app.task(bind=True, name="training.tune_model")
def tune_model_task(
    self,
    plugin_key: str,
    feature_ids: list[str],
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    param_space: dict,
    n_trials: int = 30,
    cv_config: dict | None = None,
    metric: str = "mean_mse",
    target_horizon: int = 1,
):
    """Optuna tuning as a background task."""
    from datetime import datetime

    async def _inner():
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from sqlalchemy import select

        from app.core.config import get_settings
        from app.domain.feature.orm import Feature
        from app.engines.model_training_engine.cross_validation import CVConfig
        from app.engines.model_training_engine.dataset_assembler import assemble_training_data
        from app.engines.model_training_engine.tuning import tune_hyperparameters

        settings = get_settings()
        async_engine = create_async_engine(settings.DATABASE_URL)
        AsyncSess = async_sessionmaker(async_engine, expire_on_commit=False)

        async with AsyncSess() as db:
            result = await db.execute(
                select(Feature).where(Feature.id.in_([uuid.UUID(fid) for fid in feature_ids]))
            )
            features = list(result.scalars().all())
            training_data = await assemble_training_data(
                db=db,
                features=features,
                symbol=symbol,
                timeframe=timeframe,
                start_date=datetime.fromisoformat(start_date),
                end_date=datetime.fromisoformat(end_date),
                target_horizon=target_horizon,
            )

        cv = CVConfig(**(cv_config or {}))
        tuning_result = tune_hyperparameters(
            plugin_key=plugin_key,
            X=training_data.X,
            y=training_data.y,
            param_space=param_space,
            n_trials=n_trials,
            cv_config=cv,
            metric=metric,
        )
        return {
            "best_params": tuning_result.best_params,
            "best_score": tuning_result.best_score,
            "n_trials": tuning_result.n_trials,
        }

    self.update_state(state="STARTED", meta={"plugin_key": plugin_key})
    return _run_async(_inner())
