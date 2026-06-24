"""
Model training endpoints: train, hyperparameter tuning, AutoML.

Separate from the plain CRUD router (`router.py`) since these trigger
actual computation through the Model Training Engine. Mounted under the
same `/models` prefix.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud_base import CRUDRepository
from app.db.session import get_db
from app.domain.feature.orm import Feature
from app.domain.model.orm import MLModel
from app.domain.model.schemas import ModelCreate, ModelUpdate
from app.domain.model.training_schemas import (
    AutoMLCandidateResponse,
    AutoMLRequest,
    AutoMLResponse,
    DatasetSpec,
    TrainRequest,
    TrainResponse,
    TuneRequest,
    TuneResponse,
)
from app.engines.model_training_engine.automl import run_automl
from app.engines.model_training_engine.dataset_assembler import assemble_training_data
from app.engines.model_training_engine.trainer import ModelTrainer
from app.engines.model_training_engine.tuning import tune_hyperparameters

router = APIRouter(prefix="/models", tags=["models"])
_model_repo = CRUDRepository[MLModel, ModelCreate, ModelUpdate](MLModel)


async def _load_features(db: AsyncSession, feature_ids: list[uuid.UUID]) -> list[Feature]:
    result = await db.execute(select(Feature).where(Feature.id.in_(feature_ids)))
    features = list(result.scalars().all())
    missing = set(feature_ids) - {f.id for f in features}
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown feature id(s): {missing}")
    return features


async def _assemble(db: AsyncSession, dataset: DatasetSpec):
    features = await _load_features(db, dataset.feature_ids)
    try:
        return await assemble_training_data(
            db=db,
            features=features,
            symbol=dataset.symbol,
            timeframe=dataset.timeframe,
            start_date=dataset.start_date,
            end_date=dataset.end_date,
            target_horizon=dataset.target_horizon,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/{model_id}/train", response_model=TrainResponse)
async def train_model(
    model_id: uuid.UUID, payload: TrainRequest, db: AsyncSession = Depends(get_db)
):
    model = await _model_repo.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    training_data = await _assemble(db, payload.dataset)
    if len(training_data.X) < 30:
        raise HTTPException(
            status_code=422,
            detail="Not enough aligned rows to train on (need at least 30 after feature "
            "warm-up and target alignment)",
        )

    trainer = ModelTrainer()
    updated_model, cv_result = await trainer.train(
        db=db,
        model=model,
        X=training_data.X,
        y=training_data.y,
        cv_config=payload.cv.to_config(),
    )

    return TrainResponse(
        model_id=updated_model.id,
        artifact_uri=updated_model.artifact_uri,
        cv_metrics=cv_result.summary(),
        n_train_rows=len(training_data.X),
        feature_columns=training_data.feature_columns,
    )


@router.post("/tune", response_model=TuneResponse)
async def tune_model(payload: TuneRequest, db: AsyncSession = Depends(get_db)):
    training_data = await _assemble(db, payload.dataset)

    space = {name: spec.model_dump(exclude_none=True) for name, spec in payload.param_space.items()}

    result = tune_hyperparameters(
        plugin_key=payload.plugin_key,
        X=training_data.X,
        y=training_data.y,
        param_space=space,
        n_trials=payload.n_trials,
        cv_config=payload.cv.to_config(),
        direction=payload.direction,
        metric=payload.metric,
    )

    return TuneResponse(
        best_params=result.best_params, best_score=result.best_score, n_trials=result.n_trials
    )


@router.post("/automl", response_model=AutoMLResponse)
async def automl(payload: AutoMLRequest, db: AsyncSession = Depends(get_db)):
    training_data = await _assemble(db, payload.dataset)

    result = run_automl(
        X=training_data.X,
        y=training_data.y,
        candidates=payload.candidates,
        cv_config=payload.cv.to_config(),
        metric=payload.metric,
    )

    return AutoMLResponse(
        leaderboard=[
            AutoMLCandidateResponse(
                plugin_key=c.plugin_key, params=c.params, score=c.score, metrics=c.metrics
            )
            for c in result.leaderboard
        ]
    )


@router.post("/{model_id}/train/async")
async def train_model_async(
    model_id: uuid.UUID, payload: TrainRequest, db: AsyncSession = Depends(get_db)
):
    """Dispatch training as a Celery background task.
    Returns immediately with a task_id the client can poll via GET /tasks/{task_id}."""
    from app.workers.training_tasks import train_model_task

    model = await _model_repo.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    ds = payload.dataset
    task = train_model_task.delay(
        model_id=str(model_id),
        feature_ids=[str(fid) for fid in ds.feature_ids],
        symbol=ds.symbol,
        timeframe=ds.timeframe,
        start_date=ds.start_date.isoformat(),
        end_date=ds.end_date.isoformat(),
        target_horizon=ds.target_horizon,
        cv_config=payload.cv.model_dump(),
    )
    return {"task_id": task.id, "status": "PENDING"}


@router.post("/tune/async")
async def tune_model_async(payload: TuneRequest, db: AsyncSession = Depends(get_db)):
    """Dispatch Optuna tuning as a Celery background task."""
    from app.workers.training_tasks import tune_model_task

    ds = payload.dataset
    space = {name: spec.model_dump(exclude_none=True) for name, spec in payload.param_space.items()}
    task = tune_model_task.delay(
        plugin_key=payload.plugin_key,
        feature_ids=[str(fid) for fid in ds.feature_ids],
        symbol=ds.symbol,
        timeframe=ds.timeframe,
        start_date=ds.start_date.isoformat(),
        end_date=ds.end_date.isoformat(),
        param_space=space,
        n_trials=payload.n_trials,
        cv_config=payload.cv.model_dump(),
        metric=payload.metric,
        target_horizon=ds.target_horizon,
    )
    return {"task_id": task.id, "status": "PENDING"}


@router.get("/plugins/available")
async def list_available_model_plugins():
    from app.plugins.models import model_registry

    return {"plugins": model_registry.list_keys()}


@router.get("/plugins/search-spaces")
async def list_search_spaces():
    """Default Optuna search spaces per plugin key — used by the Model Builder UI."""
    from app.engines.model_training_engine.search_spaces import DEFAULT_SEARCH_SPACES

    return DEFAULT_SEARCH_SPACES
