"""
Experiment CRUD endpoints + comparison.

Experiments are the central artifact of the Experiment Tracker UI: every
training / tuning run records a row here (and, in Phase 6, also a linked
MLflow run). The comparison endpoint supports the "Compare Runs" surface
in the UI.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud_base import CRUDRepository
from app.db.session import get_db
from app.domain.model.experiment_orm import Experiment
from app.domain.model.experiment_schemas import (
    ExperimentCreate,
    ExperimentRead,
    ExperimentUpdate,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])
repo = CRUDRepository[Experiment, ExperimentCreate, ExperimentUpdate](Experiment)


@router.post("", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
async def create_experiment(payload: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    return await repo.create(db, payload)


@router.get("", response_model=list[ExperimentRead])
async def list_experiments(
    strategy_id: uuid.UUID | None = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    if strategy_id:
        result = await db.execute(
            select(Experiment)
            .where(Experiment.strategy_id == strategy_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    return await repo.list(db, skip=skip, limit=limit)


@router.get("/{experiment_id}", response_model=ExperimentRead)
async def get_experiment(experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    exp = await repo.get(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.patch("/{experiment_id}", response_model=ExperimentRead)
async def update_experiment(
    experiment_id: uuid.UUID, payload: ExperimentUpdate, db: AsyncSession = Depends(get_db)
):
    exp = await repo.get(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return await repo.update(db, exp, payload)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    exp = await repo.get(db, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    await repo.delete(db, exp)


@router.post("/compare")
async def compare_experiments(
    experiment_ids: list[uuid.UUID], db: AsyncSession = Depends(get_db)
):
    """Side-by-side comparison of up to 10 experiment runs."""
    if len(experiment_ids) > 10:
        raise HTTPException(status_code=400, detail="Can compare at most 10 experiments at once")

    result = await db.execute(
        select(Experiment).where(Experiment.id.in_(experiment_ids))
    )
    experiments = list(result.scalars().all())
    if len(experiments) != len(experiment_ids):
        found = {e.id for e in experiments}
        missing = set(experiment_ids) - found
        raise HTTPException(status_code=404, detail=f"Experiment(s) not found: {missing}")

    rows = []
    for exp in experiments:
        rows.append(
            {
                "id": str(exp.id),
                "strategy_id": str(exp.strategy_id),
                "feature_version": exp.feature_version,
                "model_version": exp.model_version,
                "dataset_version": exp.dataset_version,
                "git_commit_hash": exp.git_commit_hash,
                "mlflow_run_id": exp.mlflow_run_id,
                "parameters": exp.parameters,
                "metrics": exp.metrics,
                "artifacts": exp.artifacts,
                "created_at": exp.created_at.isoformat(),
            }
        )

    # Diff metrics across runs so the UI can highlight winners per metric.
    all_metric_keys = sorted({k for r in rows for k in r["metrics"]})
    metric_diff = {}
    for key in all_metric_keys:
        vals = {r["id"]: r["metrics"].get(key) for r in rows}
        numeric = [v for v in vals.values() if isinstance(v, (int, float))]
        metric_diff[key] = {
            "values": vals,
            "best_id": min(vals, key=lambda eid: vals[eid] if isinstance(vals[eid], (int, float)) else float("inf"))
            if numeric
            else None,
        }

    return {"experiments": rows, "metric_diff": metric_diff}
