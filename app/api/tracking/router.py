"""
Experiment tracking endpoints (MLflow-backed).

The Experiment Tracker UI calls these rather than hitting MLflow directly,
so we can add platform-specific filtering, pagination, and enrichment
without coupling the frontend to MLflow's API contract.

Routes
------
GET  /tracking/runs                  — search / list runs
GET  /tracking/runs/{run_id}         — single run detail
GET  /tracking/runs/{run_id}/metrics — full metric history
GET  /tracking/runs/{run_id}/params  — hyperparameters
POST /tracking/runs/compare          — side-by-side diff (up to 10)
GET  /tracking/experiments           — list MLflow experiments
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.mlflow_client import (
    get_run,
    get_run_metrics,
    get_run_params,
    search_runs,
)

router = APIRouter(prefix="/tracking", tags=["tracking"])


# ── List / search ─────────────────────────────────────────────────────────

@router.get("/runs")
async def list_runs(
    run_type: Optional[str] = None,
    strategy_id: Optional[str] = None,
    limit: int = 50,
):
    """List MLflow runs with optional filters on run_type and strategy_id."""
    filters = []
    if run_type:
        filters.append(f"tags.run_type = '{run_type}'")
    if strategy_id:
        filters.append(f"tags.strategy_id = '{strategy_id}'")

    filter_string = " AND ".join(filters)
    runs = search_runs(filter_string=filter_string, max_results=limit)
    return {"runs": runs, "total": len(runs)}


# ── Single run ────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"MLflow run {run_id} not found")
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "artifact_uri": run.info.artifact_uri,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


@router.get("/runs/{run_id}/metrics")
async def get_run_metric_history(run_id: str):
    metrics = get_run_metrics(run_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"run_id": run_id, "metrics": metrics}


@router.get("/runs/{run_id}/params")
async def get_run_param_detail(run_id: str):
    params = get_run_params(run_id)
    return {"run_id": run_id, "params": params}


# ── Comparison ────────────────────────────────────────────────────────────

class RunCompareRequest(BaseModel):
    run_ids: list[str] = Field(..., min_length=2, max_length=10)


@router.post("/runs/compare")
async def compare_runs(payload: RunCompareRequest):
    """Side-by-side metric and param diff across up to 10 MLflow runs."""
    run_details = []
    for run_id in payload.run_ids:
        run = get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        run_details.append({
            "run_id": run_id,
            "run_name": run.info.run_name,
            "run_type": run.data.tags.get("run_type", "unknown"),
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
        })

    all_metric_keys = sorted({k for r in run_details for k in r["metrics"]})
    all_param_keys = sorted({k for r in run_details for k in r["params"]})

    metric_diff = {}
    for key in all_metric_keys:
        vals = {r["run_id"]: r["metrics"].get(key) for r in run_details}
        numeric = {rid: v for rid, v in vals.items() if isinstance(v, (int, float))}
        best_id = max(numeric, key=lambda rid: numeric[rid]) if numeric else None
        metric_diff[key] = {"values": vals, "best_run_id": best_id}

    param_diff = {}
    for key in all_param_keys:
        vals = {r["run_id"]: r["params"].get(key) for r in run_details}
        unique_values = set(str(v) for v in vals.values() if v is not None)
        param_diff[key] = {"values": vals, "varies": len(unique_values) > 1}

    return {
        "runs": run_details,
        "metric_diff": metric_diff,
        "param_diff": param_diff,
    }


# ── Experiments ───────────────────────────────────────────────────────────

@router.get("/experiments")
async def list_experiments():
    """List all MLflow experiments in this tracking server."""
    import mlflow
    from app.core.mlflow_client import _setup_mlflow
    _setup_mlflow()
    experiments = mlflow.search_experiments()
    return {
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "artifact_location": e.artifact_location,
                "lifecycle_stage": e.lifecycle_stage,
            }
            for e in experiments
        ]
    }
