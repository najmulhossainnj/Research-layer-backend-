"""
MLflow tracking client façade.

Every interaction with MLflow goes through this module so the rest of the
codebase stays free of direct mlflow imports.  This makes it trivial to
swap providers (Weights & Biases, Neptune, etc.) later without touching
engine or API code.

Responsibilities
----------------
- Ensure experiments exist before logging to them
- Start / end runs with full lineage tags (dataset, feature, model,
  signal, git commit, strategy versions)
- Log params, metrics, and artifacts (model weights, equity curves,
  trade lists) to the run
- Return run_ids for storage in the `Experiment` and `MLModel` rows
"""
import os
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Optional

import mlflow
import mlflow.artifacts

from app.core.config import get_settings


def _get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


@lru_cache
def _setup_mlflow() -> str:
    """Configure MLflow once and return the experiment id."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # Point artifact store at MinIO / S3 if configured
    if settings.MLFLOW_ARTIFACT_ROOT:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.S3_ENDPOINT_URL)
        os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.S3_ACCESS_KEY)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.S3_SECRET_KEY)

    experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            settings.MLFLOW_EXPERIMENT_NAME,
            artifact_location=settings.MLFLOW_ARTIFACT_ROOT or None,
        )
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


# ── Lineage tags ───────────────────────────────────────────────────────────

def _lineage_tags(
    strategy_id: Optional[str] = None,
    dataset_version: Optional[str] = None,
    feature_version: Optional[str] = None,
    model_version: Optional[str] = None,
    signal_version: Optional[str] = None,
    git_commit: Optional[str] = None,
    run_type: str = "training",
) -> dict[str, str]:
    git = git_commit or _get_git_commit() or "unknown"
    tags: dict[str, str] = {
        "mlflow.runName": run_type,
        "run_type": run_type,
        "git_commit": git,
    }
    if strategy_id:
        tags["strategy_id"] = strategy_id
    if dataset_version:
        tags["dataset_version"] = dataset_version
    if feature_version:
        tags["feature_version"] = feature_version
    if model_version:
        tags["model_version"] = model_version
    if signal_version:
        tags["signal_version"] = signal_version
    return tags


# ── Context manager ────────────────────────────────────────────────────────

@contextmanager
def mlflow_run(
    run_name: str,
    run_type: str = "training",
    strategy_id: Optional[str] = None,
    dataset_version: Optional[str] = None,
    feature_version: Optional[str] = None,
    model_version: Optional[str] = None,
    signal_version: Optional[str] = None,
    git_commit: Optional[str] = None,
    nested: bool = False,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that starts an MLflow run with full lineage tags."""
    experiment_id = _setup_mlflow()
    tags = _lineage_tags(
        strategy_id=strategy_id,
        dataset_version=dataset_version,
        feature_version=feature_version,
        model_version=model_version,
        signal_version=signal_version,
        git_commit=git_commit,
        run_type=run_type,
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
        nested=nested,
    ) as run:
        yield run


# ── Logging helpers ────────────────────────────────────────────────────────

def log_params(params: dict[str, Any]) -> None:
    """Log a flat dict of hyperparameters to the active MLflow run."""
    flat = {str(k): str(v) for k, v in params.items()}
    mlflow.log_params(flat)


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact_bytes(data: bytes, artifact_path: str, filename: str) -> str:
    """Write bytes to a temp file and log to MLflow artifacts. Returns artifact URI."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / filename
        p.write_bytes(data)
        mlflow.log_artifact(str(p), artifact_path=artifact_path)
    run = mlflow.active_run()
    if run:
        return f"{mlflow.get_artifact_uri(artifact_path)}/{filename}"
    return ""


def log_artifact_file(local_path: str, artifact_path: str = "") -> str:
    mlflow.log_artifact(local_path, artifact_path=artifact_path)
    return f"{mlflow.get_artifact_uri(artifact_path)}/{Path(local_path).name}"


def get_run(run_id: str) -> Optional[mlflow.entities.Run]:
    try:
        _setup_mlflow()
        return mlflow.get_run(run_id)
    except Exception:
        return None


def get_run_metrics(run_id: str) -> dict:
    run = get_run(run_id)
    return dict(run.data.metrics) if run else {}


def get_run_params(run_id: str) -> dict:
    run = get_run(run_id)
    return dict(run.data.params) if run else {}


def search_runs(
    filter_string: str = "",
    order_by: list[str] | None = None,
    max_results: int = 100,
) -> list[dict]:
    """Search runs in the default experiment. Returns list of dicts."""
    _setup_mlflow()
    settings = get_settings()
    runs = mlflow.search_runs(
        experiment_names=[settings.MLFLOW_EXPERIMENT_NAME],
        filter_string=filter_string,
        order_by=order_by or ["start_time DESC"],
        max_results=max_results,
        output_format="list",
    )
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
            "tags": dict(r.data.tags),
        }
        for r in runs
    ]
