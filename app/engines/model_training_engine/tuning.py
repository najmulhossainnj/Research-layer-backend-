"""
Hyperparameter tuning via Optuna.

Search spaces are declared declaratively (so they can come straight from
the Model Builder UI's dynamically-generated config form) and scored via
the shared CV evaluator, keeping tuning and final-train metrics
apples-to-apples.

Search space spec format, per parameter:
    {"type": "float", "low": 0.001, "high": 0.3, "log": true}
    {"type": "int", "low": 2, "high": 12}
    {"type": "categorical", "choices": ["gbtree", "dart"]}
"""
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from app.engines.model_training_engine.cross_validation import CVConfig
from app.engines.model_training_engine.evaluation import evaluate_with_cv

ParamSpace = dict[str, dict[str, Any]]


@dataclass
class TuningResult:
    best_params: dict
    best_score: float
    n_trials: int
    study_summary: list[dict]


def _suggest(trial: optuna.Trial, name: str, spec: dict) -> Any:
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported param spec type: {kind}")


def tune_hyperparameters(
    plugin_key: str,
    X: pd.DataFrame,
    y: pd.Series,
    param_space: ParamSpace,
    n_trials: int = 30,
    cv_config: CVConfig | None = None,
    direction: str = "minimize",
    metric: str = "mean_mse",
) -> TuningResult:
    cv_config = cv_config or CVConfig()
    study_summary: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in param_space.items()}
        cv_result = evaluate_with_cv(plugin_key, params, X, y, cv_config)
        score = getattr(cv_result, metric)
        study_summary.append({"trial": trial.number, "params": params, "score": score})
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    return TuningResult(
        best_params=study.best_params,
        best_score=study.best_value,
        n_trials=len(study.trials),
        study_summary=study_summary,
    )
