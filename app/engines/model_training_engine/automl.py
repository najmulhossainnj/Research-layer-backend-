"""
AutoML.

Runs the same CV protocol across a set of candidate model plugins (and
optionally a per-plugin hyperparameter search) and ranks them by a chosen
metric, so the Model Builder UI / AI Researcher agent can pick a sensible
default without the user hand-comparing every model family.
"""
from dataclasses import dataclass, field

import pandas as pd

from app.engines.model_training_engine.cross_validation import CVConfig
from app.engines.model_training_engine.evaluation import evaluate_with_cv
from app.engines.model_training_engine.tuning import ParamSpace, tune_hyperparameters


@dataclass
class CandidateResult:
    plugin_key: str
    params: dict
    score: float
    metrics: dict


@dataclass
class AutoMLResult:
    leaderboard: list[CandidateResult] = field(default_factory=list)

    @property
    def best(self) -> CandidateResult | None:
        return self.leaderboard[0] if self.leaderboard else None


def run_automl(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: dict[str, dict],
    cv_config: CVConfig | None = None,
    metric: str = "mean_mse",
    tune_trials: int = 0,
) -> AutoMLResult:
    """
    candidates: {plugin_key: default_params} for plugins to evaluate as-is,
                e.g. {"ml.xgboost": {"max_depth": 6}}.
                Pass `tune_trials > 0` to instead Optuna-tune each candidate
                using its params as a fixed search space-free default
                (use `tune_candidates` below for per-candidate search spaces).
    """
    cv_config = cv_config or CVConfig()
    results: list[CandidateResult] = []

    for plugin_key, params in candidates.items():
        cv_result = evaluate_with_cv(plugin_key, params, X, y, cv_config)
        score = getattr(cv_result, metric)
        results.append(
            CandidateResult(
                plugin_key=plugin_key, params=params, score=score, metrics=cv_result.summary()
            )
        )

    ascending = metric != "mean_directional_accuracy"  # lower is better except accuracy
    results.sort(key=lambda r: r.score, reverse=not ascending)
    return AutoMLResult(leaderboard=results)


def tune_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    search_spaces: dict[str, ParamSpace],
    cv_config: CVConfig | None = None,
    metric: str = "mean_mse",
    n_trials: int = 20,
) -> AutoMLResult:
    """Like run_automl, but each candidate gets its own Optuna search
    rather than fixed default params."""
    cv_config = cv_config or CVConfig()
    results: list[CandidateResult] = []

    for plugin_key, space in search_spaces.items():
        tuning = tune_hyperparameters(
            plugin_key=plugin_key,
            X=X,
            y=y,
            param_space=space,
            n_trials=n_trials,
            cv_config=cv_config,
            metric=metric,
        )
        cv_result = evaluate_with_cv(plugin_key, tuning.best_params, X, y, cv_config)
        results.append(
            CandidateResult(
                plugin_key=plugin_key,
                params=tuning.best_params,
                score=tuning.best_score,
                metrics=cv_result.summary(),
            )
        )

    ascending = metric != "mean_directional_accuracy"
    results.sort(key=lambda r: r.score, reverse=not ascending)
    return AutoMLResult(leaderboard=results)
