"""
Default hyperparameter search spaces.

Each entry maps a plugin key to the `ParamSpace` dict accepted by
`tune_hyperparameters()` and the `POST /models/tune` endpoint.  The Model
Builder UI reads these from `GET /models/plugins/search-spaces` so the
config form is always driven from the same source of truth as the tuner,
rather than being hardcoded in the frontend.
"""
from typing import Any

DEFAULT_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "ml.xgboost": {
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
        "n_estimators": {"type": "int", "low": 100, "high": 800},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
    },
    "ml.lightgbm": {
        "num_leaves": {"type": "int", "low": 16, "high": 256, "log": True},
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
        "n_estimators": {"type": "int", "low": 100, "high": 800},
        "min_child_samples": {"type": "int", "low": 5, "high": 100},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    },
    "ml.catboost": {
        "depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
        "iterations": {"type": "int", "low": 100, "high": 800},
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "log": True},
    },
    "ml.random_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 600},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
    },
    "dl.lstm": {
        "hidden_size": {"type": "categorical", "choices": ["32", "64", "128", "256"]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "seq_len": {"type": "int", "low": 10, "high": 60},
        "epochs": {"type": "int", "low": 10, "high": 50},
    },
}
