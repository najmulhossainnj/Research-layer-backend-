"""
Model plugin package. Import each model plugin module so it self-registers.
"""
from app.plugins.base import BaseModel
from app.plugins.registry import PluginRegistry

model_registry: PluginRegistry[BaseModel] = PluginRegistry("model")

from app.plugins.models import (  # noqa: E402,F401
    catboost_model,
    lightgbm_model,
    lstm_model,
    random_forest_model,
    xgboost_model,
)
