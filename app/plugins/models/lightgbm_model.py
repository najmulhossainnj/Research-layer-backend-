"""
LightGBM model plugin.
"""
import pandas as pd

from app.plugins.base import BaseModel
from app.plugins.models import model_registry


@model_registry.register("ml.lightgbm")
class LightGBMModel(BaseModel):
    def __init__(self, **params):
        super().__init__(**params)
        self._model = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb

        self._model = lgb.LGBMRegressor(
            num_leaves=self.params.get("num_leaves", 31),
            learning_rate=self.params.get("learning_rate", 0.05),
            n_estimators=self.params.get("n_estimators", 200),
            min_child_samples=self.params.get("min_child_samples", 20),
        )
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")
        return pd.Series(self._model.predict(X), index=X.index)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save")
        self._model.booster_.save_model(path)

    def load(self, path: str) -> None:
        import lightgbm as lgb

        self._model = lgb.Booster(model_file=path)
