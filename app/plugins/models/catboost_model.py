"""
CatBoost model plugin.
"""
import pandas as pd

from app.plugins.base import BaseModel
from app.plugins.models import model_registry


@model_registry.register("ml.catboost")
class CatBoostModel(BaseModel):
    def __init__(self, **params):
        super().__init__(**params)
        self._model = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        from catboost import CatBoostRegressor

        self._model = CatBoostRegressor(
            depth=self.params.get("depth", 6),
            learning_rate=self.params.get("learning_rate", 0.05),
            iterations=self.params.get("iterations", 200),
            loss_function="RMSE",
            verbose=False,
        )
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")
        return pd.Series(self._model.predict(X), index=X.index)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save")
        self._model.save_model(path)

    def load(self, path: str) -> None:
        from catboost import CatBoostRegressor

        self._model = CatBoostRegressor()
        self._model.load_model(path)
