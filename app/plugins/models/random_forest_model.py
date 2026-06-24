"""
Random Forest model plugin.
"""
import pickle

import pandas as pd

from app.plugins.base import BaseModel
from app.plugins.models import model_registry


@model_registry.register("ml.random_forest")
class RandomForestModel(BaseModel):
    def __init__(self, **params):
        super().__init__(**params)
        self._model = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", None),
            min_samples_leaf=self.params.get("min_samples_leaf", 5),
            n_jobs=self.params.get("n_jobs", -1),
            random_state=42,
        )
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")
        return pd.Series(self._model.predict(X), index=X.index)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save")
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
