"""
XGBoost model plugin (example ML family implementation).

Other model families (LightGBM, CatBoost, Random Forest, ARIMA, GARCH,
LSTM/GRU/Transformer, ensembles) follow the same `BaseModel` contract and
register themselves under their own key in this package.
"""
import pandas as pd

from app.plugins.base import BaseModel
from app.plugins.models import model_registry


@model_registry.register("ml.xgboost")
class XGBoostModel(BaseModel):
    def __init__(self, **params):
        super().__init__(**params)
        self._model = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        import xgboost as xgb

        self._model = xgb.XGBRegressor(
            max_depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.05),
            n_estimators=self.params.get("n_estimators", 200),
        )
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model has not been trained or loaded yet")
        return pd.Series(self._model.predict(X), index=X.index)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save - model not trained")
        self._model.save_model(path)

    def load(self, path: str) -> None:
        import xgboost as xgb

        self._model = xgb.XGBRegressor()
        self._model.load_model(path)
