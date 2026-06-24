"""
Training dataset assembler.

Bridges the Feature Engine/Store (Phase 2) and the Model Training Engine
(Phase 3): joins one or more generated feature datasets into a wide X
matrix and derives a y target from forward returns on the underlying
close price. Row alignment and NaN handling (from indicator warm-up
windows or forward-looking targets) happen here so the trainer always
receives a clean, aligned (X, y) pair.
"""
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.feature.orm import Feature
from app.engines.feature_engine.market_data_client import get_market_data_client
from app.engines.feature_engine.pipeline import FeaturePipeline


@dataclass
class TrainingDataset:
    X: pd.DataFrame
    y: pd.Series
    feature_columns: list[str]


async def assemble_training_data(
    db: AsyncSession,
    features: list[Feature],
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    target_horizon: int = 1,
) -> TrainingDataset:
    """
    target_horizon: number of bars ahead the forward return target looks.
    e.g. target_horizon=1 -> predict next-bar return.
    """
    market_data_client = get_market_data_client()
    ohlcv = await market_data_client.get_ohlcv(symbol, timeframe, start_date, end_date)
    if ohlcv.empty:
        raise ValueError("Market Data Layer returned no OHLCV data for the requested range")

    pipeline = FeaturePipeline()
    X = await pipeline.run_many(
        db=db,
        features=features,
        market_data=ohlcv,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        join=True,
    )

    # Forward return target: (close[t+h] / close[t]) - 1
    y = ohlcv["close"].shift(-target_horizon) / ohlcv["close"] - 1.0
    y.name = "forward_return"

    aligned = X.join(y, how="inner").dropna()
    feature_columns = [c for c in aligned.columns if c != "forward_return"]

    return TrainingDataset(
        X=aligned[feature_columns], y=aligned["forward_return"], feature_columns=feature_columns
    )
