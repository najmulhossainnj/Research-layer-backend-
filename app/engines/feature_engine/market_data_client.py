"""
Market Data Layer client.

The Research Layer never owns market data — it consumes OHLCV, news,
fundamentals, and macro data from the external Market Data Layer over
HTTP, per the system interface contract in the spec. This client is the
single integration point; swap the implementation (or point MARKET_DATA_URL
elsewhere) without touching the Feature Engine.
"""
from datetime import datetime
from functools import lru_cache

import httpx
import pandas as pd

from app.core.config import get_settings


class MarketDataClient:
    def __init__(self, base_url: str | None = None):
        settings = get_settings()
        self.base_url = base_url or getattr(settings, "MARKET_DATA_URL", "http://localhost:8001")

    async def get_ohlcv(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from the Market Data Layer.

        Expected response: JSON list of {timestamp, open, high, low, close, volume}.
        """
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        }
        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
            resp = await client.get("/api/v1/ohlcv", params=params)
            resp.raise_for_status()
            rows = resp.json()

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    async def get_news(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        params = {"symbol": symbol, "start": start_date.isoformat(), "end": end_date.isoformat()}
        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
            resp = await client.get("/api/v1/news", params=params)
            resp.raise_for_status()
            rows = resp.json()
        return pd.DataFrame(rows)


@lru_cache
def get_market_data_client() -> MarketDataClient:
    return MarketDataClient()
