"""
News sentiment feature plugins.

Exposes the full FinBERT sentiment pipeline as a `BaseFeature` plugin so
it integrates transparently with the Feature Engine, Feature Store, and
Strategy Builder UI exactly like any technical indicator.

Plugins registered here:
  news.finbert_sentiment  — full 4-score + volume feature set
  news.sentiment_momentum — rolling mean of composite sentiment score
  news.sentiment_divergence — spread between positive and negative scores
"""
import pandas as pd

from app.engines.news_engine.sentiment_pipeline import aggregate_daily_sentiment
from app.engines.feature_engine.market_data_client import get_market_data_client
from app.plugins.base import BaseFeature
from app.plugins.features import feature_registry


@feature_registry.register("news.finbert_sentiment")
class FinBERTSentimentFeature(BaseFeature):
    """
    Full FinBERT sentiment feature set.

    params:
        text_col    : str   column in the news DataFrame (default "headline")
        date_col    : str   publication timestamp column (default "published_at")
        symbol      : str   ticker to fetch news for (injected at compute time)
        start_date  : str   ISO datetime
        end_date    : str   ISO datetime

    Requires the Market Data Layer to expose GET /api/v1/news?symbol=&start=&end=.
    """

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        `data` is the OHLCV DataFrame whose index defines the trading day
        grid.  News is fetched from the Market Data Layer inside the plugin
        because it is a separate data source from OHLCV.
        """
        import asyncio
        from datetime import datetime

        symbol = self.params.get("symbol", "")
        start = self.params.get("start_date")
        end = self.params.get("end_date")
        text_col = self.params.get("text_col", "headline")
        date_col = self.params.get("date_col", "published_at")

        if not symbol or not start or not end:
            # Return zeros if params are incomplete — safe fallback for
            # pipeline runs that don't configure the news feature correctly
            return _zero_sentiment(data.index)

        # Fetch news synchronously (plugin compute() is called from async
        # context via run_in_executor in production; here we use a new loop)
        async def _fetch():
            client = get_market_data_client()
            return await client.get_news(
                symbol,
                datetime.fromisoformat(start),
                datetime.fromisoformat(end),
            )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _fetch())
                    news_df = future.result(timeout=30)
            else:
                news_df = loop.run_until_complete(_fetch())
        except Exception:
            return _zero_sentiment(data.index)

        return aggregate_daily_sentiment(
            news_df=news_df,
            price_index=data.index,
            text_col=text_col,
            date_col=date_col,
        )


@feature_registry.register("news.sentiment_momentum")
class SentimentMomentumFeature(BaseFeature):
    """
    Rolling mean of composite sentiment (positive - negative) score.

    Requires FinBERT sentiment columns already in `data` (pass this plugin
    downstream from news.finbert_sentiment in the pipeline).

    params:
        window : int  rolling window in bars (default 5)
    """

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get("window", 5)
        if "positive_score" not in data.columns or "negative_score" not in data.columns:
            return pd.DataFrame(
                {"sentiment_momentum": pd.Series(dtype=float)}, index=data.index
            )
        composite = data["positive_score"] - data["negative_score"]
        momentum = composite.rolling(window, min_periods=1).mean()
        return pd.DataFrame({"sentiment_momentum": momentum}, index=data.index)


@feature_registry.register("news.sentiment_divergence")
class SentimentDivergenceFeature(BaseFeature):
    """
    Positive score minus negative score at each bar — a signed sentiment
    indicator in [-1, +1].

    params:
        normalise : bool  z-score normalise over a rolling window (default True)
        window    : int   normalisation window (default 20)
    """

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        if "positive_score" not in data.columns or "negative_score" not in data.columns:
            return pd.DataFrame(
                {"sentiment_divergence": pd.Series(dtype=float)}, index=data.index
            )

        divergence = data["positive_score"] - data["negative_score"]

        if self.params.get("normalise", True):
            window = self.params.get("window", 20)
            mu = divergence.rolling(window, min_periods=1).mean()
            sigma = divergence.rolling(window, min_periods=1).std().replace(0, 1.0)
            divergence = (divergence - mu) / sigma

        return pd.DataFrame({"sentiment_divergence": divergence}, index=data.index)


def _zero_sentiment(index: pd.Index) -> pd.DataFrame:
    import numpy as np
    return pd.DataFrame(
        {
            "positive_score":   0.0,
            "negative_score":   0.0,
            "neutral_score":    1.0,
            "uncertainty_score": 0.0,
            "article_volume":   0,
        },
        index=index,
    )
