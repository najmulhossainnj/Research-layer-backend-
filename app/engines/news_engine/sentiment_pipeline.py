"""
News sentiment pipeline.

Full pipeline from raw news → FinBERT scores → Feature Store parquet:

  1. Fetch news articles/headlines from the Market Data Layer
  2. Score each article with FinBERT (batched, cached by content hash)
  3. Aggregate per-article scores to daily bars aligned with the OHLCV index
  4. Produce a DataFrame with the five feature columns the spec requires:
       positive_score, negative_score, neutral_score,
       uncertainty_score, article_volume
  5. Persist through the Feature Store (versioned, reproducible)

Article-level score caching
---------------------------
Each article is identified by SHA-256 of its text.  Scores are cached in
Redis so a daily re-run for a universe of 500 stocks doesn't re-score
articles that appeared the day before.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

from app.core.cache import FeatureCache
from app.engines.news_engine.finbert_adapter import SentimentScores, score_texts


_ARTICLE_CACHE = FeatureCache(ttl_seconds=86400 * 7)  # 7-day TTL


def _article_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _get_cached_score(text: str) -> Optional[SentimentScores]:
    raw = _ARTICLE_CACHE.get("article_scores", _article_cache_key(text))
    if raw is None:
        return None
    data = json.loads(raw)
    return SentimentScores(**data)


def _cache_score(text: str, score: SentimentScores) -> None:
    payload = json.dumps(score._asdict()).encode()
    _ARTICLE_CACHE.set("article_scores", _article_cache_key(text), payload)


def score_articles_cached(texts: list[str]) -> list[SentimentScores]:
    """Score articles, using per-article Redis cache to avoid redundant inference."""
    results: list[Optional[SentimentScores]] = [None] * len(texts)
    uncached_idx: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = _get_cached_score(text)
        if cached is not None:
            results[i] = cached
        else:
            uncached_idx.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        fresh_scores = score_texts(uncached_texts)
        for i, score in zip(uncached_idx, fresh_scores):
            results[i] = score
            _cache_score(texts[i], score)

    return [r for r in results if r is not None]


def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    text_col: str = "headline",
    date_col: str = "published_at",
) -> pd.DataFrame:
    """
    Score all articles and aggregate to daily bars aligned with `price_index`.

    Parameters
    ----------
    news_df     : DataFrame with at least `text_col` and `date_col` columns.
    price_index : DatetimeIndex from the OHLCV close series — defines which
                  trading days to produce sentiment rows for.
    text_col    : Column name containing article text/headline.
    date_col    : Column name containing the publication timestamp.

    Returns
    -------
    DataFrame indexed by trading day with columns:
        positive_score, negative_score, neutral_score,
        uncertainty_score, article_volume
    Dates with no news get NaN scores and article_volume = 0.
    """
    if news_df.empty or text_col not in news_df.columns:
        return _empty_sentiment(price_index)

    news_df = news_df.copy()
    news_df[date_col] = pd.to_datetime(news_df[date_col], utc=True, errors="coerce")
    news_df = news_df.dropna(subset=[date_col])
    news_df["trade_date"] = news_df[date_col].dt.normalize().dt.tz_localize(None)

    texts = news_df[text_col].fillna("").tolist()
    scores = score_articles_cached(texts)

    if not scores:
        return _empty_sentiment(price_index)

    news_df = news_df.iloc[: len(scores)].copy()
    news_df["positive"]    = [s.positive    for s in scores]
    news_df["negative"]    = [s.negative    for s in scores]
    news_df["neutral"]     = [s.neutral     for s in scores]
    news_df["uncertainty"] = [s.uncertainty for s in scores]

    daily = (
        news_df
        .groupby("trade_date")
        .agg(
            positive_score=("positive",    "mean"),
            negative_score=("negative",    "mean"),
            neutral_score=("neutral",      "mean"),
            uncertainty_score=("uncertainty", "mean"),
            article_volume=("positive",    "count"),
        )
    )

    # Reindex to the full trading day grid; fill missing days
    result = daily.reindex(price_index.normalize())
    result["article_volume"] = result["article_volume"].fillna(0).astype(int)
    return result


def _empty_sentiment(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = price_index.normalize()
    return pd.DataFrame(
        {
            "positive_score":   np.nan,
            "negative_score":   np.nan,
            "neutral_score":    np.nan,
            "uncertainty_score": np.nan,
            "article_volume":   0,
        },
        index=idx,
    )
