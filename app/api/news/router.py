"""
News sentiment endpoints.

  POST /news/score          — score a list of raw texts with FinBERT
  POST /news/aggregate      — fetch + score + aggregate for a symbol/range
  GET  /news/features       — list registered news feature plugin keys
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

router = APIRouter(prefix="/news", tags=["news"])


class ScoreRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


class ArticleScore(BaseModel):
    positive: float
    negative: float
    neutral: float
    uncertainty: float


class ScoreResponse(BaseModel):
    scores: list[ArticleScore]
    n_scored: int


@router.post("/score", response_model=ScoreResponse)
async def score_texts_endpoint(payload: ScoreRequest):
    """Score raw news texts with FinBERT. No DB or feature store involved."""
    from app.engines.news_engine.finbert_adapter import score_texts
    try:
        scores = score_texts(payload.texts)
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return ScoreResponse(
        scores=[ArticleScore(**s._asdict()) for s in scores],
        n_scored=len(scores),
    )


class AggregateRequest(BaseModel):
    symbol: str
    start_date: datetime
    end_date: datetime
    text_col: str = "headline"
    date_col: str = "published_at"


@router.post("/aggregate")
async def aggregate_sentiment(payload: AggregateRequest):
    """
    Fetch news from the Market Data Layer, score with FinBERT, and return
    daily aggregated sentiment scores aligned to trading days.
    """
    from app.engines.feature_engine.market_data_client import get_market_data_client
    from app.engines.news_engine.sentiment_pipeline import aggregate_daily_sentiment

    client = get_market_data_client()
    try:
        ohlcv = await client.get_ohlcv(
            payload.symbol, "1d", payload.start_date, payload.end_date
        )
        news_df = await client.get_news(
            payload.symbol, payload.start_date, payload.end_date
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market Data Layer error: {e}")

    if ohlcv.empty:
        raise HTTPException(status_code=422, detail="No OHLCV data for the requested range")

    try:
        result = aggregate_daily_sentiment(
            news_df=news_df,
            price_index=ohlcv.index,
            text_col=payload.text_col,
            date_col=payload.date_col,
        )
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result_clean = result.reset_index()
    result_clean.columns = ["date"] + list(result.columns)
    result_clean["date"] = result_clean["date"].astype(str)
    return {
        "symbol": payload.symbol,
        "n_days": len(result),
        "data": result_clean.to_dict(orient="records"),
    }


@router.get("/features")
async def list_news_feature_plugins():
    from app.plugins.features import feature_registry
    news_keys = [k for k in feature_registry.list_keys() if k.startswith("news.")]
    return {"plugins": news_keys}
