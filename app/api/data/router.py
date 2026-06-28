# app/api/data/router.py

"""
Data inspection proxy.

Lets the frontend browse raw market data without exposing the
Data Layer API key to the browser. The Research Layer forwards
requests and adds auth headers server-side.

Routes:
  GET /data/ohlcv          → Data Layer /api/v1/ohlcv
  GET /data/news           → Data Layer /api/v1/news
  GET /data/fundamentals   → Data Layer /api/v1/fundamentals
  GET /data/macro          → Data Layer /api/v1/macro
  GET /data/symbols/search → symbol search/validation helper
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.engines.feature_engine.market_data_client import (
    get_ohlcv_async,
    get_news_async,
    get_fundamentals_async,
    get_macro_async,
)

router = APIRouter(prefix="/data", tags=["data-inspection"])


@router.get("/ohlcv")
async def inspect_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start: str = "2020-01-01",
    end: str = "2025-01-01",
):
    try:
        df = await get_ohlcv_async(symbol, timeframe, start, end)
        records = df.reset_index().to_dict(orient="records")
        # Convert timestamps to strings for JSON serialisation
        for r in records:
            r["timestamp"] = str(r["timestamp"])
        return {
            "symbol":    symbol,
            "timeframe": timeframe,
            "start":     start,
            "end":       end,
            "n_bars":    len(records),
            "data":      records,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/news")
async def inspect_news(
    symbol: str,
    start: str = "2020-01-01",
    end: str = "2025-01-01",
    limit: int = Query(50, le=500),
):
    try:
        df = await get_news_async(symbol, start, end)
        if df.empty:
            return {"symbol": symbol, "n_articles": 0, "data": []}
        records = df.head(limit).to_dict(orient="records")
        return {
            "symbol":     symbol,
            "n_articles": len(df),
            "showing":    len(records),
            "data":       records,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/fundamentals")
async def inspect_fundamentals(symbol: str):
    try:
        return await get_fundamentals_async(symbol)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/macro")
async def inspect_macro(
    series: str,
    start: str = "2020-01-01",
    end: str = "2025-01-01",
):
    try:
        df = await get_macro_async(series, start, end)
        if df.empty:
            return {"series": series, "n_points": 0, "data": []}
        records = df.to_dict(orient="records")
        for r in records:
            r["date"] = str(r["date"])
        return {"series": series, "n_points": len(records), "data": records}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/symbols/search")
async def search_symbol(q: str = Query(..., min_length=1)):
    """
    Quick symbol validation — tries to fetch 5 bars of recent data.
    Returns whether the symbol is valid and a price snapshot.
    """
    try:
        df = await get_ohlcv_async(q.upper(), "1d", "2024-01-01", "2025-01-01")
        latest = df.iloc[-1] if not df.empty else None
        return {
            "symbol":  q.upper(),
            "valid":   True,
            "latest_close": float(latest["close"]) if latest is not None else None,
            "latest_date":  str(df.index[-1]) if not df.empty else None,
            "n_bars_available": len(df),
        }
    except ValueError:
        return {"symbol": q.upper(), "valid": False}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
