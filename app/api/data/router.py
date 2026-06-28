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
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

# Import the singleton factory from your market data client
from app.engines.feature_engine.market_data_client import get_market_data_client

router = APIRouter(prefix="/data", tags=["data-inspection"])


def parse_date(date_str: str) -> datetime:
    """Helper to parse query string dates into datetime objects required by the client."""
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid date format: '{date_str}'. Use ISO format (YYYY-MM-DD)."
        )


@router.get("/ohlcv")
async def inspect_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start: str = "2020-01-01",
    end: str = "2025-01-01",
):
    client = get_market_data_client()
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    
    try:
        df = await client.get_ohlcv(symbol, timeframe, start_dt, end_dt)
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
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/news")
async def inspect_news(
    symbol: str,
    start: str = "2020-01-01",
    end: str = "2025-01-01",
    limit: int = Query(50, le=500),
):
    client = get_market_data_client()
    start_dt = parse_date(start)
    end_dt = parse_date(end)

    try:
        df = await client.get_news(symbol, start_dt, end_dt)
        if df.empty:
            return {"symbol": symbol, "n_articles": 0, "data": []}
        records = df.head(limit).to_dict(orient="records")
        return {
            "symbol":     symbol,
            "n_articles": len(df),
            "showing":    len(records),
            "data":       records,
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/fundamentals")
async def inspect_fundamentals(symbol: str):
    client = get_market_data_client()
    try:
        return await client.get_fundamentals(symbol)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/macro")
async def inspect_macro(
    series: str,
    start: str = "2020-01-01",
    end: str = "2025-01-01",
):
    client = get_market_data_client()
    start_dt = parse_date(start)
    end_dt = parse_date(end)

    try:
        df = await client.get_macro(series, start_dt, end_dt)
        if df.empty:
            return {"series": series, "n_points": 0, "data": []}
        records = df.to_dict(orient="records")
        for r in records:
            r["date"] = str(r["date"])
        return {"series": series, "n_points": len(records), "data": records}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/symbols/search")
async def search_symbol(q: str = Query(..., min_length=1)):
    """
    Quick symbol validation — tries to fetch 5 bars of recent data.
    Returns whether the symbol is valid and a price snapshot.
    """
    client = get_market_data_client()
    start_dt = parse_date("2024-01-01")
    end_dt = parse_date("2025-01-01")

    try:
        df = await client.get_ohlcv(q.upper(), "1d", start_dt, end_dt)
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
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
      
