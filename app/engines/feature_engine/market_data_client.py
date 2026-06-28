"""
Market Data Layer client.

The Research Layer never owns market data — it consumes OHLCV, news,
fundamentals, and macro data from the external Market Data Layer over
HTTP, per the system interface contract in the spec. This client is the
single integration point; swap the implementation (or point MARKET_DATA_URL
elsewhere) without touching the Feature Engine.

Changes from v1:
  - Added X-API-Key header on every request (Data Service auth requirement).
  - Added get_fundamentals() and get_macro() for completeness.
  - Improved error handling: maps HTTP status codes to typed exceptions
    instead of the generic raise_for_status() which hides root cause.
  - Retry logic on network errors (not on 4xx client errors).
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from typing import Any

import httpx
import pandas as pd

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MarketDataClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.base_url = (
            base_url
            or getattr(settings, "MARKET_DATA_URL", None)
            or getattr(settings, "DATA_SERVICE_URL", "http://localhost:8001")
        )
        self.api_key = (
            api_key
            or getattr(settings, "DATA_SERVICE_API_KEY", "dev-api-key-change-in-production")
        )
        self._headers = {
            "X-API-Key": self.api_key,
            "Accept":    "application/json",
        }

    # ── OHLCV ─────────────────────────────────────────────────────────────

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from the Data Service.

        Returns a DataFrame indexed by timestamp with columns:
          [open, high, low, close, volume]

        Returns an empty DataFrame (not raises) when no data exists for
        the range — the Feature Engine checks for emptiness and raises
        its own 422 downstream.
        """
        params = {
            "symbol":    symbol,
            "timeframe": timeframe,
            "start":     start_date.date().isoformat(),
            "end":       end_date.date().isoformat(),
        }

        rows = await self._get("/api/v1/ohlcv", params=params)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    # ── News ──────────────────────────────────────────────────────────────

    async def get_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch news articles from the Data Service.

        Returns a DataFrame with at minimum: [headline, published_at].
        Returns an empty DataFrame when no news exists — the FinBERT
        pipeline checks for emptiness and returns zero sentiment scores.
        """
        params = {
            "symbol": symbol,
            "start":  start_date.date().isoformat(),
            "end":    end_date.date().isoformat(),
        }

        rows = await self._get("/api/v1/news", params=params)
        return pd.DataFrame(rows)

    # ── Fundamentals ──────────────────────────────────────────────────────

    async def get_fundamentals(self, symbol: str) -> dict[str, Any]:
        """
        Fetch latest fundamental metrics for a symbol.

        Returns a dict: {symbol, pe_ratio, pb_ratio, revenue_growth,
        earnings_surprise, market_cap, eps, as_of}.
        Values are None when the provider does not supply them.
        """
        params = {"symbol": symbol}
        return await self._get("/api/v1/fundamentals", params=params)

    # ── Macro ─────────────────────────────────────────────────────────────

    async def get_macro(
        self,
        series: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch a macro time series from the Data Service.

        Args:
            series: One of CPI, FED_FUNDS_RATE, GDP_GROWTH, UNEMPLOYMENT.

        Returns a DataFrame with columns: [date, series, value].
        Returns an empty DataFrame when no data exists for the range.
        """
        params = {
            "series": series,
            "start":  start_date.date().isoformat(),
            "end":    end_date.date().isoformat(),
        }

        rows = await self._get("/api/v1/macro", params=params)
        if not rows:
            return pd.DataFrame(columns=["date", "series", "value"])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── Internal HTTP helper ──────────────────────────────────────────────

    async def _get(self, path: str, params: dict) -> list | dict:
        """
        Async GET with auth header and typed error handling.

        Retries once on network errors. Never retries on 4xx responses —
        those indicate a client-side problem (bad symbol, bad date format)
        that won't resolve on retry.

        Raises:
            ValueError  — 400 (bad params) or 404 (not found)
            PermissionError — 401/403 (wrong or missing API key)
            RuntimeError — 429 (rate limit), 503 (upstream unavailable),
                           or persistent network failure
        """
        last_exc: Exception | None = None

        for attempt in range(1, 3):       # max 2 attempts
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self._headers,
                    timeout=30.0,
                ) as client:
                    resp = await client.get(path, params=params)

                # ── Success ───────────────────────────────────────────────
                if resp.status_code == 200:
                    return resp.json()

                # ── Client errors — don't retry ───────────────────────────
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text

                if resp.status_code == 400:
                    raise ValueError(f"Bad request to Data Service ({path}): {detail}")

                if resp.status_code in (401, 403):
                    raise PermissionError(
                        f"Data Service auth failed ({resp.status_code}). "
                        "Check DATA_SERVICE_API_KEY in Research Layer settings. "
                        f"Detail: {detail}"
                    )

                if resp.status_code == 404:
                    raise ValueError(f"Not found in Data Service ({path}): {detail}")

                if resp.status_code == 429:
                    raise RuntimeError(
                        f"Data Service rate limit hit ({path}). "
                        f"Retry after: {resp.headers.get('Retry-After', 'unknown')}s"
                    )

                if resp.status_code == 503:
                    raise RuntimeError(
                        f"Data Service upstream unavailable ({path}): {detail}"
                    )

                raise RuntimeError(
                    f"Unexpected HTTP {resp.status_code} from Data Service ({path}): {detail}"
                )

            except (ValueError, PermissionError):
                raise   # client errors — never retry

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                logger.warning(
                    "Data Service network error (attempt %d/2) %s %s: %s",
                    attempt, path, params, exc,
                )
                continue    # retry on network errors

            except RuntimeError:
                raise       # server errors — surface immediately

        raise RuntimeError(
            f"Data Service unreachable after 2 attempts ({path}): {last_exc}"
        )


# ── Singleton factory ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_market_data_client() -> MarketDataClient:
    """
    Returns a shared MarketDataClient instance.

    The instance is created once on first call and cached for the lifetime
    of the process. Settings (URL, API key) are read at construction time.
    """
    return MarketDataClient()
