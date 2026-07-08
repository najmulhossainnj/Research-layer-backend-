"""
YahooProvider — wraps yfinance for OHLCV and fundamentals.

Design decisions:
- Every yfinance call is wrapped in asyncio.to_thread (blocking SDK).
- Retry: exponential back-off via tenacity, max 3 attempts.
- Circuit breaker: failure count stored in memory. If >= threshold, raises
  ProviderUnavailableError immediately without calling the upstream.
- Never call yf.download() with multiple symbols — one symbol per call.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ingestion.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from shared.config import settings

logger = logging.getLogger(__name__)

# Timeframe mapping: Research Layer string → yfinance interval
_TIMEFRAME_MAP: dict[str, str] = {
    "1m":  "1mo",   # monthly
    "1w":  "1wk",
    "1d":  "1d",
    "1h":  "1h",
    "30m": "30m",
    "15m": "15m",
    "5m":  "5m",
}

# Fundamental field mapping: yfinance key → canonical field name
_FUNDAMENTAL_MAP: dict[str, str] = {
    "trailingPE":        "pe_ratio",
    "priceToBook":       "pb_ratio",
    "revenueGrowth":     "revenue_growth",
    "earningsSurprise":  "earnings_surprise",   # note: not always present
    "marketCap":         "market_cap",
    "trailingEps":       "eps",
}

_CB_KEY_PREFIX = "cb:yahoo:"


class InMemoryCircuitBreaker:
    """Thread-safe in-memory circuit breaker for local mode."""
    
    def __init__(self, threshold: int = 5, reset_timeout: int = 60):
        self._failures: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()
        self._threshold = threshold
        self._reset_timeout = reset_timeout
    
    def is_open(self, key: str) -> bool:
        with self._lock:
            if key not in self._failures:
                return False
            count, timestamp = self._failures[key]
            # Check if reset timeout has passed
            if time.time() - timestamp > self._reset_timeout:
                del self._failures[key]
                return False
            return count >= self._threshold
    
    def record_failure(self, key: str) -> None:
        with self._lock:
            if key in self._failures:
                count, _ = self._failures[key]
                self._failures[key] = (count + 1, time.time())
            else:
                self._failures[key] = (1, time.time())
    
    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
    
    def get_failure_count(self, key: str) -> int:
        with self._lock:
            if key not in self._failures:
                return 0
            count, timestamp = self._failures[key]
            # Check if reset timeout has passed
            if time.time() - timestamp > self._reset_timeout:
                del self._failures[key]
                return 0
            return count


# Global circuit breaker instance
_circuit_breaker = InMemoryCircuitBreaker(
    threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
    reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT
)


class YahooProvider(BaseProvider):
    name = "yahoo"
    supported_timeframes = list(_TIMEFRAME_MAP.keys())

    def __init__(self) -> None:
        # Initialize a persistent requests session with proper browser headers
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ── Circuit breaker ────────────────────────────────────────────────────

    async def _check_circuit(self, key: str) -> None:
        if _circuit_breaker.is_open(key):
            count = _circuit_breaker.get_failure_count(key)
            raise ProviderUnavailableError(
                f"Yahoo Finance circuit breaker open for key={key}. "
                f"Failures={count} >= threshold={settings.CIRCUIT_BREAKER_THRESHOLD}"
            )

    async def _record_failure(self, key: str) -> None:
        _circuit_breaker.record_failure(key)

    async def _reset_circuit(self, key: str) -> None:
        _circuit_breaker.reset(key)

    # ── OHLCV ─────────────────────────────────────────────────────────────

    async def download_ohlcv(
        self, symbol: str, timeframe: str, start: str, end: str
    ) -> pd.DataFrame:
        cb_key = f"{_CB_KEY_PREFIX}ohlcv:{symbol}"
        await self._check_circuit(cb_key)

        yf_interval = _TIMEFRAME_MAP.get(timeframe)
        if yf_interval is None:
            raise ProviderError(f"Unsupported timeframe '{timeframe}' for YahooProvider")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _fetch() -> pd.DataFrame:
            # Injecting custom requests session to bypass cloud hosting blocks
            ticker = yf.Ticker(symbol, session=self._session)
            df = ticker.history(
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True,
                actions=False,
            )
            return df

        try:
            df = await asyncio.to_thread(_fetch)
            await self._reset_circuit(cb_key)
            logger.info("YahooProvider.download_ohlcv: %s %s rows=%d", symbol, timeframe, len(df))
            return df
        except Exception as exc:
            await self._record_failure(cb_key)
            logger.error("YahooProvider.download_ohlcv failed: %s — %s", symbol, exc)
            raise ProviderError(f"Yahoo Finance OHLCV fetch failed for {symbol}: {exc}") from exc

    # ── News ──────────────────────────────────────────────────────────────

    async def download_news(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError(
            "YahooProvider does not implement news. Use NewsProvider."
        )

    # ── Fundamentals ──────────────────────────────────────────────────────

    async def download_fundamentals(self, symbol: str) -> dict[str, Any]:
        cb_key = f"{_CB_KEY_PREFIX}fundamentals:{symbol}"
        await self._check_circuit(cb_key)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _fetch() -> dict[str, Any]:
            # Injecting custom requests session here as well
            ticker = yf.Ticker(symbol, session=self._session)
            info = ticker.info or {}
            result: dict[str, Any] = {"symbol": symbol}
            for yf_key, canonical_key in _FUNDAMENTAL_MAP.items():
                val = info.get(yf_key)
                if val is not None:
                    result[canonical_key] = float(val)
            result["as_of"] = datetime.now(timezone.utc).date().isoformat()
            return result

        try:
            data = await asyncio.to_thread(_fetch)
            await self._reset_circuit(cb_key)
            return data
        except Exception as exc:
            await self._record_failure(cb_key)
            raise ProviderError(
                f"Yahoo Finance fundamentals fetch failed for {symbol}: {exc}"
            ) from exc

    # ── Macro ─────────────────────────────────────────────────────────────

    async def download_macro(self, series: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError(
            "YahooProvider does not implement macro. Use FREDProvider."
        )

    # ── Metadata ──────────────────────────────────────────────────────────

    async def symbols(self) -> list[str]:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]

    async def health(self) -> bool:
        try:
            df = await asyncio.to_thread(
                lambda: yf.Ticker("AAPL", session=self._session).history(period="1d", interval="1d")
            )
            return not df.empty
        except Exception:
            return False
