"""
Unified application configuration.

Centralizes all environment-driven settings for the Unified Quant Research Platform.
Uses pydantic-settings so values can be overridden via environment variables
or a .env file without touching code.

Combines settings from:
- Research Layer: Strategies, Features, Models, Signals, Backtests, Validation
- Data Layer: Market data ingestion and delivery

Local backend: SQLite + DiskCache + APScheduler
"""
import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_default_db_path() -> str:
    """Get default database path in app data directory."""
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(app_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "research_layer.db")


def _get_default_cache_dir() -> str:
    """Get default cache directory."""
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(app_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class Settings(BaseSettings):
    # ── Service metadata ────────────────────────────────────────────
    APP_NAME: str = "Unified Quant Research Platform"
    APP_ENV: str = "local"
    API_V1_PREFIX: str = "/api/v1"

    # ── Database (SQLite) ────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite+aiosqlite:///{_get_default_db_path()}"
    TIMESCALE_URL: Optional[str] = None

    # ── Cache (DiskCache) ───────────────────────────────────────────
    CACHE_DIR: str = _get_default_cache_dir()
    CACHE_TTL_DEFAULT: int = 3600  # 1 hour default TTL

    # ── Background Tasks (APScheduler) ───────────────────────────────
    TASK_THREAD_POOL_SIZE: int = 4
    TASK_MAX_WORKERS: int = 2

    # ── Object storage ───────────────────────────────────────────────
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_ARTIFACTS: str = "research-artifacts"
    S3_BUCKET_FEATURES: str = "feature-store"

    # ── Experiment tracking ───────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "research-layer"
    MLFLOW_ARTIFACT_ROOT: str = "s3://research-artifacts/mlflow"

    # ── Messaging ────────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    EVENT_BACKEND: str = "noop"  # Disabled for local backend

    # ── Security ─────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Data Layer Configuration ─────────────────────────────────────
    # Used when Data Layer runs as separate service
    DATA_SERVICE_URL: str = "http://localhost:8001"
    DATA_SERVICE_API_KEY: str = "dev-api-key-change-in-production"

    # Market Data Providers (for embedded data layer)
    YFINANCE_COUNTRY: str = "US"
    FRED_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None

    # Data Layer Cache TTL (seconds)
    CACHE_TTL_OHLCV_DAILY: int = 86400  # 24 hours
    CACHE_TTL_OHLCV_INTRADAY: int = 3600  # 1 hour
    CACHE_TTL_NEWS: int = 21600  # 6 hours
    CACHE_TTL_FUNDAMENTALS: int = 86400  # 24 hours
    CACHE_TTL_MACRO: int = 259200  # 72 hours

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
