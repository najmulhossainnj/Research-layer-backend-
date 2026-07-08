from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import os
from typing import Optional


def _get_default_db_path() -> str:
    """Get default database path in app data directory."""
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(app_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "research_layer.db")


def _get_default_cache_dir() -> str:
    """Get default cache directory."""
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cache_dir = os.path.join(app_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class Settings(BaseSettings):
    # Changed case_sensitive to False so it reliably grabs Render environment vars
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # ── Object Storage (MinIO / S3) ──────────────────────────────────────
    MINIO_ENDPOINT: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "market-data"
    MINIO_SECURE: bool = False

    # ── Database (SQLite) ────────────────────────────────────────────────────────
    SQLITE_PATH: str = _get_default_db_path()

    # ── Cache (DiskCache) ──────────────────────────────────────────────
    CACHE_DIR: str = _get_default_cache_dir()

    # ── Provider API Keys ─────────────────────────────────────────────────
    NEWSAPI_KEY: Optional[str] = None        
    FRED_API_KEY: Optional[str] = None       

    # ── Authentication ────────────────────────────────────────────────────
    DATA_SERVICE_API_KEY: str = "dev-api-key-change-in-production"

    # ── Application ───────────────────────────────────────────────────────
    APP_ENV: str = "local"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8001

    # ── Circuit Breaker ───────────────────────────────────────────────────
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RESET_TIMEOUT: int = 60

    # ── Cache TTLs (seconds) ──────────────────────────────────────────────
    CACHE_TTL_OHLCV_DAILY: int = 86_400       
    CACHE_TTL_OHLCV_INTRADAY: int = 3_600     
    CACHE_TTL_NEWS: int = 21_600              
    CACHE_TTL_FUNDAMENTALS: int = 86_400      
    CACHE_TTL_MACRO: int = 259_200            


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
