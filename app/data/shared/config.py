from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Changed case_sensitive to False so it reliably grabs Render environment vars
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # ── Object Storage (MinIO / S3) ──────────────────────────────────────
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str
    MINIO_SECURE: bool = True

    # ── PostgreSQL ────────────────────────────────────────────────────────
    DATABASE_URL: str

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str

    # ── Provider API Keys ─────────────────────────────────────────────────
    NEWSAPI_KEY: Optional[str] = None        
    FRED_API_KEY: Optional[str] = None       

    # ── Authentication ────────────────────────────────────────────────────
    DATA_SERVICE_API_KEY: str 

    # ── Application ───────────────────────────────────────────────────────
    APP_ENV: str = "production"
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

    @model_validator(mode="before")
    @classmethod
    def fix_async_driver_prefix(cls, data: dict) -> dict:
        """Automatically converts Render's postgres:// to postgresql+asyncpg://"""
        url = data.get("DATABASE_URL") or data.get("database_url")
        if url and isinstance(url, str):
            if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            
            # Write it back into the dictionary keys Pydantic expects
            data["DATABASE_URL"] = url
            data["database_url"] = url
        return data


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
