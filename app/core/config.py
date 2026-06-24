"""
Application configuration.

Centralizes all environment-driven settings for the Research Layer service.
Uses pydantic-settings so values can be overridden via environment variables
or a .env file without touching code.
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Service metadata ---
    APP_NAME: str = "Research Layer"
    APP_ENV: str = "local"
    API_V1_PREFIX: str = "/api/v1"

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/research_layer"
    TIMESCALE_URL: Optional[str] = None

    # --- Cache / broker ---
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # --- Object storage ---
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_ARTIFACTS: str = "research-artifacts"
    S3_BUCKET_FEATURES: str = "feature-store"

    # --- Experiment tracking ---
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "research-layer"
    MLFLOW_ARTIFACT_ROOT: str = "s3://research-artifacts/mlflow"

    # --- External layer integrations ---
    MARKET_DATA_URL: str = "http://localhost:8001"

    # --- Messaging ---
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    EVENT_BACKEND: str = "kafka"  # "kafka" | "nats" | "noop"

    # --- Security ---
    SECRET_KEY: str = "change-me-in-production"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
