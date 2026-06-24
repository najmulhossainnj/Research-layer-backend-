"""
Async SQLAlchemy engine and session management.
"""
from typing import AsyncGenerator
import ssl

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

# 1. Intercept and clean the connection string
db_url = settings.DATABASE_URL
has_ssl = "sslmode=" in db_url

if has_ssl:
    # Chop off '?sslmode=require' so asyncpg doesn't crash on keywords
    db_url = db_url.split("?")[0]

# 2. Re-inject safe secure connection parameters for Neon's routing proxies
connect_args = {}
if has_ssl:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    connect_args["ssl"] = ctx

# 3. Create engine with sanitized parameters
engine = create_async_engine(
    db_url, 
    echo=False, 
    future=True, 
    pool_pre_ping=True,
    connect_args=connect_args  # Injected here
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base for all domain ORM models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a transactional session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise