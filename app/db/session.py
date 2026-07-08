"""
Async SQLAlchemy engine and session management for SQLite.

Uses aiosqlite for async SQLite support.
"""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

settings = get_settings()

# Build SQLite URL from path directly
db_url = f"sqlite+aiosqlite:///{settings.SQLITE_PATH}"

# SQLite: NullPool for async compatibility, disable thread checking
engine = create_async_engine(
    db_url,
    echo=False,
    future=True,
    poolclass=NullPool,
    connect_args={"check_same_thread": False},
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


async def init_db() -> None:
    """Initialize the database by creating all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close the database engine."""
    await engine.dispose()