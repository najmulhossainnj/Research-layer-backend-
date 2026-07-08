from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from shared.config import settings

# Build SQLite connection URL from path
db_url = f"sqlite+aiosqlite:///{settings.SQLITE_PATH}"

# SQLite-specific: disable thread checking
connect_args = {"check_same_thread": False}

# Create engine with pool settings
engine = create_async_engine(
    db_url,
    echo=settings.APP_ENV == "development",
    connect_args=connect_args,
)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager that yields a DB session and handles commit / rollback."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a session."""
    async with AsyncSessionLocal() as session:
        yield session
