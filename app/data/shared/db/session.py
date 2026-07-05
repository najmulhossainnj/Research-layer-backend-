from contextlib import asynccontextmanager
from typing import AsyncGenerator
import ssl
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy import event  # Added standard event module
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from shared.config import settings

# 1. Clean 'sslmode' from the connection string securely
db_url = settings.DATABASE_URL
has_ssl = "sslmode" in db_url

if has_ssl:
    parsed_url = urlparse(db_url)
    query_params = parse_qsl(parsed_url.query)
    # Strip any sslmode key out completely
    filtered_params = [param for param in query_params if param[0] != 'sslmode']
    new_query = urlencode(filtered_params)
    db_url = urlunparse(parsed_url._replace(query=new_query))

# 2. Re-inject safe secure connection parameters
connect_args = {}
if has_ssl or "render.com" in db_url or "neon.tech" in db_url:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    connect_args["ssl"] = ctx

# 3. Create engine with your pool settings
engine = create_async_engine(
    db_url,
    echo=settings.APP_ENV == "development",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,          
    connect_args=connect_args,   
)

# 4. Deep fallback safety check
# We bind the listener to engine.sync_engine because AsyncEngine doesn't have an events attribute
def receive_do_connect(dialect, conn_rec, cargs, cparams):
    cparams.pop("sslmode", None)

event.listen(engine.sync_engine, "do_connect", receive_do_connect)


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
