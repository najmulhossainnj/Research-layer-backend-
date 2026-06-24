"""
Alembic migration environment.

Uses the app's async engine settings and imports the model registry so
autogenerate can detect schema changes across all domain models.
"""
import asyncio
from logging.config import fileConfig
import ssl

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import get_settings
from app.db.session import Base
import app.db.models_registry  # noqa: F401  (registers all ORM models)

config = context.config
settings = get_settings()

# 1. Take the Neon URL string
db_url = settings.DATABASE_URL

# 2. Chop off "?sslmode=require" completely so asyncpg doesn't throw a ClientConfigurationError
if "?" in db_url:
    db_url = db_url.split("?")[0]

config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    # 3. Create a clean, verified SSL context that Neon requires to stay open
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"ssl": ctx},  # Inject the secure context safely here
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())