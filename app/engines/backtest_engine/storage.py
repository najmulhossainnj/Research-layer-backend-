"""
Backtest result storage.

Persists the two large artifacts (equity curve as parquet, trade list as
parquet) to S3/MinIO and writes the metrics flat-dict into the
`Backtest.metrics` JSON column. Keeps the ORM row small while giving the
UI direct S3 URLs for large time-series downloads.
"""
import io
import uuid

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.storage import get_storage_client
from app.domain.backtest.orm import Backtest, BacktestStatus
from app.engines.backtest_engine.result import BacktestResult


async def persist_backtest_result(
    db: AsyncSession,
    backtest: Backtest,
    result: BacktestResult,
) -> Backtest:
    settings = get_settings()
    storage = get_storage_client()
    bucket = settings.S3_BUCKET_ARTIFACTS
    run_id = uuid.uuid4().hex

    # ── Equity curve → parquet ───────────────────────────────────────
    eq_buf = io.BytesIO()
    result.equity_curve.to_frame(name="equity").to_parquet(eq_buf, index=True)
    equity_key = f"backtests/{backtest.id}/{run_id}/equity_curve.parquet"
    equity_uri = storage.put_bytes(bucket, equity_key, eq_buf.getvalue())

    # ── Trades → parquet ─────────────────────────────────────────────
    trade_buf = io.BytesIO()
    result.trades.to_parquet(trade_buf, index=False)
    trades_key = f"backtests/{backtest.id}/{run_id}/trades.parquet"
    trades_uri = storage.put_bytes(bucket, trades_key, trade_buf.getvalue())

    # ── Drawdowns → parquet ──────────────────────────────────────────
    dd_buf = io.BytesIO()
    result.drawdowns.to_frame(name="drawdown").to_parquet(dd_buf, index=True)
    dd_key = f"backtests/{backtest.id}/{run_id}/drawdowns.parquet"
    storage.put_bytes(bucket, dd_key, dd_buf.getvalue())

    # ── Flatten metrics + update ORM row ─────────────────────────────
    flat = result.metrics.to_flat_dict()
    flat["engine_stats"] = result.engine_stats
    flat["drawdowns_uri"] = f"s3://{bucket}/{dd_key}"

    backtest.status = BacktestStatus.COMPLETED
    backtest.metrics = flat
    backtest.equity_curve_uri = equity_uri
    backtest.trades_uri = trades_uri

    db.add(backtest)
    await db.commit()
    await db.refresh(backtest)
    return backtest


def load_equity_curve(backtest: Backtest) -> pd.Series:
    storage = get_storage_client()
    bucket, key = storage.parse_uri(backtest.equity_curve_uri)
    raw = storage.get_bytes(bucket, key)
    df = pd.read_parquet(io.BytesIO(raw))
    return df["equity"]


def load_trades(backtest: Backtest) -> pd.DataFrame:
    storage = get_storage_client()
    bucket, key = storage.parse_uri(backtest.trades_uri)
    raw = storage.get_bytes(bucket, key)
    return pd.read_parquet(io.BytesIO(raw))
