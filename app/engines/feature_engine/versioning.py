"""
Versioning utilities for the Feature Store.

A feature dataset's version hash is derived purely from its inputs
(plugin key, params, symbol, timeframe, date range, and a fingerprint of
the underlying source data). Identical inputs always produce the identical
hash, which is what makes regeneration reproducible: re-running the same
pipeline against the same source data yields a byte-identical cache key.
"""
import hashlib
import json
from datetime import datetime
from typing import Any


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def compute_version_hash(
    plugin_key: str,
    params: dict,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    source_fingerprint: str = "",
) -> str:
    payload = _stable_json(
        {
            "plugin_key": plugin_key,
            "params": params,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "source_fingerprint": source_fingerprint,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_source_fingerprint(data_bytes: bytes) -> str:
    """Fingerprint of the raw OHLCV/alt-data slice a feature was computed from.

    Used to detect when upstream market data has been revised (corporate
    actions, restatements, late-arriving data) and a feature needs
    regeneration even though its definition hasn't changed.
    """
    return hashlib.sha256(data_bytes).hexdigest()
