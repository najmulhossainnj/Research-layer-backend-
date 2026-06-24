"""
Backtest engine registry.

Maps engine key strings (stored in `Backtest.engine`) to their adapter
classes. The factory instantiates the right adapter with a config dict,
keeping the execution pipeline free of any engine-specific import logic.
"""
from app.engines.backtest_engine.backtrader_adapter import BacktraderAdapter
from app.engines.backtest_engine.vectorbt_adapter import VectorBTAdapter
from app.plugins.base import BaseBacktestEngine

_REGISTRY: dict[str, type[BaseBacktestEngine]] = {
    VectorBTAdapter.key: VectorBTAdapter,
    BacktraderAdapter.key: BacktraderAdapter,
}

# Canonical key aliases that match Backtest.engine DB values
_ALIASES: dict[str, str] = {
    "vectorbt":   VectorBTAdapter.key,
    "backtrader": BacktraderAdapter.key,
}


def get_engine(engine_key: str, config: dict) -> BaseBacktestEngine:
    resolved = _ALIASES.get(engine_key, engine_key)
    cls = _REGISTRY.get(resolved)
    if cls is None:
        available = list(_REGISTRY) + list(_ALIASES)
        raise ValueError(
            f"Unknown backtest engine '{engine_key}'. Available: {available}"
        )
    return cls(**config)


def list_engines() -> list[str]:
    return sorted(_ALIASES)
