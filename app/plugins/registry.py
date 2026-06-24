"""
Generic plugin registry.

Plugins register themselves with `@registry.register("plugin.key")` at
import time. New plugins are added by dropping a module into the relevant
`app/plugins/<category>/` package and importing it from that package's
`__init__.py` — no core engine code changes needed.
"""
from typing import Callable, Generic, Type, TypeVar

PluginT = TypeVar("PluginT")


class PluginRegistry(Generic[PluginT]):
    def __init__(self, category: str):
        self._category = category
        self._plugins: dict[str, Type[PluginT]] = {}

    def register(self, key: str) -> Callable[[Type[PluginT]], Type[PluginT]]:
        def decorator(cls: Type[PluginT]) -> Type[PluginT]:
            if key in self._plugins:
                raise ValueError(f"Plugin key '{key}' already registered in {self._category}")
            cls.key = key  # type: ignore[attr-defined]
            self._plugins[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> Type[PluginT]:
        if key not in self._plugins:
            raise KeyError(f"No {self._category} plugin registered under key '{key}'")
        return self._plugins[key]

    def list_keys(self) -> list[str]:
        return sorted(self._plugins.keys())

    def create(self, key: str, **params) -> PluginT:
        return self.get(key)(**params)
