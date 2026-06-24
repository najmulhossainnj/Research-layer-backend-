"""
Signal plugin package.
"""
from app.plugins.base import BaseSignalGenerator
from app.plugins.registry import PluginRegistry

signal_registry: PluginRegistry[BaseSignalGenerator] = PluginRegistry("signal")

from app.plugins.signals import (  # noqa: E402,F401
    long_short,
    portfolio,
    ranking,
    threshold,
)
