"""
Feature plugin package. Import each module so plugins self-register.
"""
from app.plugins.base import BaseFeature
from app.plugins.registry import PluginRegistry

feature_registry: PluginRegistry[BaseFeature] = PluginRegistry("feature")

from app.plugins.features import (  # noqa: E402,F401
    automated,
    news_sentiment,
    statistical,
    technical,
)
