"""
Import hub for all ORM models.

Alembic's env.py and any `Base.metadata.create_all()` bootstrap code should
import this module so every mapped class is registered before metadata
operations run.
"""
from app.domain.backtest.orm import Backtest  # noqa: F401
from app.domain.feature.dataset_orm import FeatureDataset  # noqa: F401
from app.domain.feature.orm import Feature  # noqa: F401
from app.domain.model.experiment_orm import Experiment  # noqa: F401
from app.domain.model.orm import MLModel  # noqa: F401
from app.domain.signal.orm import SignalLogic  # noqa: F401
from app.domain.strategy.orm import Strategy  # noqa: F401
