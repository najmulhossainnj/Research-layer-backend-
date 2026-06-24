"""
ModelTrainer (Phase 6 — MLflow-integrated).

Trains the final model, records cross-validated metrics, persists the
artifact to S3/MinIO, and logs a fully-lineaged MLflow run — all in one
atomic workflow so every training event is reproducible and auditable.
"""
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.storage import get_storage_client
from app.domain.model.orm import MLModel
from app.engines.model_training_engine.cross_validation import CVConfig
from app.engines.model_training_engine.evaluation import CVResult, evaluate_with_cv
from app.engines.tracking.service import ExperimentTracker
from app.plugins.models import model_registry


class ModelTrainer:
    def __init__(self):
        self._settings = get_settings()
        self._storage = get_storage_client()
        self._tracker = ExperimentTracker()

    async def train(
        self,
        db: AsyncSession,
        model: MLModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv_config: CVConfig | None = None,
        strategy_id: uuid.UUID | None = None,
        feature_version_hashes: list[str] | None = None,
        symbol: str = "",
        timeframe: str = "1d",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> tuple[MLModel, CVResult]:
        from datetime import datetime
        cv_config = cv_config or CVConfig()

        # 1. Cross-validated performance report.
        cv_result = evaluate_with_cv(model.model_type, model.parameters, X, y, cv_config)

        # 2. Fit final model on full dataset.
        plugin_cls = model_registry.get(model.model_type)
        final_model = plugin_cls(**model.parameters)
        final_model.train(X, y)

        # 3. Persist artifact to S3/MinIO.
        artifact_uri, artifact_bytes = self._save_artifact(model, final_model)

        # 4. Update model row.
        model.metrics = cv_result.summary()
        model.artifact_uri = artifact_uri
        db.add(model)
        await db.commit()
        await db.refresh(model)

        # 5. Log to MLflow + create Experiment row.
        if strategy_id:
            await self._tracker.log_training_run(
                db=db,
                strategy_id=strategy_id,
                model=model,
                cv_result=cv_result,
                feature_version_hashes=feature_version_hashes or [],
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date or datetime.utcnow(),
                end_date=end_date or datetime.utcnow(),
                artifact_bytes=artifact_bytes,
            )

        return model, cv_result

    def _save_artifact(self, model: MLModel, trained_plugin) -> tuple[str, bytes]:
        bucket = self._settings.S3_BUCKET_ARTIFACTS
        key = f"models/{model.id}/{uuid.uuid4().hex}.model"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = str(Path(tmpdir) / "artifact.model")
            trained_plugin.save(local_path)
            artifact_bytes = Path(local_path).read_bytes()

        uri = self._storage.put_bytes(bucket, key, artifact_bytes,
                                      content_type="application/octet-stream")
        return uri, artifact_bytes
