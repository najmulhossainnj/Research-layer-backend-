"""
Signal generation pipeline.

Responsibility: given a trained model artifact (referenced by an `MLModel`
row) and a feature DataFrame, produce a signal series. Supports two paths:

1. **Rule-engine path** (visual Signal Builder):
   Load the `SignalLogic` rule tree and run the recursive evaluator.

2. **Plugin path** (programmatic signal generators):
   Instantiate a `BaseSignalGenerator` plugin and call `.generate()`.

Both paths accept the same inputs and produce the same output shape so the
Backtest Engine doesn't need to know which path was used.
"""
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.model.orm import MLModel
from app.domain.signal.orm import SignalLogic
from app.engines.signal_engine.rule_engine import evaluate_rule_tree
from app.plugins.models import model_registry
from app.plugins.signals import signal_registry


@dataclass
class SignalResult:
    signals: pd.Series          # index-aligned with predictions
    predictions: pd.DataFrame   # raw model output (for audit/display)
    metadata: dict


class SignalPipeline:

    # ------------------------------------------------------------------
    # Rule-engine path
    # ------------------------------------------------------------------

    async def generate_from_rule_tree(
        self,
        db: AsyncSession,
        signal_logic_id,
        model: MLModel,
        feature_data: pd.DataFrame,
    ) -> SignalResult:
        """Load a persisted SignalLogic and evaluate its rule tree."""
        result = await db.execute(
            select(SignalLogic).where(SignalLogic.id == signal_logic_id)
        )
        signal_logic = result.scalar_one_or_none()
        if signal_logic is None:
            raise ValueError(f"SignalLogic {signal_logic_id} not found")

        predictions = self._run_model(model, feature_data)
        signal_input = feature_data.join(predictions, how="left")

        signals = evaluate_rule_tree(
            df=signal_input,
            rule_tree=signal_logic.rule_tree if isinstance(signal_logic.rule_tree, list)
                      else [signal_logic.rule_tree],
            output_mode=signal_logic.output_mode,
            position_mode=signal_logic.position_mode,
        )

        return SignalResult(
            signals=signals,
            predictions=predictions,
            metadata={
                "signal_logic_id": str(signal_logic_id),
                "model_id": str(model.id),
                "output_mode": signal_logic.output_mode,
                "position_mode": signal_logic.position_mode,
                "n_signals": int(signals.nunique()),
            },
        )

    # ------------------------------------------------------------------
    # Plugin path
    # ------------------------------------------------------------------

    async def generate_from_plugin(
        self,
        model: MLModel,
        feature_data: pd.DataFrame,
        plugin_key: str,
        plugin_params: dict,
    ) -> SignalResult:
        """Run a signal generator plugin directly (no DB lookup)."""
        predictions = self._run_model(model, feature_data)

        plugin_cls = signal_registry.get(plugin_key)
        generator = plugin_cls(**plugin_params)
        result_df = generator.generate(predictions)

        signals = result_df["signal"] if "signal" in result_df.columns \
            else pd.Series("HOLD", index=feature_data.index)

        return SignalResult(
            signals=signals,
            predictions=predictions,
            metadata={
                "plugin_key": plugin_key,
                "plugin_params": plugin_params,
                "model_id": str(model.id),
                "n_signals": int(signals.nunique()),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_model(self, model: MLModel, feature_data: pd.DataFrame) -> pd.DataFrame:
        """Load model artifact from plugin and produce a predictions DataFrame."""
        plugin_cls = model_registry.get(model.model_type)
        m = plugin_cls(**model.parameters)
        if model.artifact_uri:
            import tempfile, pathlib
            from app.core.storage import get_storage_client
            storage = get_storage_client()
            bucket, key = storage.parse_uri(model.artifact_uri)
            raw = storage.get_bytes(bucket, key)
            with tempfile.TemporaryDirectory() as tmp:
                p = pathlib.Path(tmp) / "artifact.model"
                p.write_bytes(raw)
                m.load(str(p))
        preds = m.predict(feature_data)
        return pd.DataFrame({"prediction": preds})
