"""
Validation Agent.

Responsibilities (from the spec):
  - Run walk-forward analysis on the assembled strategy
  - Run CPCV if walk-forward passes
  - Write validation results back into ctx
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentContext, AgentResult, AgentRole, BaseAgent


class ValidationAgent(BaseAgent):
    role = AgentRole.VALIDATION

    def __init__(self, db: AsyncSession):
        self._db = db

    async def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        if not ctx.strategy_id or not ctx.feature_ids or not ctx.best_model_plugin:
            return self._fail(
                "Insufficient context for validation",
                ["Requires strategy_id, feature_ids, best_model_plugin"],
            )

        results: dict = {}
        errors: list[str] = []

        # ── Walk-Forward ──────────────────────────────────────────────
        try:
            from app.engines.validation_engine.engine import (
                ValidationEngine, ValidationConfig,
            )
            from app.engines.validation_engine.walk_forward import WalkForwardConfig

            ve = ValidationEngine()
            wf_result = await ve.validate_strategy(
                db=self._db,
                strategy_id=ctx.strategy_id,
                plugin_key=ctx.best_model_plugin,
                model_params=ctx.best_model_params,
                feature_ids=ctx.feature_ids,
                symbol=ctx.symbols[0],
                timeframe=ctx.timeframe,
                start_date=ctx.start_date,
                end_date=ctx.end_date,
                config=ValidationConfig(
                    wf=WalkForwardConfig(n_splits=5, method="rolling"),
                ),
            )
            results["walk_forward"] = {
                "passed": wf_result.passed,
                "aggregate": wf_result.walk_forward.aggregate,
                "overfitting_score": wf_result.walk_forward.overfitting_score,
                "mlflow_run_id": wf_result.mlflow_run_id,
            }
        except Exception as exc:
            errors.append(f"Walk-forward failed: {exc}")
            results["walk_forward"] = {"passed": False, "error": str(exc)}

        # ── CPCV (only if WF passed) ───────────────────────────────────
        wf_passed = results.get("walk_forward", {}).get("passed", False)
        if wf_passed:
            try:
                from app.engines.validation_engine.engine import CPCVValidationEngine
                from app.engines.validation_engine.cpcv import CPCVConfig

                cpcv_engine = CPCVValidationEngine()
                cpcv_result = await cpcv_engine.validate(
                    db=self._db,
                    strategy_id=ctx.strategy_id,
                    plugin_key=ctx.best_model_plugin,
                    model_params=ctx.best_model_params,
                    feature_ids=ctx.feature_ids,
                    symbol=ctx.symbols[0],
                    timeframe=ctx.timeframe,
                    start_date=ctx.start_date,
                    end_date=ctx.end_date,
                    cpcv_config=CPCVConfig(n_splits=6, n_test_splits=2),
                )
                results["cpcv"] = {
                    "passed": cpcv_result.passed,
                    "pbo": cpcv_result.cpcv.pbo,
                    "deflated_sharpe": cpcv_result.cpcv.deflated_sharpe,
                    "aggregate": cpcv_result.cpcv.aggregate,
                    "mlflow_run_id": cpcv_result.mlflow_run_id,
                }
            except Exception as exc:
                errors.append(f"CPCV failed: {exc}")
                results["cpcv"] = {"passed": False, "error": str(exc)}
        else:
            results["cpcv"] = {"skipped": True, "reason": "Walk-forward did not pass"}

        overall_passed = (
            results.get("walk_forward", {}).get("passed", False)
            and results.get("cpcv", {}).get("passed", False)
        )

        summary = (
            f"Validation {'PASSED' if overall_passed else 'FAILED'}. "
            f"WF: {'✓' if results['walk_forward'].get('passed') else '✗'}, "
            f"CPCV: {'✓' if results.get('cpcv', {}).get('passed') else '✗' if not results.get('cpcv', {}).get('skipped') else '—'}"
        )

        return self._ok(
            summary=summary,
            details=results,
            ctx_updates={"validation_results": results},
        ) if not errors else AgentResult(
            role=self.role,
            success=overall_passed,
            summary=summary,
            details=results,
            errors=errors,
            ctx_updates={"validation_results": results},
        )
