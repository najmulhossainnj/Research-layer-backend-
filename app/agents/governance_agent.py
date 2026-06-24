"""
Governance Agent.

Responsibilities (from the spec):
  - Detect overfitting (IS/OOS performance gap)
  - Detect data leakage (future-peeking features, look-ahead bias)
  - Detect unstable parameters (high sensitivity to small param changes)
  - Write governance_flags into ctx; block promotion if critical flags raised
"""
from __future__ import annotations

import numpy as np

from app.agents.base import AgentContext, AgentResult, AgentRole, BaseAgent

# Severity levels for flags
_CRITICAL = "CRITICAL"
_WARNING  = "WARNING"
_INFO     = "INFO"


class GovernanceAgent(BaseAgent):
    role = AgentRole.GOVERNANCE

    async def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        flags: list[str] = []
        details: dict = {}

        # ── 1. Overfitting detection ──────────────────────────────────
        of_flags, of_details = self._check_overfitting(ctx)
        flags.extend(of_flags)
        details["overfitting"] = of_details

        # ── 2. Leakage detection ──────────────────────────────────────
        lk_flags, lk_details = self._check_leakage(ctx)
        flags.extend(lk_flags)
        details["leakage"] = lk_details

        # ── 3. Parameter stability ────────────────────────────────────
        ps_flags, ps_details = self._check_parameter_stability(ctx)
        flags.extend(ps_flags)
        details["parameter_stability"] = ps_details

        # ── 4. Minimum sample size ────────────────────────────────────
        ms_flags, ms_details = self._check_sample_size(ctx)
        flags.extend(ms_flags)
        details["sample_size"] = ms_details

        n_critical = sum(1 for f in flags if _CRITICAL in f)
        n_warning  = sum(1 for f in flags if _WARNING in f)

        summary = (
            f"Governance check: {n_critical} critical flag(s), "
            f"{n_warning} warning(s), {len(flags) - n_critical - n_warning} info."
        )
        if not flags:
            summary = "Governance check: no issues detected. ✓"

        return AgentResult(
            role=self.role,
            success=n_critical == 0,
            summary=summary,
            details=details,
            errors=[f for f in flags if _CRITICAL in f],
            ctx_updates={"governance_flags": flags},
        )

    # ── Overfitting ────────────────────────────────────────────────────

    def _check_overfitting(self, ctx: AgentContext) -> tuple[list[str], dict]:
        flags: list[str] = []
        details: dict = {}

        wf = ctx.validation_results.get("walk_forward", {})
        agg = wf.get("aggregate", {})

        overfit_score = wf.get("overfitting_score", None)
        if overfit_score is not None:
            details["overfitting_score"] = overfit_score
            if overfit_score > 5.0:
                flags.append(
                    f"[{_CRITICAL}] Severe overfitting: IS/OOS Sharpe ratio = "
                    f"{overfit_score:.2f} (threshold 5.0)"
                )
            elif overfit_score > 3.0:
                flags.append(
                    f"[{_WARNING}] Moderate overfitting: IS/OOS ratio = {overfit_score:.2f}"
                )

        cpcv = ctx.validation_results.get("cpcv", {})
        pbo = cpcv.get("pbo", None)
        if pbo is not None:
            details["pbo"] = pbo
            if pbo > 0.75:
                flags.append(
                    f"[{_CRITICAL}] High Probability of Backtest Overfitting: "
                    f"PBO = {pbo:.2%}"
                )
            elif pbo > 0.55:
                flags.append(f"[{_WARNING}] Elevated PBO = {pbo:.2%}")

        mean_oos = agg.get("mean_oos_sharpe", None)
        if mean_oos is not None and mean_oos < 0:
            flags.append(
                f"[{_CRITICAL}] Negative mean OOS Sharpe ({mean_oos:.3f}): "
                "strategy likely overfits in-sample."
            )

        return flags, details

    # ── Leakage ────────────────────────────────────────────────────────

    def _check_leakage(self, ctx: AgentContext) -> tuple[list[str], dict]:
        flags: list[str] = []
        details: dict = {}

        # Check that a gap was used in CPCV config
        cpcv_data = ctx.validation_results.get("cpcv", {})
        agg = cpcv_data.get("aggregate", {})

        # Heuristic: if OOS Sharpe is suspiciously high, flag for manual review
        mean_oos = agg.get("mean_oos_sharpe", None)
        if mean_oos is not None and mean_oos > 4.0:
            flags.append(
                f"[{_WARNING}] Unusually high OOS Sharpe ({mean_oos:.2f}). "
                "Verify no look-ahead bias in feature computation."
            )
            details["suspicious_oos_sharpe"] = mean_oos

        # Check all feature plugin keys for known leaky patterns
        leaky_patterns = ["future", "forward", "next", "lead"]
        for feat in ctx.candidate_features:
            key = feat.get("plugin_key", "")
            if any(p in key.lower() for p in leaky_patterns):
                flags.append(
                    f"[{_CRITICAL}] Feature '{key}' may use future information."
                )
                details["leaky_feature"] = key

        # Check target horizon vs embargo
        if not ctx.validation_results.get("cpcv", {}).get("skipped"):
            details["embargo_check"] = "CPCV embargo applied"
        else:
            flags.append(
                f"[{_INFO}] CPCV was skipped; embargo leakage check not performed."
            )

        return flags, details

    # ── Parameter stability ────────────────────────────────────────────

    def _check_parameter_stability(self, ctx: AgentContext) -> tuple[list[str], dict]:
        flags: list[str] = []
        details: dict = {}

        tuning = ctx.tuning_results
        if not tuning or tuning.get("skipped"):
            details["status"] = "Tuning was skipped; stability check not applicable."
            return flags, details

        best_params = ctx.best_model_params
        details["best_params"] = best_params

        # Heuristic: learning_rate near bounds is a stability warning
        lr = best_params.get("learning_rate")
        if lr is not None:
            if lr < 0.005:
                flags.append(
                    f"[{_WARNING}] learning_rate={lr} is very small — model may be "
                    "slow to adapt or tuning hit the lower bound."
                )
            elif lr > 0.25:
                flags.append(
                    f"[{_WARNING}] learning_rate={lr} is large — model may be unstable."
                )

        # High max_depth often correlates with overfitting
        depth = best_params.get("max_depth") or best_params.get("depth")
        if depth is not None and depth >= 10:
            flags.append(
                f"[{_WARNING}] Tree depth={depth} is high — increased overfitting risk."
            )

        wf_folds = ctx.validation_results.get("walk_forward", {}).get("aggregate", {})
        std_sharpe = wf_folds.get("std_oos_sharpe", None)
        mean_sharpe = wf_folds.get("mean_oos_sharpe", None)
        if std_sharpe is not None and mean_sharpe and abs(mean_sharpe) > 1e-6:
            cv_ratio = std_sharpe / abs(mean_sharpe)
            details["sharpe_cv_ratio"] = round(cv_ratio, 3)
            if cv_ratio > 1.0:
                flags.append(
                    f"[{_WARNING}] High OOS Sharpe variance across folds "
                    f"(CV ratio={cv_ratio:.2f}): strategy performance is unstable."
                )

        return flags, details

    # ── Sample size ────────────────────────────────────────────────────

    def _check_sample_size(self, ctx: AgentContext) -> tuple[list[str], dict]:
        flags: list[str] = []
        details: dict = {}

        if ctx.start_date and ctx.end_date:
            days = (ctx.end_date - ctx.start_date).days
            details["dataset_days"] = days
            if days < 252:
                flags.append(
                    f"[{_WARNING}] Dataset spans only {days} days (<1 year). "
                    "Results may not be statistically significant."
                )
            if days < 60:
                flags.append(
                    f"[{_CRITICAL}] Dataset too short ({days} days) for reliable "
                    "walk-forward validation."
                )

        n_features = len(ctx.feature_ids)
        details["n_features"] = n_features
        if n_features > 50:
            flags.append(
                f"[{_WARNING}] {n_features} features — high dimensionality increases "
                "overfitting risk. Consider feature selection."
            )

        return flags, details
