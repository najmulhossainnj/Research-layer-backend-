"""
Research Manager Agent.

Coordinates all specialist agents in the correct dependency order:

  1. Feature Discovery  — candidate features, SHAP pruning
  2. Model Discovery    — AutoML leaderboard, best plugin selected
  3. Hyperparameter     — Optuna tuning of the winning model
  4. Backtest           — train model + execute backtest
  5. Validation         — walk-forward + CPCV
  6. Governance         — overfitting / leakage / stability flags

The manager also:
  - Parses free-text queries ("Build a momentum strategy using RSI and news")
    into structured AgentContext fields
  - Creates the Strategy row at the start of the session
  - Returns a streaming async generator of step updates so the AI
    Researcher chat interface can show real-time progress
  - Assembles a final research report summarising all agent outputs
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.backtest_agent import BacktestAgent
from app.agents.base import AgentContext, AgentResult, AgentRole, BaseAgent
from app.agents.feature_discovery_agent import FeatureDiscoveryAgent
from app.agents.governance_agent import GovernanceAgent
from app.agents.hyperparameter_agent import HyperparameterAgent
from app.agents.model_discovery_agent import ModelDiscoveryAgent
from app.agents.validation_agent import ValidationAgent


def _parse_query(query: str, ctx: AgentContext) -> AgentContext:
    """
    Extract symbols, timeframe, and date-range hints from free-text query.
    Fills ctx in-place; all fields are optional — agents fall back to
    defaults if not found.
    """
    q = query.upper()

    # Symbols: uppercase 1-5 letter words that look like tickers
    tickers = re.findall(r"\b([A-Z]{1,5})\b", q)
    # Filter out common English stop-words that match the pattern
    stop = {"A", "I", "AND", "OR", "THE", "FOR", "ON", "IN", "AT",
            "TO", "OF", "VS", "BY", "RSI", "ATR", "MACD", "USING",
            "WITH", "BUILD", "TEST", "USE", "COMPARE", "AGAINST"}
    symbols = [t for t in tickers if t not in stop]
    if symbols:
        ctx.symbols = symbols[:5]

    # Timeframe
    if "weekly" in query.lower() or "1w" in query.lower():
        ctx.timeframe = "1w"
    elif "hourly" in query.lower() or "1h" in query.lower():
        ctx.timeframe = "1h"
    else:
        ctx.timeframe = "1d"

    # Date range (default last 3 years if not supplied)
    if ctx.end_date is None:
        ctx.end_date = datetime.utcnow()
    if ctx.start_date is None:
        ctx.start_date = ctx.end_date - timedelta(days=3 * 365)

    return ctx


class ResearchManagerAgent(BaseAgent):
    role = AgentRole.RESEARCH_MANAGER

    def __init__(self, db: AsyncSession):
        self._db = db

    async def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """
        Sequential orchestration. For streaming use `run_streaming()`.
        """
        results: dict[str, AgentResult] = {}

        ctx = _parse_query(ctx.user_query, ctx)
        ctx = await self._ensure_strategy(ctx)

        agents: list[BaseAgent] = [
            FeatureDiscoveryAgent(self._db),
            ModelDiscoveryAgent(self._db),
            HyperparameterAgent(self._db),
            BacktestAgent(self._db),
            ValidationAgent(self._db),
            GovernanceAgent(),
        ]

        for agent in agents:
            result = await agent.run(ctx)
            results[agent.role.value] = result
            # Merge ctx_updates back into context
            for k, v in result.ctx_updates.items():
                if hasattr(ctx, k):
                    setattr(ctx, k, v)
            # Stop pipeline on critical governance failures
            if agent.role == AgentRole.GOVERNANCE and not result.success:
                break

        return self._ok(
            summary=self._build_summary(ctx, results),
            details={"agent_results": {k: asdict(v) for k, v in results.items()},
                     "context": self._ctx_snapshot(ctx)},
            ctx_updates={},
        )

    async def run_streaming(
        self, ctx: AgentContext, **kwargs
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator yielding one progress dict per agent completion.
        Consumed by the SSE endpoint in the AI Researcher chat interface.
        """
        ctx = _parse_query(ctx.user_query, ctx)
        ctx = await self._ensure_strategy(ctx)

        yield {"event": "start", "session_id": ctx.session_id,
               "message": f"Research session started. Query: '{ctx.user_query}'",
               "strategy_id": str(ctx.strategy_id)}

        agents: list[BaseAgent] = [
            FeatureDiscoveryAgent(self._db),
            ModelDiscoveryAgent(self._db),
            HyperparameterAgent(self._db),
            BacktestAgent(self._db),
            ValidationAgent(self._db),
            GovernanceAgent(),
        ]

        for agent in agents:
            yield {"event": "agent_start", "role": agent.role.value,
                   "message": f"Running {agent.role.value}…"}

            result = await agent.run(ctx)

            for k, v in result.ctx_updates.items():
                if hasattr(ctx, k):
                    setattr(ctx, k, v)

            yield {
                "event":   "agent_done",
                "role":    agent.role.value,
                "success": result.success,
                "summary": result.summary,
                "details": result.details,
                "errors":  result.errors,
            }

            if agent.role == AgentRole.GOVERNANCE and not result.success:
                yield {"event": "pipeline_halted",
                       "reason": "Critical governance flags raised",
                       "flags": ctx.governance_flags}
                return

        yield {
            "event":    "complete",
            "session_id": ctx.session_id,
            "strategy_id": str(ctx.strategy_id),
            "summary":  self._build_summary(ctx, {}),
            "context":  self._ctx_snapshot(ctx),
        }

    # ── Helpers ────────────────────────────────────────────────────────

    async def _ensure_strategy(self, ctx: AgentContext) -> AgentContext:
        """Create a Strategy row if the session doesn't already have one."""
        if ctx.strategy_id:
            return ctx

        from app.domain.strategy.orm import Strategy
        strategy = Strategy(
            name=f"agent:{ctx.user_query[:60]}",
            description=ctx.user_query,
            universe=ctx.symbols or [],
            timeframe=ctx.timeframe,
        )
        self._db.add(strategy)
        await self._db.commit()
        await self._db.refresh(strategy)
        ctx.strategy_id = strategy.id
        return ctx

    @staticmethod
    def _build_summary(ctx: AgentContext, results: dict) -> str:
        parts = [f"Research session {ctx.session_id[:8]}:"]
        if ctx.symbols:
            parts.append(f"Universe: {', '.join(ctx.symbols)}")
        if ctx.best_model_plugin:
            parts.append(f"Model: {ctx.best_model_plugin}")
        if ctx.validation_results:
            wf = ctx.validation_results.get("walk_forward", {})
            parts.append(f"WF passed: {wf.get('passed', '?')}")
        if ctx.governance_flags:
            parts.append(f"Governance: {len(ctx.governance_flags)} flag(s)")
        return " | ".join(parts)

    @staticmethod
    def _ctx_snapshot(ctx: AgentContext) -> dict:
        return {
            "session_id":        ctx.session_id,
            "symbols":           ctx.symbols,
            "timeframe":         ctx.timeframe,
            "strategy_id":       str(ctx.strategy_id) if ctx.strategy_id else None,
            "feature_ids":       [str(f) for f in ctx.feature_ids],
            "model_id":          str(ctx.model_id) if ctx.model_id else None,
            "best_model_plugin": ctx.best_model_plugin,
            "best_model_params": ctx.best_model_params,
            "backtest_ids":      [str(b) for b in ctx.backtest_ids],
            "governance_flags":  ctx.governance_flags,
            "validation_passed": (
                ctx.validation_results.get("walk_forward", {}).get("passed", False)
                and ctx.validation_results.get("cpcv", {}).get("passed", False)
            ),
        }
