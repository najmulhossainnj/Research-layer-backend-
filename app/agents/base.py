"""
Agent base class and shared message protocol.

Every specialist agent in the Research System is a `BaseAgent` subclass.
Agents communicate through a shared `AgentContext` (mutable research state
passed between agents by the Research Manager) and return structured
`AgentResult` objects that the manager assembles into a full research plan
or conversational response.

Design principles
-----------------
- Agents are stateless; all state lives in `AgentContext`.
- Every agent exposes one primary coroutine: `run(ctx) -> AgentResult`.
- Agents may call platform services (Feature Engine, Model Trainer, etc.)
  directly as async calls — they never block the event loop.
- The Research Manager is the only agent that dispatches to other agents;
  specialist agents never call each other directly.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class AgentRole(str, Enum):
    FEATURE_DISCOVERY   = "feature_discovery"
    MODEL_DISCOVERY     = "model_discovery"
    HYPERPARAMETER      = "hyperparameter"
    BACKTEST            = "backtest"
    VALIDATION          = "validation"
    GOVERNANCE          = "governance"
    RESEARCH_MANAGER    = "research_manager"


@dataclass
class AgentContext:
    """
    Shared research state threaded through all agent calls in a session.
    The Research Manager populates and updates this as agents complete.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = ""

    # Target universe / instrument
    symbols: list[str] = field(default_factory=list)
    timeframe: str = "1d"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Research artefact IDs populated as agents run
    strategy_id: Optional[uuid.UUID] = None
    feature_ids: list[uuid.UUID] = field(default_factory=list)
    model_id: Optional[uuid.UUID] = None
    signal_logic_id: Optional[uuid.UUID] = None
    backtest_ids: list[uuid.UUID] = field(default_factory=list)

    # Intermediate results passed between agents
    candidate_features: list[dict] = field(default_factory=list)   # plugin_key + params
    candidate_models: list[dict] = field(default_factory=list)     # plugin_key + params
    best_model_plugin: Optional[str] = None
    best_model_params: dict = field(default_factory=dict)
    tuning_results: dict = field(default_factory=dict)
    backtest_metrics: dict = field(default_factory=dict)
    validation_results: dict = field(default_factory=dict)
    governance_flags: list[str] = field(default_factory=list)

    # Full conversation log for the AI Researcher chat interface
    messages: list[dict] = field(default_factory=list)

    # Extra kwargs agents can stash arbitrary state
    extra: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    role: AgentRole
    success: bool
    summary: str                         # One-line human-readable summary
    details: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    ctx_updates: dict = field(default_factory=dict)  # Fields to merge back into AgentContext


class BaseAgent(ABC):
    role: AgentRole

    @abstractmethod
    async def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Execute the agent's primary task given the current research context."""
        raise NotImplementedError

    def _ok(self, summary: str, details: dict = None, ctx_updates: dict = None) -> AgentResult:
        return AgentResult(
            role=self.role, success=True, summary=summary,
            details=details or {}, ctx_updates=ctx_updates or {},
        )

    def _fail(self, summary: str, errors: list[str]) -> AgentResult:
        return AgentResult(
            role=self.role, success=False, summary=summary, errors=errors,
        )
