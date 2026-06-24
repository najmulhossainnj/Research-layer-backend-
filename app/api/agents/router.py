"""
AI Researcher endpoints.

  POST /agents/research        — run full pipeline, return final report
  POST /agents/research/stream — SSE stream of per-agent progress events
  POST /agents/chat            — single-turn query routed to the manager
  GET  /agents/sessions/{id}   — retrieve a past session context snapshot
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentContext
from app.agents.research_manager import ResearchManagerAgent
from app.db.session import get_db

router = APIRouter(prefix="/agents", tags=["agents"])

# In-process session store (replace with Redis in production)
_sessions: dict[str, dict] = {}


class ResearchRequest(BaseModel):
    query: str = Field(..., description=(
        "Natural language research instruction, e.g. "
        "'Build a momentum strategy using RSI and news sentiment on AAPL'"
    ))
    symbols: list[str] = Field(default_factory=list)
    timeframe: str = "1d"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    strategy_id: Optional[str] = None


def _build_context(payload: ResearchRequest) -> AgentContext:
    ctx = AgentContext(user_query=payload.query)
    if payload.symbols:
        ctx.symbols = payload.symbols
    ctx.timeframe = payload.timeframe
    ctx.start_date = payload.start_date
    ctx.end_date = payload.end_date
    if payload.strategy_id:
        import uuid
        ctx.strategy_id = uuid.UUID(payload.strategy_id)
    return ctx


# ── Synchronous research run ──────────────────────────────────────────────

@router.post("/research")
async def run_research(payload: ResearchRequest, db: AsyncSession = Depends(get_db)):
    """
    Run the full research pipeline synchronously.
    For long runs use /research/stream instead.
    """
    ctx = _build_context(payload)
    manager = ResearchManagerAgent(db)
    result = await manager.run(ctx)

    snapshot = result.details.get("context", {})
    _sessions[ctx.session_id] = snapshot

    return {
        "session_id": ctx.session_id,
        "success":    result.success,
        "summary":    result.summary,
        "details":    result.details,
        "errors":     result.errors,
    }


# ── SSE streaming research run ────────────────────────────────────────────

@router.post("/research/stream")
async def stream_research(payload: ResearchRequest, db: AsyncSession = Depends(get_db)):
    """
    Stream per-agent progress as Server-Sent Events.
    Each event is a JSON object: {event, role, summary, details, …}.
    """
    ctx = _build_context(payload)

    async def _sse_generator() -> AsyncGenerator[str, None]:
        manager = ResearchManagerAgent(db)
        async for update in manager.run_streaming(ctx):
            data = json.dumps(update, default=str)
            yield f"data: {data}\n\n"
            await asyncio.sleep(0)  # yield control to the event loop

        # Store final snapshot
        _sessions[ctx.session_id] = {
            "session_id":   ctx.session_id,
            "strategy_id":  str(ctx.strategy_id) if ctx.strategy_id else None,
            "symbols":      ctx.symbols,
            "feature_ids":  [str(f) for f in ctx.feature_ids],
            "model_id":     str(ctx.model_id) if ctx.model_id else None,
            "best_model":   ctx.best_model_plugin,
            "governance":   ctx.governance_flags,
        }

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Single-turn chat (routes to manager without full pipeline) ────────────

class ChatMessage(BaseModel):
    query: str
    session_id: Optional[str] = None
    context_override: dict = Field(default_factory=dict)


@router.post("/chat")
async def chat(payload: ChatMessage, db: AsyncSession = Depends(get_db)):
    """
    Single-turn AI Researcher chat. If a session_id is provided the
    existing context is rehydrated so follow-up questions work correctly.
    """
    ctx = AgentContext(user_query=payload.query)

    # Rehydrate from previous session if supplied
    if payload.session_id and payload.session_id in _sessions:
        prev = _sessions[payload.session_id]
        import uuid
        ctx.symbols     = prev.get("symbols", [])
        ctx.feature_ids = [uuid.UUID(f) for f in prev.get("feature_ids", [])]
        if prev.get("strategy_id"):
            ctx.strategy_id = uuid.UUID(prev["strategy_id"])
        if prev.get("model_id"):
            ctx.model_id = uuid.UUID(prev["model_id"])
        ctx.best_model_plugin = prev.get("best_model")
        ctx.governance_flags  = prev.get("governance", [])

    # Override with any explicit context fields
    for k, v in payload.context_override.items():
        if hasattr(ctx, k):
            setattr(ctx, k, v)

    # Route simple queries via manager (no full pipeline)
    manager = ResearchManagerAgent(db)
    result = await manager.run(ctx)
    _sessions[ctx.session_id] = result.details.get("context", {})

    return {
        "session_id": ctx.session_id,
        "reply":      result.summary,
        "details":    result.details,
        "success":    result.success,
    }


# ── Session retrieval ─────────────────────────────────────────────────────

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    snapshot = _sessions.get(session_id)
    if not snapshot:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")
    return snapshot


@router.get("/sessions")
async def list_sessions():
    return {"sessions": list(_sessions.keys()), "total": len(_sessions)}
