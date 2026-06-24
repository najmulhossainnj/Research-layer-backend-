"""
Schemas for SignalLogic CRUD and signal generation.
"""
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Rule tree node schemas (used by the Signal Builder UI) ────────────────

class LeafRule(BaseModel):
    """A single condition: field operator value."""
    field: str
    operator: Literal[">", "<", ">=", "<=", "==", "!="]
    value: float | str


class BranchRule(BaseModel):
    """A combinator node that groups child rules."""
    combinator: Literal["AND", "OR"]
    rules: list[Any]  # list[LeafRule | BranchRule] — Pydantic v2 recursive


class RuleGroup(BaseModel):
    """Top-level rule group: combinator + rules → action."""
    action: Literal["BUY", "SELL", "HOLD"] = "BUY"
    combinator: Literal["AND", "OR"] = "AND"
    rules: list[Any]  # list[LeafRule | BranchRule]


# ── SignalLogic CRUD schemas ───────────────────────────────────────────────

class SignalLogicCreate(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    rule_tree: list[dict] = Field(
        ...,
        description="List of RuleGroup dicts produced by the Signal Builder UI",
    )
    output_mode: Literal["discrete", "numeric"] = "discrete"
    position_mode: Literal["long_only", "long_short", "portfolio"] = "long_short"
    strategy_id: Optional[uuid.UUID] = None


class SignalLogicUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    rule_tree: Optional[list[dict]] = None
    output_mode: Optional[str] = None
    position_mode: Optional[str] = None
    strategy_id: Optional[uuid.UUID] = None


class SignalLogicRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: Optional[str] = None
    rule_tree: list[dict]
    output_mode: str
    position_mode: str
    strategy_id: Optional[uuid.UUID] = None
    version: int
    created_at: datetime
    updated_at: datetime


# ── Signal generation request/response ───────────────────────────────────

class SignalGenerateRequest(BaseModel):
    """Generate signals for a symbol/range using a trained model + rule tree."""
    model_id: uuid.UUID
    feature_ids: list[uuid.UUID]
    signal_logic_id: Optional[uuid.UUID] = Field(
        None,
        description="Use a persisted rule tree. Mutually exclusive with plugin_key.",
    )
    plugin_key: Optional[str] = Field(
        None,
        description="Use a signal generator plugin directly. Mutually exclusive with signal_logic_id.",
    )
    plugin_params: dict = Field(default_factory=dict)
    symbol: str
    timeframe: str = "1d"
    start_date: datetime
    end_date: datetime
    target_horizon: int = 1


class SignalSummary(BaseModel):
    total_bars: int
    buy_count: int
    sell_count: int
    hold_count: int
    signal_rate: float  # fraction of bars with non-HOLD signal


class SignalGenerateResponse(BaseModel):
    signal_logic_id: Optional[uuid.UUID]
    plugin_key: Optional[str]
    model_id: uuid.UUID
    symbol: str
    metadata: dict
    summary: SignalSummary
    preview: list[dict] = Field(
        default_factory=list,
        description="First 20 rows of (timestamp, prediction, signal)",
    )
