"""
Rule engine.

Evaluates the JSON rule tree produced by the Signal Builder UI against a
pandas DataFrame, producing a signal series. Supports:

  - Operators: >, <, >=, <=, ==, !=
  - Combinators: AND, OR (arbitrarily nested)
  - Multiple rule groups each mapping to a different action (BUY/SELL/HOLD)
  - First-match-wins semantics when multiple groups can fire on the same row

Rule tree schema
----------------
A SignalLogic.rule_tree is a list of "rule groups", each structured as:

    {
        "action": "BUY",          # BUY | SELL | HOLD
        "combinator": "AND",      # AND | OR
        "rules": [
            {"field": "prediction", "operator": ">", "value": 0.7},
            {
                "combinator": "OR",
                "rules": [
                    {"field": "sentiment", "operator": ">", "value": 0.5},
                    {"field": "rsi",       "operator": "<", "value": 30}
                ]
            }
        ]
    }

A leaf rule node must have `field`, `operator`, and `value`.
A branch node must have `combinator` and `rules` (no `field`).
"""
import operator as op
from typing import Any

import pandas as pd

_OPS: dict[str, Any] = {
    ">":  op.gt,
    "<":  op.lt,
    ">=": op.ge,
    "<=": op.le,
    "==": op.eq,
    "!=": op.ne,
}


def _eval_leaf(row: pd.Series, rule: dict) -> bool:
    field = rule["field"]
    operator = rule["operator"]
    value = rule["value"]
    if field not in row:
        raise KeyError(
            f"Rule references field '{field}' which is not present in the signal "
            f"DataFrame. Available columns: {list(row.index)}"
        )
    fn = _OPS.get(operator)
    if fn is None:
        raise ValueError(f"Unsupported operator '{operator}'. Choose from: {list(_OPS)}")
    return bool(fn(row[field], value))


def _eval_node(row: pd.Series, node: dict) -> bool:
    """Recursively evaluate a rule node (leaf or branch)."""
    if "combinator" not in node:
        # Leaf node
        return _eval_leaf(row, node)

    combinator = node["combinator"].upper()
    children = node.get("rules", [])

    if combinator == "AND":
        return all(_eval_node(row, child) for child in children)
    if combinator == "OR":
        return any(_eval_node(row, child) for child in children)

    raise ValueError(f"Unsupported combinator '{combinator}'. Use AND or OR.")


def evaluate_rule_tree(
    df: pd.DataFrame,
    rule_tree: list[dict],
    default_action: str = "HOLD",
    output_mode: str = "discrete",
    position_mode: str = "long_short",
) -> pd.Series:
    """
    Apply the rule tree to every row of `df` and return a signal Series.

    Parameters
    ----------
    df            : DataFrame containing model predictions + any feature columns
                    referenced in rule conditions.
    rule_tree     : List of rule-group dicts (each has `action`, `combinator`,
                    `rules`). First match per row wins.
    default_action: Emitted when no rule group fires. Default: "HOLD".
    output_mode   : "discrete" → "BUY"/"SELL"/"HOLD"
                    "numeric"  → +1 / 0 / -1
    position_mode : "long_only"  → SELL signals suppressed (become HOLD)
                    "long_short" → both directions allowed
    """
    _ACTION_TO_NUMERIC = {"BUY": 1, "HOLD": 0, "SELL": -1}
    signals = []

    for _, row in df.iterrows():
        action = default_action
        for group in rule_tree:
            node = {
                "combinator": group.get("combinator", "AND"),
                "rules": group.get("rules", []),
            }
            if _eval_node(row, node):
                action = group.get("action", default_action)
                break  # first-match wins

        if position_mode == "long_only" and action == "SELL":
            action = "HOLD"

        if output_mode == "numeric":
            signals.append(_ACTION_TO_NUMERIC.get(action, 0))
        else:
            signals.append(action)

    return pd.Series(signals, index=df.index, name="signal")
