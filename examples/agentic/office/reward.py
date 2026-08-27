"""Deterministic artifact reward for the Office agentic demo."""

from __future__ import annotations

import json
from typing import Any

NON_OK_TURN_DISCOUNT = 0.95


def reward_fn(record) -> float:
    """Score the final graded artifact and discount unsuccessful grader turns."""

    graded_turns = []
    for result in record.tool_results:
        content = _decode_content(result.get("content"))
        if any(key in content for key in ("artifact_score", "progress_score", "artifact_issues")):
            graded_turns.append(content)
    if not graded_turns:
        return 0.0

    final_score = max(_bounded_score(graded_turns[-1].get(key)) for key in ("artifact_score", "progress_score"))
    non_ok_turns = sum(not _grader_ok(turn) for turn in graded_turns)
    return final_score * NON_OK_TURN_DISCOUNT**non_ok_turns


def _bounded_score(value: Any) -> float:
    if not isinstance(value, int | float):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def _grader_ok(content: dict[str, Any]) -> bool:
    return _bounded_score(content.get("artifact_score")) >= 1.0 and not content.get("artifact_issues")


def _decode_content(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}
