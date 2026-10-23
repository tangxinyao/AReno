"""Reward for the STS2 ReAct example.

The reward is recomputed by replaying the agent's accepted actions on a fresh
simulator built from the record's seed. The simulator is deterministic, so the
replay reaches the same final state the rollout did.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import MAX_INVALID_ACTIONS, Episode, Sts2ReActEnv, parse_action  # noqa: E402

INVALID_REWARD = -1.0


def reward_fn(record: Any) -> float:
    """Score one episode: enemy damage dealt plus a bonus for winning."""

    source = dict(record.source_record)
    env = Sts2ReActEnv(
        Episode(
            encounter=str(source["encounter"]),
            seed=int(source["seed"]),
            ascension=int(source.get("ascension", 0)),
        )
    )
    try:
        env.reset()
        invalid = 0
        for message in record.messages:
            if env.done:
                break
            if message.get("role") != "assistant":
                continue
            action = parse_action(_content_text(message.get("content")))
            if action is None or action not in env.legal_ids():
                invalid += 1
                if invalid >= MAX_INVALID_ACTIONS:
                    return INVALID_REWARD
                continue
            env.step(action)
        return score(env)
    finally:
        env.close()


def score(env: Sts2ReActEnv) -> float:
    """Half credit for the share of enemy HP removed, half for winning."""

    return 0.5 * env.enemy_damage_fraction() + 0.5 * float(env.won)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""
