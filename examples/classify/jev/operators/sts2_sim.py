"""Operator-contract adapter for the in-house `sts2_sim` RunLoop.

Replaces the earlier `r33hab_sts2.py` shim: that one shelled out to a
NativeAOT emulator via `sts2_gym`, which was slow and required a .NET
toolchain. The new backend wraps the pure-Python simulator under
`../sts2_sim/` so a rollout worker can drive /reset -> /step without any
external build or process boundary.

Loads `sts2_sim` from the sibling directory via importlib so this file
is importable from any CWD and does not require `examples` to be a
Python package. Keep the public StatePacket shape 1:1 with
`examples/classify/jev/operator.schema.json`.

Phase 1 scope: ironclad A0, single Jaw Worm combat (full_run scope is
reported in StatePacket.info for schema compatibility — later phases
will fill in map / shops / events to make the full-run claim honest).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any


_SIM_ROOT = Path(__file__).resolve().parent.parent / "sts2_sim"
_SIM_CACHE_KEY = "_sts2_sim_for_operator_shim"


def _load_sim():
    cached = sys.modules.get(_SIM_CACHE_KEY)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        _SIM_CACHE_KEY,
        _SIM_ROOT / "__init__.py",
        submodule_search_locations=[str(_SIM_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not resolve sts2_sim at {_SIM_ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SIM_CACHE_KEY] = module
    spec.loader.exec_module(module)
    return module


_sim = _load_sim()
RunLoop = _sim.RunLoop
RunLoopError = _sim.RunLoopError
Screen = _sim.Screen
Outcome = _sim.Outcome


def _stable_seed(label: str) -> int:
    """Deterministic 63-bit int seed from an arbitrary string label.

    63 bits (not 64) because random.Random treats the seed as a signed
    integer and we want the hash to stay positive for debuggability.
    """

    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _render_state_text(state: Any) -> str:
    """Short, stable text digest for Jev's score head.

    Keep this tight — the reference backbone budgets ~1536 tokens for
    the full prompt and the state is only a small slice of that. Expand
    only when a decision point proves ambiguous without more context.
    """

    player = state.player
    combat = state.combat
    lines: list[str] = [
        f"screen={state.screen}",
        f"hp={player.hp}/{player.max_hp}" if player is not None else "hp=?",
    ]
    if player is not None:
        lines.append(f"gold={player.gold}")
    if combat is not None:
        lines.append(f"turn={combat.turn} energy={player.energy if player else '?'} block={player.block if player else 0}")
        alive = [m for m in combat.monsters if m.alive]
        if alive:
            enemy_strs = [
                f"{m.name}#{pos}[{m.hp}/{m.max_hp} block={m.block} intent={m.queued_move or '?'}]"
                for pos, m in enumerate(alive)
            ]
            lines.append("enemies=" + ", ".join(enemy_strs))
        else:
            lines.append("enemies=none")
        if player is not None and player.hand:
            # Collapse by card_id so a hand of five Strikes reads as
            # "strike x5" rather than filling the budget.
            counts: dict[str, int] = {}
            for cid in player.hand:
                counts[cid] = counts.get(cid, 0) + 1
            hand_strs = [
                (f"{cid} x{n}" if n > 1 else cid) for cid, n in counts.items()
            ]
            lines.append("hand=" + ", ".join(hand_strs))
    return "\n".join(lines)


class Sts2SimBackend:
    """In-process operator-contract backend over the in-house `sts2_sim`.

    Not a HTTP server; the rollout worker calls `reset` and `step`
    directly. Wrap in FastAPI if a cross-machine transport is wanted —
    the StatePacket is already JSON-safe.
    """

    BACKEND_ID = "sts2-sim"
    BACKEND_VERSION = "phase1"
    GAME_VERSION = "sts2-sim@phase1"

    def __init__(self) -> None:
        self._loops: dict[str, RunLoop] = {}
        self._config: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Contract endpoints

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.BACKEND_ID,
            "backend_version": self.BACKEND_VERSION,
            "characters": ["ironclad"],
            "ascensions": [0],
            "episode_scopes": ["full_run"],
            "game_versions": [self.GAME_VERSION],
            "supports_snapshot": False,
            "paired_seed_bit_exact": True,
            "approx_step_latency_ms": None,
            "max_parallel_episodes": None,
            "notes": (
                "In-house Python simulator; Phase 1 covers a single Jaw Worm "
                "combat for Ironclad A0. Later phases extend to full Act 1-3 runs."
            ),
        }

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require(request, ["character", "ascension", "episode_scope"])
        if request["character"] != "ironclad":
            raise ValueError("sts2-sim only supports ironclad")
        if request["episode_scope"] != "full_run":
            raise ValueError("sts2-sim only supports the full_run scope")
        ascension = int(request["ascension"])
        if ascension != 0:
            raise ValueError("sts2-sim Phase 1 only supports ascension 0")

        episode_id = request.get("episode_id") or str(uuid.uuid4())
        seed_label = str(request.get("seed") or f"STS2SIM-{episode_id[:8].upper()}")
        seed_int = _stable_seed(seed_label)

        loop = RunLoop(seed=seed_int, character="ironclad", ascension=ascension)
        packet = loop.reset()
        self._loops[episode_id] = loop
        self._config[episode_id] = {
            "character": "ironclad",
            "ascension": ascension,
            "seed_label": seed_label,
        }
        return self._wrap(episode_id, packet, reward=0.0)

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require(request, ["episode_id", "action_id"])
        episode_id = request["episode_id"]
        loop = self._loops.get(episode_id)
        if loop is None:
            raise ValueError(f"unknown episode_id {episode_id!r}")
        action_id = request["action_id"]
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("action_id must be a non-empty string")
        try:
            packet = loop.step(action_id)
        except RunLoopError as exc:
            raise ValueError(f"RunLoop rejected action {action_id!r}: {exc}") from exc
        reward = self._reward_for(packet)
        return self._wrap(episode_id, packet, reward=reward)

    def close(self, episode_id: str) -> None:
        loop = self._loops.pop(episode_id, None)
        self._config.pop(episode_id, None)
        if loop is not None:
            loop.close()

    # ------------------------------------------------------------------
    # Internals

    def _wrap(self, episode_id: str, packet: dict[str, Any], *, reward: float) -> dict[str, Any]:
        loop = self._loops[episode_id]
        config = self._config[episode_id]
        state = loop.state
        player = state.player
        combat = state.combat

        enemies_alive = sum(1 for m in combat.monsters if m.alive) if combat else None
        outcome_str: str | None = None
        if packet["done"]:
            outcome_str = "win" if state.outcome == Outcome.VICTORY else "loss"

        info = {
            "hp": player.hp if player is not None else 0,
            "max_hp": player.max_hp if player is not None else 0,
            "gold": player.gold if player is not None else None,
            "floor": state.floor,
            "act": state.act,
            "energy": player.energy if player is not None else None,
            "turn": combat.turn if combat is not None else None,
            "enemies_alive": enemies_alive,
            "outcome": outcome_str,
            "backend": self.BACKEND_ID,
            "backend_version": self.BACKEND_VERSION,
            "game_version": self.GAME_VERSION,
            "seed": config["seed_label"],
            "character": config["character"],
            "ascension": config["ascension"],
            "episode_scope": "full_run",
        }
        return {
            "episode_id": episode_id,
            "step": packet["step"],
            "done": packet["done"],
            "reward": reward,
            "decision_point": packet["decision_point"],
            "state_text": _render_state_text(state),
            "state_struct": None,
            "candidates": packet["candidates"],
            "info": info,
        }

    def _reward_for(self, packet: dict[str, Any]) -> float:
        """Phase 1 sparse reward: +1 victory / -1 defeat / 0 otherwise.

        Dense shaping (per-hit damage, hp preservation) can be added once
        we see what the Jev scorer actually needs to converge."""

        if not packet["done"]:
            return 0.0
        outcome = packet["outcome"]
        if outcome == Outcome.VICTORY:
            return 1.0
        if outcome == Outcome.DEATH:
            return -1.0
        return 0.0

    @staticmethod
    def _require(payload: dict[str, Any], keys: list[str]) -> None:
        missing = [k for k in keys if k not in payload]
        if missing:
            raise ValueError(f"missing required fields: {missing}")


__all__ = ["Sts2SimBackend"]
