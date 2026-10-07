"""Operator contract adapter for r33hab/sts2 (NativeAOT STS2 emulator).

Wraps `sts2_gym.Sts2RunEnv` behind the shapes in `operator.schema.json`.
Each `reset`/`step` returns a dict that matches `StatePacket`; `candidates`
hold one entry per legal integer action under the current phase mask.

Prereqs:

    git clone https://github.com/r33hab/sts2 /tmp/r33hab_sts2
    cd /tmp/r33hab_sts2
    bash scripts/build.sh linux-x64            # or osx-arm64 / osx-x64 / win-x64
    export STS2_LIB_PATH=$PWD/out
    export PYTHONPATH=$PWD/src

The default character/ascension pin is Ironclad A0, which is what the emulator
exposes end-to-end today. Pass another `character` in `ResetRequest` once the
backend reports it via `/capabilities`.

Rendering: `state_text` is a short, stable text digest readable by the Jev
score head; keep an eye on the max_seq_len budget (1536 tokens on the
reference Ling-3.0-tiny recipe) and extend only when a decision needs more.
This first cut keeps rendering minimal so a smoke test can land before the
digest format gets opinionated.
"""

from __future__ import annotations

import uuid
from typing import Any


_PHASE_TO_DECISION_POINT: dict[int, str] | None = None
_PHASE_NAMES: dict[int, str] | None = None


def _load_sts2_gym():
    try:
        import sts2_gym  # noqa: F401  — importing registers the native library check
        from sts2_gym import Sts2RunEnv
        from sts2_gym import run_constants as C
    except ImportError as exc:
        raise ImportError(
            "sts2_gym is not importable. Build r33hab/sts2 and point PYTHONPATH at "
            "src/ plus STS2_LIB_PATH at out/. See the docstring in this file."
        ) from exc
    return Sts2RunEnv, C


def _init_phase_tables(C) -> None:
    global _PHASE_TO_DECISION_POINT, _PHASE_NAMES
    if _PHASE_TO_DECISION_POINT is not None:
        return
    _PHASE_TO_DECISION_POINT = {
        C.PHASE_COMBAT: "combat_play",
        C.PHASE_CARD_REWARD: "card_reward",
        C.PHASE_MAP: "map_select",
        C.PHASE_REST: "rest_site",
        C.PHASE_SHOP: "shop",
        C.PHASE_RELIC_REWARD: "boss_relic",
        C.PHASE_COMPLETE: "game_over",
        C.PHASE_EVENT: "event_choice",
        C.PHASE_ANCIENT: "neow_bonus",
        C.PHASE_TRANSFORM_SELECT: "event_choice",
        C.PHASE_TREASURE: "card_reward",
        C.PHASE_CRYSTAL_SPHERE: "event_choice",
        C.PHASE_BUNDLE_SELECT: "card_reward",
    }
    _PHASE_NAMES = {
        C.PHASE_COMBAT: "combat",
        C.PHASE_CARD_REWARD: "card_reward",
        C.PHASE_MAP: "map",
        C.PHASE_REST: "rest",
        C.PHASE_SHOP: "shop",
        C.PHASE_RELIC_REWARD: "relic_reward",
        C.PHASE_COMPLETE: "complete",
        C.PHASE_EVENT: "event",
        C.PHASE_ANCIENT: "neow",
        C.PHASE_TRANSFORM_SELECT: "transform_select",
        C.PHASE_TREASURE: "treasure",
        C.PHASE_CRYSTAL_SPHERE: "crystal_sphere",
        C.PHASE_BUNDLE_SELECT: "bundle_select",
    }


class R33habSts2Backend:
    """Minimal in-process operator-contract backend over `Sts2RunEnv`.

    Not a HTTP server; the rollout worker is expected to call `reset` and
    `step` directly, or wrap this class in FastAPI if a cross-machine
    transport is wanted. The method bodies validate request shapes against
    the StatePacket/ResetRequest/StepRequest contracts and let dict output
    pass through to JSON encoding without further massaging.
    """

    BACKEND_ID = "r33hab-sts2"
    GAME_VERSION = "sts2@v0.107.1"

    def __init__(self, *, max_episode_steps: int = 200):
        self._Sts2RunEnv, self._C = _load_sts2_gym()
        _init_phase_tables(self._C)
        self._max_episode_steps = int(max_episode_steps)
        self._envs: dict[str, Any] = {}
        self._steps: dict[str, int] = {}
        self._config: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Contract endpoints

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.BACKEND_ID,
            "backend_version": None,
            "characters": ["ironclad"],
            "ascensions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "episode_scopes": ["full_run"],
            "game_versions": [self.GAME_VERSION],
            "supports_snapshot": True,
            "paired_seed_bit_exact": True,
            "approx_step_latency_ms": None,
            "max_parallel_episodes": None,
            "notes": "Ironclad A0 verified end-to-end; higher ascensions compile but have light test coverage.",
        }

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require(request, ["character", "ascension", "episode_scope"])
        if request["character"] != "ironclad":
            raise ValueError("r33hab-sts2 currently only exposes ironclad end-to-end")
        if request["episode_scope"] != "full_run":
            raise ValueError("r33hab-sts2 only runs the full-run scope")
        ascension = int(request["ascension"])
        if not 0 <= ascension <= 10:
            raise ValueError("ascension must be in 0..10")

        episode_id = request.get("episode_id") or str(uuid.uuid4())
        seed = request.get("seed") or f"R33HAB-{episode_id[:8].upper()}"
        env = self._Sts2RunEnv(
            seed=seed,
            ascension=ascension,
            max_episode_steps=self._max_episode_steps,
        )
        obs, info = env.reset()
        self._envs[episode_id] = env
        self._steps[episode_id] = 0
        self._config[episode_id] = {
            "character": "ironclad",
            "ascension": ascension,
            "seed": str(seed),
        }
        return self._state_packet(episode_id, reward=0.0, done=False, info=info)

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require(request, ["episode_id", "action_id"])
        episode_id = request["episode_id"]
        env = self._envs.get(episode_id)
        if env is None:
            raise ValueError(f"unknown episode_id {episode_id!r}")
        try:
            action = int(request["action_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("action_id must be an integer-valued string") from exc
        _obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        self._steps[episode_id] += 1
        packet = self._state_packet(episode_id, reward=float(reward), done=done, info=info)
        if done:
            # Keep the env around so the caller can introspect; `close` drops it.
            pass
        return packet

    def close(self, episode_id: str) -> None:
        env = self._envs.pop(episode_id, None)
        self._steps.pop(episode_id, None)
        self._config.pop(episode_id, None)
        if env is not None:
            env.close()

    # ------------------------------------------------------------------
    # Internals

    def _state_packet(self, episode_id: str, *, reward: float, done: bool, info: dict) -> dict[str, Any]:
        env = self._envs[episode_id]
        phase = int(info["phase"])
        assert _PHASE_TO_DECISION_POINT is not None
        decision_point = "game_over" if done else _PHASE_TO_DECISION_POINT.get(phase, "game_over")

        candidates = (
            [{"id": "terminal", "text": "<game_over>"}]
            if done
            else self._enumerate_candidates(env, phase, info)
        )
        if not candidates:
            decision_point = "game_over"
            candidates = [{"id": "terminal", "text": "<no_legal_actions>"}]

        state_text = self._render_state_text(phase, info)
        config = self._config[episode_id]
        outcome: str | None = None
        if done:
            outcome = "win" if info.get("player_won") else "loss"

        return {
            "episode_id": episode_id,
            "step": self._steps[episode_id],
            "done": done,
            "reward": reward,
            "decision_point": decision_point,
            "state_text": state_text,
            "state_struct": None,
            "candidates": candidates,
            "info": {
                "hp": int(info.get("player_hp", 0)),
                "max_hp": int(info.get("player_max_hp", 0)),
                "gold": _optional_int(info.get("gold")),
                "floor": _optional_int(info.get("floor")),
                "act": _as_text(info.get("act")),
                "energy": None,
                "turn": None,
                "enemies_alive": None,
                "outcome": outcome,
                "backend": self.BACKEND_ID,
                "backend_version": None,
                "game_version": self.GAME_VERSION,
                "seed": config["seed"],
                "character": config["character"],
                "ascension": config["ascension"],
                "episode_scope": "full_run",
            },
        }

    def _enumerate_candidates(self, env, phase: int, info: dict) -> list[dict[str, str]]:
        mask = env.action_masks()
        legal = [i for i, allowed in enumerate(mask) if bool(allowed)]
        return [{"id": str(action), "text": self._render_candidate(action, phase, info)} for action in legal]

    def _render_candidate(self, action: int, phase: int, info: dict) -> str:
        """Minimal human-readable text per candidate. Keep stable, extend later."""

        assert _PHASE_NAMES is not None
        phase_name = _PHASE_NAMES.get(phase, f"phase{phase}")
        if phase == self._C.PHASE_ANCIENT:
            options = tuple(info.get("neow_options") or ())
            if action < len(options):
                return f"[neow:{action}] option_id={options[action]}"
        if phase == self._C.PHASE_MAP:
            choices = tuple(info.get("map_choices") or ())
            if action < len(choices):
                node = choices[action]
                return f"[map:{action}] node_type={node.get('node_type')} xy=({node.get('x')},{node.get('y')})"
        if phase == self._C.PHASE_CARD_REWARD:
            rewards = tuple(info.get("card_rewards") or ())
            if action < len(rewards):
                return f"[card_reward:{action}] card_id={rewards[action]}"
        return f"[{phase_name}:{action}]"

    def _render_state_text(self, phase: int, info: dict) -> str:
        assert _PHASE_NAMES is not None
        parts = [
            f"phase={_PHASE_NAMES.get(phase, f'phase{phase}')}",
            f"hp={info.get('player_hp')}/{info.get('player_max_hp')}",
            f"floor={info.get('floor')}",
            f"act={info.get('act')}",
            f"gold={info.get('gold')}",
            f"deck_size={info.get('deck_size')}",
        ]
        if phase == self._C.PHASE_COMBAT:
            parts.append(f"encounter={info.get('encounter')}")
        return "\n".join(parts)

    @staticmethod
    def _require(payload: dict[str, Any], keys: list[str]) -> None:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f"missing required fields: {missing}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = ["R33habSts2Backend"]
