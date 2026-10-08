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

Scope: Ironclad, ascension 0-10, a whole run: Neow, three acts of map
(Overgrowth or Underdocks, Hive, Glory), combats, rewards, rest sites,
shops and treasure rooms, ending at the last boss or on death.
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


_INTENT_LABEL = {
    "attack": "Attack", "defend": "Defend", "buff": "Buff", "debuff": "Debuff",
    "strong_debuff": "StrongDebuff", "status": "Status", "card_debuff": "CardDebuff",
    "summon": "Summon", "heal": "Heal", "stun": "Stun", "sleep": "Sleep",
    "escape": "Escape", "death_blow": "DeathBlow", "unknown": "Unknown",
}


def _intent_text(loop: Any, monster: Any) -> str:
    """Intent readout in the live adapter's `type label` shape ("Attack 7x2")."""

    move_id = monster.queued_move
    if move_id in (None, "__reviving__"):
        return "Heal" if move_id else "?"
    if move_id == "__stunned__":
        return "Stun"
    mdef = loop.monsters[monster.monster_id]
    move = mdef.moves[move_id]
    ctx = loop.combat_ctx
    parts = []
    for intent in move.intents:
        label = _INTENT_LABEL[intent]
        if intent == "attack":
            step = next((s for s in move.effects if s.verb == "attack"), None)
            if step is not None:
                base = _sim_value(ctx, step.args["damage"])
                hits = _sim_value(ctx, step.args.get("hits", 1))
                per_hit = ctx._powered_amount(base, monster, ctx.player)
                label += f" {per_hit}" + (f"x{hits}" if hits > 1 else "")
        parts.append(label)
    return "/".join(parts)


def _sim_value(ctx: Any, raw: Any) -> int:
    if isinstance(raw, list):
        base, asc, level = raw
        return int(asc if ctx.ascension >= level else base)
    return int(raw)


def _powers_text(powers: dict[str, int], defs: dict[str, Any]) -> str:
    out = []
    for pid, amount in powers.items():
        name = defs[pid].name if pid in defs else pid
        out.append(f"{name}={amount}")
    return ", ".join(out)


_LIVE_STATE_TYPE = {
    "neow": "event", "ancient": "event", "map": "map", "rewards": "rewards", "card_reward": "card_reward",
    "card_select": "card_select", "rest": "rest_site", "shop": "shop", "treasure": "treasure",
    "event": "event", "game_over": "game_over",
}


def _render_state_text(loop: Any) -> str:
    """Compact digest for Jev's score head, shaped like the live adapter's.

    Same line layout as `sts2mcp_live.render_state_text` so a policy trained
    on the sim reads the live game the same way.
    """

    state = loop.state
    player = state.player
    combat = state.combat
    screen = state.screen
    in_combat = screen == "combat"
    if in_combat:
        screen = state.room or "monster"
    else:
        screen = _LIVE_STATE_TYPE.get(screen, screen)
    lines: list[str] = [
        f"screen={screen} act={state.act} floor={state.floor}",
        f"hp={player.hp}/{player.max_hp} gold={player.gold}" if player is not None else "hp=?",
    ]
    if player is not None and player.relics:
        lines.append("relics=" + ", ".join(_relic_name(loop, r) for r in player.relics))
    if player is not None and any(player.potions):
        lines.append("potions=" + ", ".join(_potion_name(loop, x) for x in player.potions if x))
    if in_combat and combat is not None and player is not None and combat.outcome is None:
        lines.append(f"round={combat.turn} energy={player.energy}/{player.max_energy} block={player.block}")
        status = _powers_text(player.powers, loop.powers)
        if status:
            lines.append("player_status=" + status)
        enemy_strs = []
        for pos, m in enumerate(m for m in combat.monsters if m.alive):
            powers = _powers_text(m.powers, loop.powers)
            enemy_strs.append(
                f"{m.name}#{pos}[{m.hp}/{m.max_hp} block={m.block} intent={_intent_text(loop, m)}"
                f"{(' ' + powers) if powers else ''}]"
            )
        lines.append("enemies=" + (", ".join(enemy_strs) if enemy_strs else "none"))
        if player.hand:
            # Collapse by card_id so a hand of five Strikes reads as "x5".
            counts: dict[str, int] = {}
            for cid in player.hand:
                counts[cid] = counts.get(cid, 0) + 1
            lines.append("hand=" + ", ".join(f"{cid} x{n}" if n > 1 else cid for cid, n in counts.items()))
        lines.append(
            f"piles draw={len(player.draw_pile)} discard={len(player.discard_pile)} "
            f"exhaust={len(player.exhaust_pile)}"
        )
        sel = combat.pending_selection
        if sel is not None:
            verb = {"exhaust": "Exhaust", "upgrade": "Upgrade", "to_draw_top": "Put on top of your Draw Pile"}
            if sel.source == "generated":
                lines.append("prompt=Make your choice.")
            else:
                lines.append(f"prompt=Choose {sel.max_count} card(s) to {verb.get(sel.purpose, sel.purpose)}.")
    elif screen == "card_select" and state.deck_select is not None:
        verb = {"upgrade": "Upgrade", "remove": "Remove"}.get(state.deck_select.purpose, state.deck_select.purpose)
        lines.append(f"prompt=Choose a card to {verb}.")
    if not in_combat and player is not None and screen in ("map", "rewards", "card_reward", "rest_site", "shop",
                                                            "card_select", "treasure"):
        lines.append(f"deck={len(player.deck)} cards")
    if screen == "event" and state.screen in ("neow", "ancient"):
        lines.append("event=Neow" if state.screen == "neow" else "event=Ancient")
    return "\n".join(lines)


def _relic_name(loop: Any, relic_id: str) -> str:
    relics = getattr(loop, "relics", None)
    if relics and relic_id in relics:
        return relics[relic_id].name
    return relic_id.replace("_", " ").title()


def _potion_name(loop: Any, potion_id: str) -> str:
    potions = getattr(loop, "potion_defs", None)
    if potions and potion_id in potions:
        return potions[potion_id].name
    return potion_id.replace("_", " ").title()


class Sts2SimBackend:
    """In-process operator-contract backend over the in-house `sts2_sim`.

    Not a HTTP server; the rollout worker calls `reset` and `step`
    directly. Wrap in FastAPI if a cross-machine transport is wanted —
    the StatePacket is already JSON-safe.
    """

    BACKEND_ID = "sts2-sim"
    BACKEND_VERSION = "run-v0.107.1"
    GAME_VERSION = "sts2-sim@v0.107.1"

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
            "ascensions": list(range(11)),
            "episode_scopes": ["full_run"],
            "game_versions": [self.GAME_VERSION],
            "supports_snapshot": False,
            "paired_seed_bit_exact": True,
            "approx_step_latency_ms": None,
            "max_parallel_episodes": None,
            "notes": (
                "In-house Python simulator of STS2 v0.107.1: every Ironclad card, every "
                "Act 1-3 monster and encounter, act maps, combat rewards, rest sites, "
                "shops and treasure rooms over a full three-act run."
            ),
        }

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require(request, ["character", "ascension", "episode_scope"])
        if request["character"] != "ironclad":
            raise ValueError("sts2-sim only supports ironclad")
        if request["episode_scope"] != "full_run":
            raise ValueError("sts2-sim only supports the full_run scope")
        ascension = int(request["ascension"])
        if not 0 <= ascension <= 10:
            raise ValueError("sts2-sim supports ascension 0-10")

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
            "state_text": _render_state_text(loop),
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
