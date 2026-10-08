"""Live-game adapter: STS2MCP game state <-> Jev candidates.

Talks to the STS2MCP mod's localhost REST API (Gennadiyev/STS2MCP,
`GET/POST /api/v1/singleplayer`). Each poll turns the game's JSON state
into an operator-contract StatePacket whose candidate ids use the same
vocabulary as `sts2_sim` (`sts2_sim/actions.py`): `"<mcp_action>[:args]"`.
`Candidate.request` is the exact POST body that executes the candidate.

Argument conventions (sim-compatible where the sim has the screen):
  play_card:{card_key}[:{enemy_pos}]
      card_key  = lower(card id) + "+1" if upgraded; identical cards in hand
                  collapse to one candidate that plays the first copy.
      enemy_pos = position in `battle.enemies` (alive enemies only).
  use_potion:{slot}[:{enemy_pos}] / discard_potion:{slot}
  every other screen: the screen's own `index` field (or x:y for cells).

`enumerate_candidates` / `render_state_text` are pure functions over the
state dict, so they are CPU-testable against recorded JSON.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_ACTIONS_PATH = Path(__file__).resolve().parent.parent / "sts2_sim" / "actions.py"
_ACTIONS_CACHE_KEY = "_sts2_sim_actions_for_live_adapter"


def _load_actions():
    cached = sys.modules.get(_ACTIONS_CACHE_KEY)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_ACTIONS_CACHE_KEY, _ACTIONS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not resolve sts2_sim actions at {_ACTIONS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ACTIONS_CACHE_KEY] = module
    spec.loader.exec_module(module)
    return module


A = _load_actions()

DEFAULT_BASE_URL = "http://localhost:15526"

COMBAT_STATE_TYPES = frozenset({"monster", "elite", "boss"})

# STS2MCP state_type -> operator decision_point.
DECISION_POINT_BY_STATE = {
    "monster": "combat_play",
    "elite": "combat_play",
    "boss": "combat_play",
    "hand_select": "hand_select",
    "rewards": "rewards",
    "card_reward": "card_reward",
    "map": "map_select",
    "event": "event_choice",
    "rest_site": "rest_site",
    "shop": "shop",
    "fake_merchant": "shop",
    "treasure": "treasure",
    "card_select": "card_select",
    "bundle_select": "bundle_select",
    "relic_select": "boss_relic",
    "crystal_sphere": "crystal_sphere",
    "game_over": "game_over",
}


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str
    request: dict[str, Any] = field(compare=False)


def card_key(card: dict[str, Any]) -> str:
    key = str(card.get("id", "unknown")).lower()
    return key + "+1" if card.get("is_upgraded") else key


def decision_point(state: dict[str, Any]) -> str:
    state_type = state.get("state_type", "unknown")
    if state_type == "event" and str(state.get("event", {}).get("event_id", "")).upper() == "NEOW":
        return "neow_bonus"
    return DECISION_POINT_BY_STATE.get(state_type, state_type)


# ---------------------------------------------------------------------------
# Candidate enumeration


def _cand(name: str, args: tuple, text: str, **params: Any) -> Candidate:
    return Candidate(id=A.format_action(name, *args), text=text, request={"action": name, **params})


def _enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list(state.get("battle", {}).get("enemies") or [])


def _enemy_label(enemy: dict[str, Any], pos: int) -> str:
    return f"{enemy.get('name', '?')}#{pos}[{enemy.get('hp', '?')}/{enemy.get('max_hp', '?')}]"


def _combat(state: dict[str, Any]) -> list[Candidate]:
    battle = state.get("battle", {})
    if not battle.get("is_play_phase", False):
        return []  # enemy turn / animations: caller re-polls
    enemies = _enemies(state)
    out: list[Candidate] = []
    seen: set[str] = set()
    for card in state.get("player", {}).get("hand") or []:
        if not card.get("can_play"):
            continue
        key = card_key(card)
        if key in seen:
            continue
        seen.add(key)
        label = f"play {card.get('name', key)} (cost {card.get('cost', '?')}): {card.get('description', '')}".rstrip(": ")
        if card.get("target_type") == "AnyEnemy":
            for pos, enemy in enumerate(enemies):
                out.append(_cand(
                    A.PLAY_CARD, (key, pos), f"{label} -> {_enemy_label(enemy, pos)}",
                    card_index=card["index"], target=enemy["entity_id"],
                ))
        else:
            out.append(_cand(A.PLAY_CARD, (key,), label, card_index=card["index"]))
    out.extend(_potions(state, in_combat=True))
    out.append(_cand(A.END_TURN, (), "end turn"))
    return out


def _potions(state: dict[str, Any], *, in_combat: bool) -> list[Candidate]:
    player = state.get("player", {})
    potions = player.get("potions") or []
    out: list[Candidate] = []
    if in_combat:
        enemies = _enemies(state)
        for potion in potions:
            if not potion.get("can_use_in_combat"):
                continue
            slot = potion["slot"]
            label = f"use potion {potion.get('name', '?')}: {potion.get('description', '')}"
            if potion.get("target_type") == "AnyEnemy":
                for pos, enemy in enumerate(enemies):
                    out.append(_cand(
                        A.USE_POTION, (slot, pos), f"{label} -> {_enemy_label(enemy, pos)}",
                        slot=slot, target=enemy["entity_id"],
                    ))
            else:
                out.append(_cand(A.USE_POTION, (slot,), label, slot=slot))
    # Discarding only matters when the belt is full and a new potion is on offer.
    if len(potions) >= int(player.get("max_potion_slots", 3)) and state.get("state_type") == "rewards":
        for potion in potions:
            slot = potion["slot"]
            out.append(_cand(A.DISCARD_POTION, (slot,), f"discard potion {potion.get('name', '?')}", slot=slot))
    return out


def _indexed(name: str, items: list[dict[str, Any]], describe, param: str = "index") -> list[Candidate]:
    return [_cand(name, (item["index"],), describe(item), **{param: item["index"]}) for item in items]


def _card_text(card: dict[str, Any]) -> str:
    up = "+" if card.get("is_upgraded") else ""
    return f"{card.get('name', '?')}{up} ({card.get('type', '?')}, cost {card.get('cost', '?')}): {card.get('description', '')}"


def _shop_item_text(item: dict[str, Any]) -> str:
    cat = item.get("category", "?")
    price = item.get("price", item.get("cost", "?"))
    if cat == "card":
        what = f"card {item.get('card_name', '?')}: {item.get('card_description', '')}"
    elif cat == "relic":
        what = f"relic {item.get('relic_name', '?')}: {item.get('relic_description', '')}"
    elif cat == "potion":
        what = f"potion {item.get('potion_name', '?')}: {item.get('potion_description', '')}"
    else:
        what = cat
    return f"buy {what} for {price} gold"


def _shop(shop: dict[str, Any]) -> list[Candidate]:
    items = [i for i in shop.get("items") or [] if i.get("is_stocked") and i.get("can_afford")]
    out = _indexed(A.SHOP_PURCHASE, items, _shop_item_text)
    if shop.get("can_proceed"):
        out.append(_cand(A.PROCEED, (), "leave"))
    return out


def enumerate_candidates(state: dict[str, Any]) -> list[Candidate]:
    """Legal Jev candidates for one STS2MCP singleplayer state.

    Returns [] for transitional / unsupported screens (enemy turn, chest
    opening, menu, overlay, unknown); the driver should re-poll.
    """

    st = state.get("state_type")
    if st in COMBAT_STATE_TYPES:
        return _combat(state)
    if st == "hand_select":
        hs = state.get("hand_select", {})
        out = _indexed(A.COMBAT_SELECT_CARD, hs.get("cards") or [],
                       lambda c: f"select {_card_text(c)}", param="card_index")
        if hs.get("can_confirm"):
            out.append(_cand(A.COMBAT_CONFIRM_SELECTION, (), "confirm selection"))
        return out
    if st == "rewards":
        rw = state.get("rewards", {})
        out = _indexed(A.CLAIM_REWARD, rw.get("items") or [],
                       lambda r: f"claim {r.get('type', '?')}: {r.get('description', '')}")
        out.extend(_potions(state, in_combat=False))
        if rw.get("can_proceed"):
            out.append(_cand(A.PROCEED, (), "proceed"))
        return out
    if st == "card_reward":
        cr = state.get("card_reward", {})
        out = _indexed(A.SELECT_CARD_REWARD, cr.get("cards") or [],
                       lambda c: f"take {_card_text(c)}", param="card_index")
        if cr.get("can_skip"):
            out.append(_cand(A.SKIP_CARD_REWARD, (), "skip card reward"))
        return out
    if st == "map":
        def node_text(n: dict[str, Any]) -> str:
            ahead = ", ".join(c.get("type", "?") for c in n.get("leads_to") or [])
            return f"go to {n.get('type', '?')} (row {n.get('row', '?')}, col {n.get('col', '?')}) leads to [{ahead}]"
        return _indexed(A.CHOOSE_MAP_NODE, state.get("map", {}).get("next_options") or [], node_text)
    if st == "event":
        ev = state.get("event", {})
        if ev.get("in_dialogue"):
            return [_cand(A.ADVANCE_DIALOGUE, (), "continue dialogue")]
        opts = [o for o in ev.get("options") or [] if not o.get("is_locked")]
        return _indexed(A.CHOOSE_EVENT_OPTION, opts,
                        lambda o: f"{o.get('title', '?')}: {o.get('description', '')}")
    if st == "rest_site":
        rs = state.get("rest_site", {})
        opts = [o for o in rs.get("options") or [] if o.get("is_enabled", True)]
        out = _indexed(A.CHOOSE_REST_OPTION, opts,
                       lambda o: f"{o.get('name', '?')}: {o.get('description', '')}")
        if rs.get("can_proceed"):
            out.append(_cand(A.PROCEED, (), "leave rest site"))
        return out
    if st == "shop":
        return _shop(state.get("shop", {}))
    if st == "fake_merchant":
        return _shop(state.get("fake_merchant", {}).get("shop", {}))
    if st == "treasure":
        tr = state.get("treasure", {})
        out = _indexed(A.CLAIM_TREASURE_RELIC, tr.get("relics") or [],
                       lambda r: f"take relic {r.get('name', '?')}: {r.get('description', '')}")
        if tr.get("can_proceed"):
            out.append(_cand(A.PROCEED, (), "leave treasure room"))
        return out
    if st == "card_select":
        cs = state.get("card_select", {})
        out = _indexed(A.SELECT_CARD, cs.get("cards") or [], lambda c: f"select {_card_text(c)}")
        if cs.get("can_confirm"):
            out.append(_cand(A.CONFIRM_SELECTION, (), "confirm selection"))
        if cs.get("can_cancel") or cs.get("can_skip"):
            out.append(_cand(A.CANCEL_SELECTION, (), "cancel / skip"))
        return out
    if st == "bundle_select":
        bs = state.get("bundle_select", {})
        out = _indexed(A.SELECT_BUNDLE, bs.get("bundles") or [],
                       lambda b: "bundle: " + "; ".join(_card_text(c) for c in b.get("cards") or []))
        if bs.get("can_confirm"):
            out.append(_cand(A.CONFIRM_BUNDLE_SELECTION, (), "confirm bundle"))
        if bs.get("can_cancel"):
            out.append(_cand(A.CANCEL_BUNDLE_SELECTION, (), "cancel bundle preview"))
        return out
    if st == "relic_select":
        rs = state.get("relic_select", {})
        out = _indexed(A.SELECT_RELIC, rs.get("relics") or [],
                       lambda r: f"take relic {r.get('name', '?')}: {r.get('description', '')}")
        if rs.get("can_skip"):
            out.append(_cand(A.SKIP_RELIC_SELECTION, (), "skip relic"))
        return out
    if st == "crystal_sphere":
        cs = state.get("crystal_sphere", {})
        out = [
            _cand(A.CRYSTAL_SPHERE_CLICK_CELL, (c["x"], c["y"]), f"reveal cell ({c['x']}, {c['y']})", x=c["x"], y=c["y"])
            for c in cs.get("clickable_cells") or []
        ]
        for tool in ("big", "small"):
            if cs.get(f"can_use_{tool}_tool") and cs.get("tool") != tool:
                out.append(_cand(A.CRYSTAL_SPHERE_SET_TOOL, (tool,), f"switch to {tool} tool", tool=tool))
        if cs.get("can_proceed"):
            out.append(_cand(A.CRYSTAL_SPHERE_PROCEED, (), "finish crystal sphere"))
        return out
    if st == "game_over":
        return [_cand(A.MENU_SELECT, ("main_menu",), "<game_over>", option="main_menu")]
    return []


# ---------------------------------------------------------------------------
# State text


def render_state_text(state: dict[str, Any]) -> str:
    """Compact digest for the score head (same spirit as the sim's)."""

    st = state.get("state_type", "unknown")
    run = state.get("run") or {}
    player = state.get("player") or {}
    lines = [
        f"screen={st} act={run.get('act', '?')} floor={run.get('floor', '?')}",
        f"hp={player.get('hp', '?')}/{player.get('max_hp', '?')} gold={player.get('gold', '?')}",
    ]
    relics = [r.get("name", "?") for r in player.get("relics") or []]
    if relics:
        lines.append("relics=" + ", ".join(relics))
    potions = [p.get("name", "?") for p in player.get("potions") or []]
    if potions:
        lines.append("potions=" + ", ".join(potions))
    if "battle" in state:
        battle = state["battle"]
        lines.append(
            f"round={battle.get('round', '?')} energy={player.get('energy', '?')}/{player.get('max_energy', '?')} "
            f"block={player.get('block', 0)}"
        )
        status = [f"{s.get('name', '?')}={s.get('amount', '')}" for s in player.get("status") or []]
        if status:
            lines.append("player_status=" + ", ".join(status))
        enemy_strs = []
        for pos, e in enumerate(_enemies(state)):
            intents = "/".join(f"{i.get('type', '?')}{(' ' + i['label']) if i.get('label') else ''}" for i in e.get("intents") or [])
            powers = ",".join(f"{s.get('name', '?')}={s.get('amount', '')}" for s in e.get("status") or [])
            enemy_strs.append(
                f"{e.get('name', '?')}#{pos}[{e.get('hp', '?')}/{e.get('max_hp', '?')} block={e.get('block', 0)}"
                f" intent={intents or '?'}{(' ' + powers) if powers else ''}]"
            )
        lines.append("enemies=" + (", ".join(enemy_strs) if enemy_strs else "none"))
        counts: dict[str, int] = {}
        for card in player.get("hand") or []:
            k = card_key(card)
            counts[k] = counts.get(k, 0) + 1
        if counts:
            lines.append("hand=" + ", ".join(f"{k} x{n}" if n > 1 else k for k, n in counts.items()))
        lines.append(
            f"piles draw={player.get('draw_pile_count', '?')} discard={player.get('discard_pile_count', '?')} "
            f"exhaust={player.get('exhaust_pile_count', '?')}"
        )
    for key, prompt_field in (("hand_select", "prompt"), ("card_select", "prompt"), ("relic_select", "prompt")):
        if key in state and state[key].get(prompt_field):
            lines.append(f"prompt={state[key][prompt_field]}")
    if st == "event":
        ev = state.get("event", {})
        lines.append(f"event={ev.get('event_name', '?')}: {ev.get('body', '')}".strip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP client + operator-contract backend


class Sts2McpClient:
    """Thin stdlib client for STS2MCP's singleplayer endpoint."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0):
        self._url = base_url.rstrip("/") + "/api/v1/singleplayer"
        self._timeout = timeout

    def get_state(self) -> dict[str, Any]:
        with urllib.request.urlopen(self._url + "?format=json", timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self._url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return json.loads(exc.read().decode("utf-8") or "{}") or {"status": "error", "error": str(exc)}


class Sts2McpLiveBackend:
    """Operator-contract view of the one live game behind STS2MCP.

    There is exactly one episode (whatever run the game is showing), so
    `reset` only snapshots the current screen; starting a run is left to
    the human or to `menu_select` outside this adapter.

    STS2MCP enqueues actions and answers before they resolve (play_card
    goes through the game's ActionQueue), so a GET right after a POST can
    still show the pre-action hand and invite a double play. `step`
    therefore polls until the state differs from the pre-action state and
    then reads the same state twice in a row (settled), or until
    `settle_timeout` elapses.
    """

    BACKEND_ID = "sts2mcp-live"

    def __init__(
        self,
        client: Sts2McpClient | None = None,
        *,
        settle_timeout: float = 10.0,
        poll_interval: float = 0.2,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self._client = client or Sts2McpClient()
        self._settle_timeout = settle_timeout
        self._poll_interval = poll_interval
        self._sleep = sleep
        self._clock = clock
        self._episode_id: str | None = None
        self._step = 0
        self._last: list[Candidate] = []
        self._last_state: dict[str, Any] | None = None
        self.last_settled = True

    def reset(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        self._episode_id = (request or {}).get("episode_id") or str(uuid.uuid4())
        self._step = 0
        return self.observe()

    def observe(self) -> dict[str, Any]:
        """Re-read the game without acting (use while candidates are empty)."""

        self.last_settled = True
        return self._packet(self._client.get_state())

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        action_id = request["action_id"]
        by_id = {c.id: c for c in self._last}
        if action_id not in by_id:
            raise ValueError(f"action {action_id!r} not in last candidates {sorted(by_id)}")
        result = self._client.post(by_id[action_id].request)
        if result.get("status") != "ok":
            raise RuntimeError(f"STS2MCP rejected {action_id!r}: {result.get('error') or result.get('message') or result}")
        self._step += 1
        return self._packet(self._wait_settled(self._last_state))

    def _wait_settled(self, before: dict[str, Any] | None) -> dict[str, Any]:
        deadline = self._clock() + self._settle_timeout
        prev = None
        while True:
            state = self._client.get_state()
            if state != before and state == prev:
                self.last_settled = True
                return state
            if self._clock() >= deadline:
                self.last_settled = False
                return state
            prev = state
            self._sleep(self._poll_interval)

    def _packet(self, state: dict[str, Any]) -> dict[str, Any]:
        assert self._episode_id is not None, "call reset() first"
        self._last_state = state
        self._last = enumerate_candidates(state)
        done = state.get("state_type") == "game_over"
        return {
            "episode_id": self._episode_id,
            "step": self._step,
            "done": done,
            "reward": 0.0,  # live play has no shaped reward; outcome is read from the game
            "decision_point": decision_point(state),
            "state_text": render_state_text(state),
            "state_struct": state,
            "candidates": [{"id": c.id, "text": c.text} for c in self._last],
            "info": {"backend": self.BACKEND_ID, "state_type": state.get("state_type"),
                     "settled": self.last_settled},
        }
