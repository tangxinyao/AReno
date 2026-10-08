"""RunLoop: Neow stub -> one Act 1 combat -> game over.

The run picks its Act 1 variant (Overgrowth or Underdocks, like the game)
and fights a random encounter from that act's weak pool, which is what
the first fight of a real run draws from. Map, rewards, shops and events
are still missing; a combat ends the episode.

Action ids use STS2MCP action names (see `actions.py`):
  neow:        "choose_event_option:0"             (skip stub)
  combat:      "play_card:{card_id}"               for non-targeted cards
               "play_card:{card_id}:{enemy_pos}"   for single-enemy cards
                 (enemy_pos = index among alive enemies, matching
                  STS2MCP's battle.enemies list which omits the dead)
               "end_turn"
  hand_select: "combat_select_card:{i}" / "combat_confirm_selection"
               (Brand, Burning Pact, True Grit+, Armaments)
  card_select: "select_card:{i}" / "confirm_selection"   (Headbutt)
  game_over:   "menu_select:main_menu"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import actions
from .combat import CombatContext, CombatError
from .effects import EffectQueue
from .encounters import build_roster
from .enums import Character, CombatPhase, DecisionPoint, Outcome, Screen
from .hooks import HookBus
from .loader import load_all, load_encounters
from .rng import Rng
from .schemas import CardSchema, EncounterSchema, MonsterSchema, PowerSchema
from .state import PlayerState, RunState


_IRONCLAD_START_HP = 80
_IRONCLAD_START_GOLD = 99
_START_ENERGY = 3

IRONCLAD_STARTING_DECK: tuple[str, ...] = (
    "strike_ironclad", "strike_ironclad", "strike_ironclad", "strike_ironclad", "strike_ironclad",
    "defend_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad",
    "bash",
)
ACT1_VARIANTS: tuple[str, ...] = ("overgrowth", "underdocks")
FIRST_COMBAT_POOL = "weak"


@dataclass
class Decision:
    id: str
    text: str


class RunLoopError(Exception):
    pass


class RunLoop:
    """Owns one run's state. Not thread-safe; one RunLoop per worker.

    `encounter` pins the fight (an encounter id from data/encounters.json,
    or a list of monster ids); by default it is drawn per seed.
    """

    def __init__(
        self,
        *,
        character: str = Character.IRONCLAD,
        ascension: int = 0,
        seed: int,
        max_steps: int = 600,
        encounter: str | list[str] | None = None,
    ) -> None:
        if character != Character.IRONCLAD:
            raise RunLoopError(f"character {character!r} not supported")
        if not 0 <= ascension <= 10:
            raise RunLoopError("ascension must be in 0..10")

        self._character = character
        self._ascension = int(ascension)
        self._master_seed = int(seed)
        self._max_steps = int(max_steps)
        self._encounter_pin = encounter

        self._rng: Rng | None = None
        self._state: RunState | None = None
        self._effects: EffectQueue | None = None
        self._hooks: HookBus | None = None
        self._combat_ctx: CombatContext | None = None
        self._cards: dict[str, CardSchema] | None = None
        self._monsters: dict[str, MonsterSchema] | None = None
        self._powers: dict[str, PowerSchema] | None = None
        self._encounters: dict[str, EncounterSchema] | None = None

    # ------------------------------------------------------------------
    # Public surface

    @property
    def rng(self) -> Rng:
        self._require_started()
        assert self._rng is not None
        return self._rng

    @property
    def state(self) -> RunState:
        self._require_started()
        assert self._state is not None
        return self._state

    @property
    def hooks(self) -> HookBus:
        self._require_started()
        assert self._hooks is not None
        return self._hooks

    @property
    def effects(self) -> EffectQueue:
        self._require_started()
        assert self._effects is not None
        return self._effects

    @property
    def combat_ctx(self) -> CombatContext:
        self._require_started()
        assert self._combat_ctx is not None
        return self._combat_ctx

    @property
    def cards(self) -> dict[str, CardSchema]:
        assert self._cards is not None
        return self._cards

    @property
    def monsters(self) -> dict[str, MonsterSchema]:
        assert self._monsters is not None
        return self._monsters

    @property
    def powers(self) -> dict[str, PowerSchema]:
        assert self._powers is not None
        return self._powers

    def reset(self) -> dict[str, Any]:
        self._rng = Rng(self._master_seed)
        self._effects = EffectQueue()
        self._hooks = HookBus()
        self._state = RunState(
            character=self._character,
            ascension=self._ascension,
            seed=self._master_seed,
            player=PlayerState(
                hp=_IRONCLAD_START_HP,
                max_hp=_IRONCLAD_START_HP,
                gold=_IRONCLAD_START_GOLD,
                max_energy=_START_ENERGY,
                energy=_START_ENERGY,
            ),
        )
        if self._cards is None:
            self._powers, self._cards, self._monsters = load_all()
            self._encounters = load_encounters(monster_ids=set(self._monsters))
        assert self._cards is not None and self._monsters is not None and self._powers is not None
        self._combat_ctx = CombatContext(
            run=self._state,
            cards=self._cards,
            monsters=self._monsters,
            powers=self._powers,
            rng=self._rng,
            hooks=self._hooks,
            effects=self._effects,
            encounters=self._encounters,
        )
        return self._packet()

    def step(self, action_id: str) -> dict[str, Any]:
        self._require_started()
        assert self._state is not None and self._combat_ctx is not None
        if self._state.is_terminal():
            raise RunLoopError("run already terminal; call reset()")
        if self._state.steps >= self._max_steps:
            raise RunLoopError("max_steps exceeded")

        legal = {d.id for d in self._decisions()}
        if action_id not in legal:
            raise RunLoopError(f"illegal action {action_id!r} at screen {self._state.screen!r}")

        if self._state.screen == Screen.NEOW:
            self._state.screen = Screen.COMBAT
            monster_ids, flags = self._pick_encounter()
            self._combat_ctx.start_combat(monster_ids, list(IRONCLAD_STARTING_DECK), monster_flags=flags)
            self._maybe_finalize_combat()
        elif self._state.screen == Screen.COMBAT:
            self._handle_combat_action(action_id)
            self._maybe_finalize_combat()
        else:  # pragma: no cover — gated by legal set
            raise RunLoopError(f"unhandled action {action_id!r} at screen {self._state.screen!r}")

        self._state.steps += 1
        return self._packet()

    def close(self) -> None:
        self._rng = None
        self._state = None
        self._effects = None
        self._hooks = None
        self._combat_ctx = None

    # ------------------------------------------------------------------
    # Internals

    def _require_started(self) -> None:
        if self._state is None:
            raise RunLoopError("RunLoop not started; call reset() first")

    def _pick_encounter(self) -> tuple[list[str], list[dict[str, Any]]]:
        assert self._state is not None and self._encounters is not None and self._rng is not None
        pin = self._encounter_pin
        stream = self._rng.stream("encounter")
        if isinstance(pin, list):
            self._state.encounter_id = None
            return list(pin), [{} for _ in pin]
        if isinstance(pin, str):
            enc = self._encounters[pin]
        else:
            act = stream.choice(ACT1_VARIANTS)
            pool = sorted((e for e in self._encounters.values() if e.act == act and e.pool == FIRST_COMBAT_POOL),
                          key=lambda e: e.encounter_id)
            enc = pool[stream.randrange(len(pool))] if pool else None
            if enc is None:
                raise RunLoopError(f"no {FIRST_COMBAT_POOL} encounters for {act}")
        self._state.encounter_id = enc.encounter_id
        roster = build_roster(enc, stream)
        return [mid for mid, _ in roster], [flags for _, flags in roster]

    def _handle_combat_action(self, action_id: str) -> None:
        ctx = self._combat_ctx
        assert ctx is not None
        try:
            name, args = actions.parse_action(action_id)
            sel = ctx.combat.pending_selection
            if name == actions.END_TURN and not args:
                ctx.end_turn()
            elif name == actions.PLAY_CARD and len(args) == 1:
                ctx.play_card(args[0])
            elif name == actions.PLAY_CARD and len(args) == 2:
                ctx.play_card(args[0], target_slot=self._alive_monsters()[int(args[1])].slot)
            elif name in (actions.COMBAT_SELECT_CARD, actions.SELECT_CARD) and sel is not None:
                ctx.toggle_selection(sel.candidates[int(args[0])])
            elif name in (actions.COMBAT_CONFIRM_SELECTION, actions.CONFIRM_SELECTION):
                ctx.confirm_selection()
            else:
                raise RunLoopError(f"unknown combat action {action_id!r}")
        except CombatError as exc:
            raise RunLoopError(f"combat engine rejected {action_id!r}: {exc}") from exc

    def _maybe_finalize_combat(self) -> None:
        assert self._state is not None and self._state.combat is not None
        combat = self._state.combat
        if combat.outcome is None:
            return
        self._state.screen = Screen.GAME_OVER
        self._state.outcome = Outcome.VICTORY if combat.outcome == "victory" else Outcome.DEATH

    def _decisions(self) -> list[Decision]:
        assert self._state is not None
        if self._state.screen == Screen.NEOW:
            return [Decision(id=actions.NEOW_SKIP, text="skip Neow (stub)")]
        if self._state.screen == Screen.COMBAT:
            return self._combat_decisions()
        if self._state.screen == Screen.GAME_OVER:
            return [Decision(id=actions.GAME_OVER_MAIN_MENU, text="<game_over>")]
        raise RunLoopError(f"no decision table for screen {self._state.screen!r}")

    def card_label(self, card_id: str) -> str:
        assert self._cards is not None and self._combat_ctx is not None
        card = self._cards[card_id]
        cost = "X" if card.x_cost else self._combat_ctx.effective_cost(card_id)
        return f"{card.name} ({card.card_type}, cost {cost}): {card.description}".rstrip(": ")

    def _combat_decisions(self) -> list[Decision]:
        assert self._state is not None and self._state.combat is not None and self._cards is not None
        ctx = self._combat_ctx
        assert ctx is not None
        combat = self._state.combat
        player = self._state.player
        assert player is not None
        if combat.phase != CombatPhase.PLAYER or combat.outcome is not None:
            raise RunLoopError(
                f"combat_decisions requested in phase {combat.phase!r} (outcome={combat.outcome!r})"
            )

        sel = combat.pending_selection
        if sel is not None:
            in_hand = sel.source == "hand"
            pile = player.hand if in_hand else player.discard_pile
            pick, confirm = (
                (actions.COMBAT_SELECT_CARD, actions.COMBAT_CONFIRM_SELECTION) if in_hand
                else (actions.SELECT_CARD, actions.CONFIRM_SELECTION)
            )
            out = []
            if len(sel.selected) < sel.max_count:
                for pos, idx in enumerate(sel.candidates):
                    if idx in sel.selected:
                        continue
                    out.append(Decision(id=actions.format_action(pick, pos),
                                        text=f"select {self.card_label(pile[idx])}"))
            if ctx.can_confirm_selection():
                out.append(Decision(id=confirm, text="confirm selection"))
            return out

        decisions: list[Decision] = []
        seen: set[str] = set()
        alive = self._alive_monsters()
        for card_id in player.hand:
            if card_id in seen:
                continue
            seen.add(card_id)
            if not ctx.can_play(card_id):
                continue
            card = self._cards[card_id]
            cost = "X" if card.x_cost else ctx.effective_cost(card_id)
            label = f"play {card.name} (cost {cost}): {card.description}".rstrip(": ")
            if card.target == "single_enemy":
                for pos, monster in enumerate(alive):
                    decisions.append(Decision(
                        id=actions.format_action(actions.PLAY_CARD, card_id, pos),
                        text=f"{label} -> {monster.name}#{pos}[{monster.hp}/{monster.max_hp}]",
                    ))
            else:
                decisions.append(Decision(id=actions.format_action(actions.PLAY_CARD, card_id), text=label))
        decisions.append(Decision(id=actions.END_TURN, text="end turn"))
        return decisions

    def _alive_monsters(self) -> list:
        assert self._state is not None and self._state.combat is not None
        return [m for m in self._state.combat.monsters if m.alive]

    def _packet(self) -> dict[str, Any]:
        assert self._state is not None
        decisions = self._decisions()
        return {
            "screen": self._state.screen,
            "decision_point": self._decision_point(),
            "step": self._state.steps,
            "done": self._state.is_terminal(),
            "outcome": self._state.outcome,
            "floor": self._state.floor,
            "act": self._state.act,
            "candidates": [{"id": d.id, "text": d.text} for d in decisions],
        }

    def _decision_point(self) -> str:
        assert self._state is not None
        if self._state.screen == Screen.COMBAT and self._state.combat is not None:
            sel = self._state.combat.pending_selection
            if sel is not None:
                return DecisionPoint.HAND_SELECT if sel.source == "hand" else DecisionPoint.CARD_SELECT
        return {
            Screen.NEOW: DecisionPoint.NEOW_BONUS,
            Screen.MAP: DecisionPoint.MAP_SELECT,
            Screen.COMBAT: DecisionPoint.COMBAT_PLAY,
            Screen.CARD_REWARD: DecisionPoint.CARD_REWARD,
            Screen.REST: DecisionPoint.REST_SITE,
            Screen.SHOP: DecisionPoint.SHOP,
            Screen.EVENT: DecisionPoint.EVENT_CHOICE,
            Screen.BOSS_RELIC: DecisionPoint.BOSS_RELIC,
            Screen.GAME_OVER: DecisionPoint.GAME_OVER,
        }[self._state.screen]
