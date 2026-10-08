"""RunLoop with Phase 1 combat wired in.

Phase 0 shipped a Neow -> game_over stub; Phase 1 closes that loop by
promoting the Neow skip action into "start the first combat", driving
CombatContext for every player decision, and transitioning to game_over
when combat ends. Later phases replace the hard-coded Jaw Worm fight with
map/event-driven combat encounters.

Action ids use STS2MCP action names (see `actions.py`):
  neow:        "choose_event_option:0"             (skip stub)
  combat:      "play_card:{card_id}"               for non-targeted cards
               "play_card:{card_id}:{enemy_slot}"  for single_enemy cards
               "end_turn"
  game_over:   "menu_select:main_menu"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import actions
from .combat import CombatContext, CombatError
from .effects import EffectQueue
from .enums import Character, CombatPhase, DecisionPoint, Outcome, Screen
from .hooks import HookBus
from .loader import load_all
from .rng import Rng
from .schemas import CardSchema, EnemySchema, PowerSchema
from .state import PlayerState, RunState


_IRONCLAD_START_HP = 80
_IRONCLAD_START_GOLD = 99
_START_ENERGY = 3

# Phase 1 scaffold: fixed starting deck and first encounter. Phase 3 will
# replace this with map-driven selection.
IRONCLAD_STARTING_DECK: tuple[str, ...] = (
    "strike", "strike", "strike", "strike", "strike",
    "defend", "defend", "defend", "defend",
    "bash",
)
PHASE1_FIRST_COMBAT: tuple[str, ...] = ("jaw_worm",)


@dataclass
class Decision:
    id: str
    text: str


class RunLoopError(Exception):
    pass


class RunLoop:
    """Owns one run's state. Not thread-safe; one RunLoop per worker.

    Phase 1 behavior:
      reset()            -> Neow screen, candidate ["choose_event_option:0"].
      step(NEOW_SKIP)    -> transitions to combat (Jaw Worm) and reports
                            the player's combat_play candidates for turn 1.
      step(play/end_turn) -> delegates to CombatContext.
      combat ends        -> screen = game_over, outcome = victory/death.
    """

    def __init__(
        self,
        *,
        character: str = Character.IRONCLAD,
        ascension: int = 0,
        seed: int,
        max_steps: int = 400,
    ) -> None:
        if character != Character.IRONCLAD:
            raise RunLoopError(f"character {character!r} not supported in Phase 1")
        if not 0 <= ascension <= 20:
            raise RunLoopError("ascension must be in 0..20")

        self._character = character
        self._ascension = int(ascension)
        self._master_seed = int(seed)
        self._max_steps = int(max_steps)

        self._rng: Rng | None = None
        self._state: RunState | None = None
        self._effects: EffectQueue | None = None
        self._hooks: HookBus | None = None
        self._combat_ctx: CombatContext | None = None
        self._cards: dict[str, CardSchema] | None = None
        self._enemies: dict[str, EnemySchema] | None = None
        self._powers: dict[str, PowerSchema] | None = None

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
            self._powers, self._cards, self._enemies = load_all()
        assert self._cards is not None and self._enemies is not None and self._powers is not None
        self._combat_ctx = CombatContext(
            run=self._state,
            cards=self._cards,
            enemies=self._enemies,
            powers=self._powers,
            rng=self._rng,
            hooks=self._hooks,
            effects=self._effects,
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
            self._combat_ctx.start_combat(
                list(PHASE1_FIRST_COMBAT),
                list(IRONCLAD_STARTING_DECK),
            )
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

    def _handle_combat_action(self, action_id: str) -> None:
        assert self._combat_ctx is not None
        try:
            name, args = actions.parse_action(action_id)
            if name == actions.END_TURN and not args:
                self._combat_ctx.end_turn()
            elif name == actions.PLAY_CARD and len(args) == 1:
                self._combat_ctx.play_card(args[0])
            elif name == actions.PLAY_CARD and len(args) == 2:
                self._combat_ctx.play_card(args[0], target_slot=int(args[1]))
            else:
                raise RunLoopError(f"unknown combat action {action_id!r}")
        except CombatError as exc:
            # Legal-set check should have caught these, so a CombatError here
            # means a bug in the enumerator. Surface it as a RunLoopError so
            # the test suite flags the mismatch rather than silently losing it.
            raise RunLoopError(f"combat engine rejected {action_id!r}: {exc}") from exc

    def _maybe_finalize_combat(self) -> None:
        assert self._state is not None and self._state.combat is not None
        combat = self._state.combat
        if combat.outcome is None:
            return
        self._state.screen = Screen.GAME_OVER
        self._state.outcome = (
            Outcome.VICTORY if combat.outcome == "victory" else Outcome.DEATH
        )

    def _decisions(self) -> list[Decision]:
        assert self._state is not None
        if self._state.screen == Screen.NEOW:
            return [Decision(id=actions.NEOW_SKIP, text="skip Neow (Phase 1 stub)")]
        if self._state.screen == Screen.COMBAT:
            return self._combat_decisions()
        if self._state.screen == Screen.GAME_OVER:
            return [Decision(id=actions.GAME_OVER_MAIN_MENU, text="<game_over>")]
        raise RunLoopError(f"no decision table for screen {self._state.screen!r}")

    def _combat_decisions(self) -> list[Decision]:
        assert self._state is not None and self._state.combat is not None and self._cards is not None
        combat = self._state.combat
        player = self._state.player
        assert player is not None
        if combat.phase != CombatPhase.PLAYER or combat.outcome is not None:
            # Enemy phase runs synchronously inside CombatContext.end_turn, so
            # from the outside we only ever observe the player phase or a
            # finalized combat. If we land here the engine has a bug — raise
            # loudly instead of returning an empty candidate list.
            raise RunLoopError(
                f"combat_decisions requested in phase {combat.phase!r} "
                f"(outcome={combat.outcome!r})"
            )

        decisions: list[Decision] = []
        seen_card_ids: set[str] = set()
        for card_id in player.hand:
            if card_id in seen_card_ids:
                continue
            seen_card_ids.add(card_id)
            card = self._cards[card_id]
            if card.unplayable:
                continue
            if card.cost > player.energy:
                continue
            if card.target == "single_enemy":
                for monster in combat.monsters:
                    if monster.alive:
                        decisions.append(Decision(
                            id=actions.format_action(actions.PLAY_CARD, card_id, monster.slot),
                            text=f"play {card.name} vs {monster.name}#{monster.slot}",
                        ))
            else:
                decisions.append(Decision(
                    id=actions.format_action(actions.PLAY_CARD, card_id),
                    text=f"play {card.name}",
                ))
        decisions.append(Decision(id=actions.END_TURN, text="end turn"))
        return decisions

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
