"""RunLoop skeleton.

Phase 0 ties together Rng + state + EffectQueue + HookBus and exposes the
narrowest `reset` / `step` surface the later phases will extend. Right now
`reset` yields a Neow screen and the only legal move is `skip`, which drops
the run straight into game_over. Phase 1 replaces the Neow stub with the real
screen, maps map/combat/etc. in subsequent phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .effects import EffectQueue
from .enums import Character, DecisionPoint, Outcome, Screen
from .hooks import HookBus
from .rng import Rng
from .state import PlayerState, RunState


_IRONCLAD_START_HP = 80
_IRONCLAD_START_GOLD = 99
_START_ENERGY = 3


@dataclass
class Decision:
    """One legal action at the current screen."""

    id: str
    text: str


class RunLoopError(Exception):
    pass


class RunLoop:
    """Owns one run's state. Not thread-safe; mirror parallel rollouts by
    spinning up one RunLoop per worker."""

    def __init__(
        self,
        *,
        character: str = Character.IRONCLAD,
        ascension: int = 0,
        seed: int,
        max_steps: int = 400,
    ) -> None:
        if character != Character.IRONCLAD:
            raise RunLoopError(f"character {character!r} not supported in Phase 0")
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
                energy=_START_ENERGY,
            ),
        )
        return self._packet()

    def step(self, action_id: str) -> dict[str, Any]:
        self._require_started()
        assert self._state is not None
        if self._state.is_terminal():
            raise RunLoopError("run already terminal; call reset()")
        if self._state.steps >= self._max_steps:
            raise RunLoopError("max_steps exceeded")

        legal = {d.id for d in self._decisions()}
        if action_id not in legal:
            raise RunLoopError(f"illegal action {action_id!r} at screen {self._state.screen!r}")

        if self._state.screen == Screen.NEOW and action_id == "skip":
            # Phase 0 stub: skipping Neow ends the run immediately. Phase 1
            # replaces this with the real Neow -> map transition.
            self._state.screen = Screen.GAME_OVER
            self._state.outcome = Outcome.DEATH
        else:  # pragma: no cover — gated by legal set, kept defensively
            raise RunLoopError(f"unhandled action {action_id!r} at screen {self._state.screen!r}")

        self._state.steps += 1
        return self._packet()

    def close(self) -> None:
        self._rng = None
        self._state = None
        self._effects = None
        self._hooks = None

    # ------------------------------------------------------------------
    # Internals

    def _require_started(self) -> None:
        if self._state is None:
            raise RunLoopError("RunLoop not started; call reset() first")

    def _decisions(self) -> list[Decision]:
        assert self._state is not None
        if self._state.screen == Screen.NEOW:
            return [Decision(id="skip", text="skip Neow (Phase 0 stub)")]
        if self._state.screen == Screen.GAME_OVER:
            return [Decision(id="terminal", text="<game_over>")]
        raise RunLoopError(f"no decision table for screen {self._state.screen!r}")

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
