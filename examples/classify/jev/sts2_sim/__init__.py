"""Pure-Python Slay the Spire 2 simulator for the Jev scoring policy.

Phase 0 only exposes the skeleton types (Rng, HookBus, EffectQueue, RunLoop)
needed by later phases; `RunLoop.reset()` yields a Neow screen and
`step("skip")` drops the run into `game_over` immediately. See `NOTICE.md`
for attribution against the three reference sims.
"""

from __future__ import annotations

from .effects import Effect, EffectQueue
from .enums import ALL_CHARACTERS, Character, CombatPhase, DecisionPoint, Outcome, Screen
from .hooks import HookBus, HookHandler
from .rng import Rng
from .run import Decision, RunLoop, RunLoopError
from .state import CombatState, MonsterState, PlayerState, RunState

__all__ = [
    "ALL_CHARACTERS",
    "Character",
    "CombatPhase",
    "CombatState",
    "Decision",
    "DecisionPoint",
    "Effect",
    "EffectQueue",
    "HookBus",
    "HookHandler",
    "MonsterState",
    "Outcome",
    "PlayerState",
    "Rng",
    "RunLoop",
    "RunLoopError",
    "RunState",
    "Screen",
]
