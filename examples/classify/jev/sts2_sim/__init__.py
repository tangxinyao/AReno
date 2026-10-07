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
from .loader import DATA_ROOT, SimDataError, load_all, load_cards, load_enemies, load_powers
from .rng import Rng
from .run import Decision, RunLoop, RunLoopError
from .schemas import (
    CardSchema,
    EffectStep,
    EnemySchema,
    MOVE_RULE_DOCS,
    MoveSchema,
    PowerSchema,
    SelectorEntry,
    VERB_DOCS,
)
from .state import CombatState, MonsterState, PlayerState, RunState

__all__ = [
    "ALL_CHARACTERS",
    "Character",
    "CardSchema",
    "CombatPhase",
    "CombatState",
    "DATA_ROOT",
    "Decision",
    "DecisionPoint",
    "Effect",
    "EffectQueue",
    "EffectStep",
    "EnemySchema",
    "HookBus",
    "HookHandler",
    "MOVE_RULE_DOCS",
    "MonsterState",
    "MoveSchema",
    "VERB_DOCS",
    "Outcome",
    "PlayerState",
    "PowerSchema",
    "Rng",
    "RunLoop",
    "RunLoopError",
    "RunState",
    "Screen",
    "SelectorEntry",
    "SimDataError",
    "load_all",
    "load_cards",
    "load_enemies",
    "load_powers",
]
