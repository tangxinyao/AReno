"""Pure-Python Slay the Spire 2 simulator for the Jev scoring policy.

Content targets STS2 v0.107.1: all 87 Ironclad cards (plus statuses, curses
and Giant Rock) and every Act 1 monster and encounter (Overgrowth and
Underdocks). `RunLoop.reset()` yields a Neow screen and
`step(actions.NEOW_SKIP)` starts an Act 1 weak-pool combat. See `NOTICE.md`
for sources and attribution.
"""

from __future__ import annotations

from . import actions
from .combat import CombatContext, CombatError
from .effects import Effect, EffectQueue
from .enums import ALL_CHARACTERS, Character, CombatPhase, DecisionPoint, Outcome, Screen
from .hooks import HookBus, HookHandler
from .loader import DATA_ROOT, SimDataError, load_all, load_cards, load_encounters, load_monsters, load_powers
from .rng import Rng
from .run import Decision, RunLoop, RunLoopError
from .schemas import (
    CardSchema,
    EffectStep,
    EncounterSchema,
    MONSTER_VERB_DOCS,
    MonsterSchema,
    MoveSchema,
    PowerSchema,
)
from .state import CombatState, MonsterState, PlayerState, RunState

__all__ = [
    "actions",
    "ALL_CHARACTERS",
    "Character",
    "CardSchema",
    "CombatContext",
    "CombatError",
    "CombatPhase",
    "CombatState",
    "DATA_ROOT",
    "Decision",
    "DecisionPoint",
    "Effect",
    "EffectQueue",
    "EffectStep",
    "EncounterSchema",
    "HookBus",
    "HookHandler",
    "MONSTER_VERB_DOCS",
    "MonsterSchema",
    "MonsterState",
    "MoveSchema",
    "Outcome",
    "PlayerState",
    "PowerSchema",
    "Rng",
    "RunLoop",
    "RunLoopError",
    "RunState",
    "Screen",
    "SimDataError",
    "load_all",
    "load_cards",
    "load_encounters",
    "load_monsters",
    "load_powers",
]
