"""Pure-Python Slay the Spire 2 simulator for the Jev scoring policy.

Content targets STS2 v0.107.1: all 87 Ironclad cards (plus statuses, curses
and Giant Rock), every monster and encounter of the three acts (Overgrowth
or Underdocks, Hive, Glory), act maps, combat rewards, rest sites, shops and
treasure rooms. `RunLoop.reset()` yields a Neow screen and a run goes on to
the last boss or the player's death. See `NOTICE.md` for sources.
"""

from __future__ import annotations

from . import actions, mapgen, rewards
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
from .state import CardRef, CombatState, MonsterState, PlayerState, RunState

__all__ = [
    "actions",
    "mapgen",
    "rewards",
    "ALL_CHARACTERS",
    "Character",
    "CardRef",
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
