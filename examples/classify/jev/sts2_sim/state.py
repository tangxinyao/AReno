"""Run and combat state containers.

Phase 0 keeps these intentionally thin: only the fields a bare `reset` ->
`step(no-op)` -> `terminal` loop touches. Phase 1 extends `PlayerState` with
deck/hand/piles and introduces `MonsterState` + `CombatState`; both are
sketched here as empty placeholders so later phases can grow them without
widening every call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import Outcome, Screen


@dataclass
class PlayerState:
    """Minimal player book-keeping. Combat details land in Phase 1."""

    hp: int
    max_hp: int
    gold: int
    energy: int
    relics: list[str] = field(default_factory=list)


@dataclass
class MonsterState:
    """Placeholder — fields filled in Phase 1 when combat lands."""

    monster_id: str
    hp: int
    max_hp: int


@dataclass
class CombatState:
    """Placeholder — populated in Phase 1."""

    turn: int = 0
    monsters: list[MonsterState] = field(default_factory=list)


@dataclass
class RunState:
    character: str
    ascension: int
    seed: int
    floor: int = 0
    act: int = 1
    screen: str = Screen.NEOW
    outcome: str = Outcome.UNDECIDED
    player: PlayerState | None = None
    combat: CombatState | None = None
    steps: int = 0

    def is_terminal(self) -> bool:
        return self.outcome != Outcome.UNDECIDED or self.screen == Screen.GAME_OVER
