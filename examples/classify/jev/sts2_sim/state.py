"""Run and combat state containers.

Phase 1 now carries full combat fields. The engine (combat.py) mutates
these in place. Keep logic out of the dataclasses themselves — only
narrow properties for convenience are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import CombatPhase, Outcome, Screen


@dataclass
class PlayerState:
    hp: int
    max_hp: int
    gold: int
    max_energy: int = 3
    energy: int = 0
    block: int = 0
    powers: dict[str, int] = field(default_factory=dict)
    powers_applied_on_turn: dict[str, int] = field(default_factory=dict)
    powers_applied_phase: dict[str, str] = field(default_factory=dict)
    deck: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    draw_pile: list[str] = field(default_factory=list)
    discard_pile: list[str] = field(default_factory=list)
    exhaust_pile: list[str] = field(default_factory=list)
    relics: list[str] = field(default_factory=list)

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class MonsterState:
    monster_id: str
    name: str
    hp: int
    max_hp: int
    slot: int
    block: int = 0
    powers: dict[str, int] = field(default_factory=dict)
    powers_applied_on_turn: dict[str, int] = field(default_factory=dict)
    powers_applied_phase: dict[str, str] = field(default_factory=dict)
    queued_move: str | None = None
    move_history: list[str] = field(default_factory=list)

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class CombatState:
    turn: int = 0
    phase: str = CombatPhase.START
    monsters: list[MonsterState] = field(default_factory=list)
    last_card_played: str | None = None
    outcome: str | None = None  # "victory" | "defeat" | None (ongoing)


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
