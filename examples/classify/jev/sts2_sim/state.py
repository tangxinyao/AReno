"""Run and combat state containers.

The engine (combat.py) mutates these in place. Keep logic out of the
dataclasses themselves — only narrow properties for convenience are
allowed.

Piles hold card ids (strings). Per-instance card state that STS2 keeps on
the card object (Rampage's growth, Infernal Blade's "free this turn") lives
in CombatState side tables keyed by card id; see combat.py for the
approximation that implies when two copies of the same card are in play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    # Move the monster announced for its next turn (its intent).
    queued_move: str | None = None
    # Every move it actually performed, oldest first (the game's StateLog).
    move_history: list[str] = field(default_factory=list)
    # State-machine bookkeeping (monster_ai.py).
    ai_state: str | None = None
    performed_once: set[str] = field(default_factory=set)
    # Position among same-kind monsters when the encounter was built
    # (Nibbit "front", Wriggler "wriggler2", Gardener "third").
    kind_index: int = 0
    stunned: bool = False
    escaped: bool = False
    # Gold taken by Thievery; returned by Heist on death.
    stolen_gold: int = 0
    # Free-form counters for monster-specific mechanics.
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.hp > 0 and not self.escaped


@dataclass
class PendingSelection:
    """A card choice the player must make mid-card (Brand, Headbutt ...)."""

    source: str  # "hand" | "discard"
    candidates: list[int]  # indices into the source pile
    min_count: int
    max_count: int
    purpose: str  # "exhaust" | "upgrade" | "to_draw_top"
    source_card: str
    selected: list[int] = field(default_factory=list)


@dataclass
class CombatState:
    turn: int = 0
    phase: str = CombatPhase.START
    monsters: list[MonsterState] = field(default_factory=list)
    last_card_played: str | None = None
    outcome: str | None = None  # "victory" | "defeat" | None (ongoing)
    # Per-turn counters, reset at the start of each player turn.
    cards_played_this_turn: int = 0
    attacks_played_this_turn: int = 0
    skills_played_this_turn: int = 0
    cards_exhausted_this_turn: int = 0
    hp_lost_this_turn: int = 0
    block_gains_this_turn: int = 0
    # Per-combat counters.
    hp_loss_events_this_combat: int = 0
    # Per-card-id side tables (see module docstring).
    bonus_damage: dict[str, int] = field(default_factory=dict)
    free_this_turn: dict[str, int] = field(default_factory=dict)
    # Player debuffs that existed when the last round ended; a debuff not
    # in here was applied this round and skips its first decrement.
    player_debuffs_at_round_start: set[str] = field(default_factory=set)
    pending_selection: PendingSelection | None = None
    # Two-Tailed Rat backups called this combat (shared by the pack, max 3).
    backup_count: int = 0


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
    encounter_id: str | None = None
    steps: int = 0

    def is_terminal(self) -> bool:
        return self.outcome != Outcome.UNDECIDED or self.screen == Screen.GAME_OVER
