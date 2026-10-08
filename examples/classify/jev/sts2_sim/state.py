"""Run and combat state containers.

The engine (combat.py) mutates these in place. Keep logic out of the
dataclasses themselves — only narrow properties for convenience are
allowed.

Piles hold card ids. Inside a combat they are `CardRef`s: strings equal to
the card id (so `cards[ref]` and `"bash" in hand` keep working) that also
carry the per-copy state STS2 keeps on the card object -- Rampage growth,
Frantic Escape's cost increase, Infernal Blade's free-this-turn, Bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import CombatPhase, Outcome, Screen


class CardRef(str):
    """One copy of a card; compares and hashes as its card id.

    Deck copies carry what persists across fights (enchantments); combat copies
    are rebuilt from them each fight and add per-fight state."""

    def __new__(cls, card_id: str, *, like: "CardRef | str | None" = None) -> "CardRef":
        obj = super().__new__(cls, card_id)
        obj.bonus = 0             # Rampage / Thrash growth
        obj.cost_bump = 0         # Frantic Escape: +1 cost per play, this combat
        obj.cost_override = None  # a set cost for the rest of combat (Snecko Oil ...)
        obj.free_turn = False     # costs 0 until the end of this turn
        obj.bound = False         # Chains of Binding
        obj.dampened_from = None  # the upgraded id Dampen took this copy down from
        obj.replay = 0            # extra plays (Hidden Gem, Soldier's Stew)
        obj.retain = False        # Retain for this combat (Anointed+)
        obj.ethereal = False      # Ethereal for this combat (Ghost Seed)
        obj.enchant = None        # Sharp / Adroit / Momentum / Royally Approved / Swift
        obj.enchant_amount = 0
        obj.enchant_spent = False  # Swift: first play only
        if isinstance(like, CardRef):
            for k, v in like.__dict__.items():
                setattr(obj, k, v)
        return obj

    def __reduce__(self):
        return (_rebuild_card_ref, (str(self), dict(self.__dict__)))


def _rebuild_card_ref(card_id: str, attrs: dict) -> CardRef:
    ref = CardRef(card_id)
    ref.__dict__.update(attrs)
    return ref


def as_ref(pile: list, index: int) -> CardRef:
    """pile[index] as a CardRef, upgrading a bare id in place."""

    item = pile[index]
    if not isinstance(item, CardRef):
        item = CardRef(item)
        pile[index] = item
    return item


def remove_ref(pile: list, ref: str) -> None:
    """Remove this exact copy (identity first, then the first equal id)."""

    for i, item in enumerate(pile):
        if item is ref:
            del pile[i]
            return
    pile.remove(ref)


def index_ref(pile: list, ref: str) -> int:
    for i, item in enumerate(pile):
        if item is ref:
            return i
    return pile.index(ref)


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
    # Per-relic counters / flags (Pen Nib count, Lizard Tail used ...).
    relic_state: dict[str, Any] = field(default_factory=dict)
    # Potion belt: one entry per slot, None when empty.
    potions: list[str | None] = field(default_factory=list)

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

    source: str  # "hand" | "discard" | "draw" | "generated" (choose from `options`)
    candidates: list[int]  # indices into the source pile
    min_count: int
    max_count: int
    purpose: str  # "exhaust" | "upgrade" | "to_draw_top"
    source_card: str
    selected: list[int] = field(default_factory=list)
    options: list[str] = field(default_factory=list)  # card ids for source "generated"


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
    hp_lost_this_combat: int = 0
    took_unblocked: bool = False  # unblocked attack damage this combat (Lava Lamp)
    # Player debuffs that existed when the last round ended; a debuff not
    # in here was applied this round and skips its first decrement.
    player_debuffs_at_round_start: set[str] = field(default_factory=set)
    pending_selection: PendingSelection | None = None
    # Two-Tailed Rat backups called this combat (shared by the pack, max 3).
    backup_count: int = 0
    cards_played_this_combat: int = 0
    # Tender: Strength / Dexterity taken this turn, given back at turn end.
    tender_taken: int = 0
    # Chains of Binding: cards bound this turn / whether a Bound card was played.
    bound_this_turn: int = 0
    bound_played_this_turn: bool = False
    # Cards a Thieving Hopper took: (monster, card). Returned if it dies.
    stolen: list[tuple[MonsterState, str]] = field(default_factory=list)
    # Knowledge Demon's pending choice: (generated card ids, disintegration amount).
    curse_choice: tuple[list[str], int] | None = None


@dataclass
class ActRooms:
    """One act's rolled encounters (ActModel.GenerateRooms)."""

    act: str
    normal: list[str]  # weak encounters first, then normal ones
    elite: list[str]
    boss: str
    second_boss: str | None = None


@dataclass
class RewardItem:
    kind: str  # "gold" | "potion" | "relic" | "card"
    gold: int = 0
    potion: str | None = None
    relic: str | None = None
    cards: list[str] = field(default_factory=list)


@dataclass
class DeckSelection:
    """A choice of cards from the master deck (Smith, card removal ...)."""

    purpose: str  # "upgrade" | "remove" | "duplicate" | "enchant"
    candidates: list[int]  # deck indices
    count: int
    source: str  # screen to return to
    selected: list[int] = field(default_factory=list)
    cancelable: bool = True
    price: int = 0
    min_count: int | None = None  # None: exactly `count`
    enchant: str | None = None
    enchant_amount: int = 0


@dataclass
class ShopItem:
    category: str  # "card" | "relic" | "potion" | "card_removal"
    item: str | None
    price: int
    stocked: bool = True
    on_sale: bool = False


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
    # --- run structure ------------------------------------------------------
    act_index: int = 0
    acts: list[ActRooms] = field(default_factory=list)
    map: Any = None  # mapgen.ActMap for the current act
    map_coord: tuple[int, int] | None = None
    normal_visited: int = 0
    elite_visited: int = 0
    unknown_odds: dict[str, float] = field(default_factory=dict)
    card_rarity_offset: float = -0.05
    potion_odds: float = 0.4
    room: str | None = None  # monster / elite / boss / rest / shop / treasure / event / ancient
    last_room: str | None = None
    rewards: list[RewardItem] = field(default_factory=list)
    card_reward: list[str] | None = None
    card_reward_item: int | None = None
    deck_select: DeckSelection | None = None
    rest_used: bool = False
    shop: list[ShopItem] = field(default_factory=list)
    removals_used: int = 0
    treasure_relics: list[str] = field(default_factory=list)
    # Relic grab bags (RelicGrabBag): rarity -> relic ids, front first.
    relic_bag: dict[str, list[str]] = field(default_factory=dict)
    shared_relic_bag: dict[str, list[str]] = field(default_factory=dict)
    # Where leaving the rewards screen goes (a relic's bonus rewards from a shop ...).
    rewards_return: str | None = None
    # Ancient event state: which event (NEOW/OROBAS/...) and the 3 relic choices.
    ancient_event: str | None = None
    ancient_choices: list[str] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.outcome != Outcome.UNDECIDED or self.screen == Screen.GAME_OVER
