"""Combat data contract: cards, monsters, powers, encounters.

Data lives as JSON under `data/`; this module defines the frozen shapes the
loader produces and the vocabulary it validates against. Behavior does not
live here:

  * card play effects are Python functions keyed by `game_id`
    (card_effects.py) and read their numbers from `CardSchema.vars`;
  * monster moves are lists of `EffectStep`s over the small monster verb set
    below, picked by a per-monster state machine (monster_ai.py);
  * power behavior is engine code keyed by power id (combat.py).

All content targets Slay the Spire 2 public build v0.107.1 (Steam build
23811903). See NOTICE.md for sources.

Damage / block rules (BuffSystem semantics, applied by combat.py):
  * Attack damage per hit = int((base + Strength + Vigor) * Weak 0.75
    * Shrink 0.7 * Vulnerable (1.5 + Cruelty%)), truncated once at the end.
    Weak/Vulnerable/Shrink multiply only "powered" damage (attacks from
    Attack cards and monster attack moves); power/status/thorns damage is
    unpowered and ignores Strength, Weak and Vulnerable.
  * Card block = int((base + Dexterity) * Frail 0.75); unpowered block
    (Plating, Feel No Pain, Rage ...) ignores both.
  * Intangible caps every damage instance and HP loss at 1.
  * Duration debuffs (Vulnerable, Weak, Frail) tick down at the end of each
    round (after the enemy turn) on both sides; a debuff the player gained
    during the current round skips that round's tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping


# ---------------------------------------------------------------------------
# Cards

CARD_TYPES: Final = frozenset({"attack", "skill", "power", "status", "curse"})
CARD_RARITIES: Final = frozenset({
    "basic", "common", "uncommon", "rare", "ancient", "token", "status", "curse", "event", "quest",
})
CARD_TARGETS: Final = frozenset({"none", "self", "single_enemy", "all_enemies", "random_enemy", "ally", "all_allies"})
CARD_KEYWORDS: Final = frozenset({"exhaust", "ethereal", "innate", "retain", "eternal", "sly"})
CARD_COLORS: Final = frozenset({"ironclad", "status", "curse", "token", "colorless"})
# Rarities a character's card reward / in-combat generator may roll.
POOL_RARITIES: Final = frozenset({"common", "uncommon", "rare"})


@dataclass(frozen=True)
class CardSchema:
    """One card printing. Upgraded printings are separate entries ("bash+1").

    `cost` is the printed energy cost (0 for X-cost cards, which set
    `x_cost`). `unplayable` covers status/curse cards with no energy cost.
    `vars` holds every number the rules text uses, already upgraded for "+1"
    entries; card_effects.py reads them by name.
    """

    card_id: str
    game_id: str
    name: str
    color: str
    card_type: str
    rarity: str
    target: str
    cost: int
    description: str
    vars: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    keywords: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()
    x_cost: bool = False
    unplayable: bool = False
    multiplayer_only: bool = False
    generated_in_combat: bool = True
    upgrade_of: str | None = None
    upgraded_from: str | None = None

    @property
    def upgraded(self) -> bool:
        return self.upgraded_from is not None

    @property
    def exhausts(self) -> bool:
        return "exhaust" in self.keywords

    @property
    def ethereal(self) -> bool:
        return "ethereal" in self.keywords

    def v(self, name: str) -> int:
        return int(self.vars[name])


# ---------------------------------------------------------------------------
# Powers

POWER_KINDS: Final = frozenset({"buff", "debuff"})
POWER_STACKS: Final = frozenset({"counter", "single", "none"})


@dataclass(frozen=True)
class PowerSchema:
    """Power metadata; behavior is engine code keyed by `power_id`."""

    power_id: str
    name: str
    kind: str
    stack: str
    description: str


# ---------------------------------------------------------------------------
# Potions / relics

POTION_RARITIES: Final = frozenset({"common", "uncommon", "rare", "token", "event"})
POTION_TARGETS: Final = frozenset({"self", "single_enemy", "all_enemies", "none"})
RELIC_RARITIES: Final = frozenset({"starter", "common", "uncommon", "rare", "shop", "ancient", "event"})


@dataclass(frozen=True)
class PotionSchema:
    """A potion; behavior is potion_effects.py keyed by `game_id`."""

    potion_id: str
    game_id: str
    name: str
    rarity: str
    pool: str  # "ironclad" | "shared" | "token"
    target: str
    description: str


@dataclass(frozen=True)
class RelicSchema:
    """A relic; behavior is relics.py keyed by `relic_id`."""

    relic_id: str
    game_id: str
    name: str
    rarity: str
    pool: str  # "ironclad" | "shared"
    description: str
    in_shops: bool = True
    # Gold / deck-growth relics the game stops offering after the Act 3 chest (floor 41).
    stops_after_act3_chest: bool = False


# ---------------------------------------------------------------------------
# Monsters

MONSTER_KINDS: Final = frozenset({"normal", "elite", "boss"})

INTENTS: Final = frozenset({
    "attack", "defend", "buff", "debuff", "strong_debuff", "status", "card_debuff",
    "summon", "heal", "stun", "sleep", "escape", "death_blow", "unknown",
})

# Monster move verbs. Every numeric arg may be an int, or a list
# [base, ascension_value, ascension_level]: the second value applies when the
# run's ascension >= level (8 = Tough Enemies, 9 = Deadly Enemies).
MONSTER_VERBS: Final[dict[str, dict[str, Any]]] = {
    "attack":      {"required": {"damage"}, "optional": {"hits", "hits_grow", "plus_dexterity"}},
    "block":       {"required": {"amount"}, "optional": set()},
    "apply_power": {"required": {"power", "amount", "target"}, "optional": set()},
    "add_card":    {"required": {"card", "pile", "count"}, "optional": set()},
    "heal":        {"required": {"amount"}, "optional": set()},
    "summon":      {"required": {"monster"}, "optional": {"count", "where", "stunned", "max_alive"}},
    "steal_gold":  {"required": {"amount"}, "optional": set()},
    "escape":      {"required": set(), "optional": set()},
    "suicide":     {"required": set(), "optional": set()},
    "special":     {"required": {"name"}, "optional": {"amount"}},
}

POWER_TARGETS: Final = frozenset({"self", "player", "allies", "all"})
CARD_PILES: Final = frozenset({"hand", "discard", "draw_random", "draw_top"})

MONSTER_VERB_DOCS: Final[dict[str, str]] = {
    "attack": "Powered attack on the player: `hits` hits of `damage` (Strength, Vigor, "
              "Weak, Shrink, Vulnerable apply per hit). Fires Thorns/Flame Barrier/Suck. "
              "`hits_grow`: +1 hit per earlier use of the move; `plus_dexterity`: adds the "
              "monster's Dexterity to the damage.",
    "block": "The monster gains `amount` block (no Dexterity/Frail).",
    "apply_power": "Apply `amount` of `power` to self / player / every other living ally "
                   "('allies') / every living monster ('all'). "
                   "Negative amounts on strength/dexterity reduce the stat; debuffs on the "
                   "player respect Artifact.",
    "add_card": "Add `count` copies of status `card` to the player's `pile` "
                "(discard / hand / draw_random = random draw-pile position / draw_top).",
    "heal": "The monster heals `amount` HP (capped at max HP).",
    "summon": "Spawn `count` of `monster` ('before' = left of the summoner, 'after' = right, "
              "'end' = rightmost). Skipped when `max_alive` of that monster already live.",
    "steal_gold": "Take up to `amount` gold from the player (Thievery); returned by Heist on death.",
    "escape": "The monster leaves combat (counts as gone, not killed).",
    "suicide": "The monster dies after its move (Gas Bomb's Explode).",
    "special": "Named monster-specific effect implemented in monster_ai.SPECIALS.",
}

# State-machine node kinds and random-branch repeat rules, mirroring the
# game's MonsterMoveStateMachine (MoveState / RandomBranchState /
# ConditionalBranchState).
AI_NODE_TYPES: Final = frozenset({"move", "random", "conditional"})
REPEAT_RULES: Final = frozenset({"can_repeat", "cannot_repeat", "can_repeat_x", "use_only_once"})
AI_CONDITIONS: Final = frozenset({
    "alone",          # no other living monster of the same kind
    "front",          # first living monster of the same kind
    "not_front",
    "kind_index",     # args.n == position among same kind at encounter build
    "starter",        # args == starter index the encounter dealt (slugs, rats, slimes)
    "has_power",      # the monster currently has power `arg`
    "flag",           # monster flag `arg` is set (Bowlbug Rock's off-balance)
    "side_alive_lt",  # fewer than `arg` living monsters on its side, itself included
    "monster_alive",  # some living monster has id `arg`
    "below_half_once",  # HP below half and move `arg` never performed
    "move_count_lt",  # "MOVE:n": the move was performed fewer than n times
})


@dataclass(frozen=True)
class EffectStep:
    verb: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class MoveSchema:
    move_id: str
    name: str
    intents: tuple[str, ...]
    effects: tuple[EffectStep, ...]


@dataclass(frozen=True)
class AiBranch:
    target: str  # state id
    weight: float = 1.0
    repeat: str = "can_repeat"
    max_times: int = 0
    cooldown: int = 0
    condition: str | None = None
    condition_arg: int | str | None = None


@dataclass(frozen=True)
class AiNode:
    node_id: str
    node_type: str
    move: str | None = None
    next: str | None = None
    must_perform_once: bool = False
    branches: tuple[AiBranch, ...] = ()


@dataclass(frozen=True)
class MonsterSchema:
    """Authored monster definition.

    `hp` / `hp_asc` are inclusive [min, max] rolls; `hp_asc` applies from
    Ascension 8 (Tough Enemies). `innate_powers` are applied when the
    monster enters combat: (power_id, amount, amount_at_asc, asc_level).
    `ai_initial` is the state id the machine starts in.
    """

    monster_id: str
    game_id: str
    name: str
    kind: str
    hp: tuple[int, int]
    hp_asc: tuple[int, int]
    moves: Mapping[str, MoveSchema]
    ai_nodes: Mapping[str, AiNode]
    ai_initial: str
    innate_powers: tuple[tuple[str, int, int, int], ...] = ()
    starting_block: int | tuple[int, int, int] = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Encounters

ROOM_TYPES: Final = frozenset({"monster", "elite", "boss", "event"})


@dataclass(frozen=True)
class EncounterSchema:
    """An encounter: which monsters appear, in slot order.

    `pool` is the act pool it is drawn from (weak / normal / elite / boss).
    Either `monsters` is a fixed roster or `generator` names a roster builder
    in encounters.py (random slime mixes, raider picks, slug/rat starters).
    """

    encounter_id: str
    name: str
    act: str
    pool: str
    room_type: str
    monsters: tuple[str, ...] = ()
    generator: str | None = None
    # EncounterTags: two encounters sharing a tag are never rolled back to back.
    tags: tuple[str, ...] = ()


__all__ = [
    "AI_CONDITIONS",
    "AI_NODE_TYPES",
    "AiBranch",
    "AiNode",
    "CARD_COLORS",
    "CARD_KEYWORDS",
    "CARD_PILES",
    "CARD_RARITIES",
    "CARD_TARGETS",
    "CARD_TYPES",
    "CardSchema",
    "EffectStep",
    "EncounterSchema",
    "INTENTS",
    "MONSTER_KINDS",
    "MONSTER_VERBS",
    "MONSTER_VERB_DOCS",
    "MonsterSchema",
    "MoveSchema",
    "POOL_RARITIES",
    "POWER_KINDS",
    "POWER_STACKS",
    "POWER_TARGETS",
    "POTION_RARITIES",
    "POTION_TARGETS",
    "PotionSchema",
    "PowerSchema",
    "RELIC_RARITIES",
    "RelicSchema",
    "REPEAT_RULES",
    "ROOM_TYPES",
]
