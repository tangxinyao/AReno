"""Combat data contract: cards, enemies, powers as frozen dataclasses.

Phase 1 data lives as JSON under `data/`; this module defines the shapes the
loader validates against. Keep schemas *data-only* — no behavior lives on
these objects, so Phase 1 engine can evolve independently.

Allowed vocabulary (verbs, target scopes, intents, card types) lives in sets
below. The loader rejects any value outside these sets with a message that
names the offending card/enemy/field. Add a verb only when a card actually
needs it; this keeps the engine's switch-table bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


# ---------------------------------------------------------------------------
# Allowed vocabulary

CARD_TYPES: Final = frozenset({"attack", "skill", "power", "status", "curse"})
CARD_RARITIES: Final = frozenset({"basic", "common", "uncommon", "rare", "special"})
CARD_TARGETS: Final = frozenset({"none", "self", "single_enemy", "all_enemies", "random_enemy"})

INTENTS: Final = frozenset({
    "attack",
    "attack_defend",
    "attack_buff",
    "attack_debuff",
    "defend",
    "defend_buff",
    "defend_debuff",
    "buff",
    "debuff",
    "strong_debuff",
    "stun",
    "sleep",
    "escape",
    "magic",
    "unknown",
})

TARGET_SCOPES: Final = frozenset({
    "self",            # the actor (player for cards, enemy for enemy moves)
    "player",          # the player (used by enemy moves)
    "single_enemy",    # one chosen enemy (player picks when card is played)
    "all_enemies",     # every alive enemy
    "random_enemy",    # engine picks uniformly at random
})

# Verbs the engine (Phase 1+) must implement. Each maps to a dict of required
# arg -> expected Python type. `target_scope` is validated separately against
# TARGET_SCOPES below.
EFFECT_VERBS: Final[dict[str, dict[str, type]]] = {
    "deal_damage":      {"amount": int, "target_scope": str, "hits": int},
    "gain_block":       {"amount": int, "target_scope": str},
    "apply_power":      {"power_id": str, "amount": int, "target_scope": str},
    "draw_cards":       {"amount": int},
    "copy_to_discard":  {},
}

# Which scopes each verb accepts. The loader cross-checks the step's
# target_scope against this map.
VERB_ALLOWED_SCOPES: Final[dict[str, frozenset[str]]] = {
    "deal_damage":   frozenset({"single_enemy", "all_enemies", "random_enemy", "player"}),
    "gain_block":    frozenset({"self"}),
    "apply_power":   frozenset({"self", "single_enemy", "all_enemies", "random_enemy", "player"}),
}

# Verbs cards may use. Enemy moves use the complement defined below.
CARD_ONLY_VERBS: Final = frozenset({"draw_cards", "copy_to_discard"})
ENEMY_ONLY_VERBS: Final = frozenset()  # no enemy-exclusive verbs yet

POWER_KINDS: Final = frozenset({"buff", "debuff"})
POWER_DURATIONS: Final = frozenset({"permanent", "turns", "end_of_turn_tick"})

MOVE_RULES: Final = frozenset({
    "always_first",     # played on combat turn 1 only
    "sequential",       # played at sequence_index after always_first entries
    "weighted",         # weighted random from the remaining pool
    "if_not_last",      # weighted, but excluded if played last turn
    "if_not_two",       # weighted, excluded if played the last two turns
})


# ---------------------------------------------------------------------------
# Schemas

@dataclass(frozen=True)
class EffectStep:
    verb: str
    args: dict[str, Any]


@dataclass(frozen=True)
class CardSchema:
    card_id: str
    name: str
    cost: int
    card_type: str
    rarity: str
    target: str
    effects: tuple[EffectStep, ...]
    upgraded_from: str | None = None
    upgrade_of: str | None = None  # points to upgraded form id for base cards


@dataclass(frozen=True)
class MoveSchema:
    move_id: str
    intent: str
    effects: tuple[EffectStep, ...]


@dataclass(frozen=True)
class SelectorEntry:
    move_id: str
    rule: str
    weight: int = 1
    sequence_index: int | None = None


@dataclass(frozen=True)
class EnemySchema:
    enemy_id: str
    name: str
    hp_min: int
    hp_max: int
    moves: dict[str, MoveSchema]
    movepicker: tuple[SelectorEntry, ...]


@dataclass(frozen=True)
class PowerSchema:
    power_id: str
    name: str
    kind: str
    duration: str
    stacks: bool = True


__all__ = [
    "CARD_ONLY_VERBS",
    "CARD_RARITIES",
    "CARD_TARGETS",
    "CARD_TYPES",
    "CardSchema",
    "EFFECT_VERBS",
    "ENEMY_ONLY_VERBS",
    "EffectStep",
    "EnemySchema",
    "INTENTS",
    "MOVE_RULES",
    "MoveSchema",
    "POWER_DURATIONS",
    "POWER_KINDS",
    "PowerSchema",
    "SelectorEntry",
    "TARGET_SCOPES",
    "VERB_ALLOWED_SCOPES",
]
