"""Flat enums for the sts2 sim.

Everything here is a plain `str`-valued constant so dataclasses stay JSON-safe
and comparisons do not accidentally depend on `IntEnum` ordering. Expand as
later phases need more states; keep each value snake_case to match the
operator schema.
"""

from __future__ import annotations

from typing import Final


class Character:
    IRONCLAD: Final = "ironclad"


ALL_CHARACTERS: Final = (Character.IRONCLAD,)


class Screen:
    NEOW: Final = "neow"
    MAP: Final = "map"
    COMBAT: Final = "combat"
    CARD_REWARD: Final = "card_reward"
    REST: Final = "rest"
    SHOP: Final = "shop"
    EVENT: Final = "event"
    BOSS_RELIC: Final = "boss_relic"
    GAME_OVER: Final = "game_over"


class Outcome:
    UNDECIDED: Final = "undecided"
    VICTORY: Final = "victory"
    DEATH: Final = "death"


class CombatPhase:
    START: Final = "start"
    PLAYER: Final = "player"
    ENEMY: Final = "enemy"
    END: Final = "end"


class DecisionPoint:
    """Mirrors operator.schema.json so Phase 5 shim mapping stays trivial."""

    NEOW_BONUS: Final = "neow_bonus"
    MAP_SELECT: Final = "map_select"
    COMBAT_PLAY: Final = "combat_play"
    CARD_REWARD: Final = "card_reward"
    REST_SITE: Final = "rest_site"
    SHOP: Final = "shop"
    EVENT_CHOICE: Final = "event_choice"
    BOSS_RELIC: Final = "boss_relic"
    GAME_OVER: Final = "game_over"
