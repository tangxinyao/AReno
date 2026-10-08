"""Combat reward rolls (r33hab/sts2 `RunRewardGenerator`, v0.107.1 rules).

  * Gold: Monster 10-20, Elite 35-45, Boss 100 (Poverty, A3+: x0.75 on the
    bounds).
  * Card reward: three Ironclad cards, no duplicates. Each card's rarity is
    rolled against Rare / Uncommon odds plus a pity offset that starts at
    -5%, grows 1% (0.5% under Scarcity, A7+) per non-Rare roll up to +40%,
    and resets on a Rare. Monster rooms 3% Rare / 37% Uncommon, Elites 10% /
    40% (Scarcity halves the Rare odds), Bosses always Rare. A missing rarity
    falls back (Common -> Uncommon -> Rare; Uncommon -> Rare -> Common; Rare
    -> Common -> Uncommon). Non-Rare cards are pre-upgraded with chance
    0.25 x act index (0.125 under Scarcity).
  * Potion drop: 40% to start, +10% after every fight without one and -10%
    after every fight with one; Elites add 12.5%.
"""

from __future__ import annotations

import random

from .schemas import CardSchema

RARE, UNCOMMON, COMMON = "rare", "uncommon", "common"
CARD_RARITY_BASE_OFFSET = -0.05
CARD_RARITY_MAX_OFFSET = 0.4
POTION_REWARD_STEP = 0.1
POVERTY = 3
# Splash offers Attacks from the other characters' pools, which the sim does not model.
UNSUPPORTED_CARDS = frozenset({"splash"})
SCARCITY = 7
CARD_REWARD_SIZE = 3

_FALLBACKS = {
    COMMON: (COMMON, UNCOMMON, RARE),
    UNCOMMON: (UNCOMMON, RARE, COMMON),
    RARE: (RARE, COMMON, UNCOMMON),
}


def reward_pool(cards: dict[str, CardSchema], color: str = "ironclad") -> list[str]:
    """Cards a reward / shop may offer: the color's Common/Uncommon/Rare, solo-legal."""

    return sorted(cid for cid, c in cards.items()
                  if c.color == color and not c.upgraded and not c.multiplayer_only
                  and cid not in UNSUPPORTED_CARDS and c.rarity in (COMMON, UNCOMMON, RARE))


def poverty_gold(ascension: int, gold: int) -> int:
    return int(gold * 0.75) if ascension >= POVERTY else gold


def combat_gold(room: str, ascension: int, rng: random.Random) -> int:
    if room == "elite":
        return rng.randint(poverty_gold(ascension, 35), poverty_gold(ascension, 45))
    if room == "boss":
        return poverty_gold(ascension, 100)
    return rng.randint(poverty_gold(ascension, 10), poverty_gold(ascension, 20))


def card_odds(room: str, ascension: int) -> tuple[float, float]:
    scarce = ascension >= SCARCITY
    if room == "boss":
        return 1.0, 0.0
    if room == "elite":
        return (0.05 if scarce else 0.1), 0.4
    if room == "shop":
        return (0.045 if scarce else 0.09), 0.37
    return (0.0149 if scarce else 0.03), 0.37


def roll_rarity(offset: float, odds: tuple[float, float], rng: random.Random, *, ascension: int,
                mutate: bool) -> tuple[str, float]:
    """One rarity roll; returns (rarity, new_offset)."""

    rare, uncommon = odds
    off = 0.0 if rare >= 1.0 else offset
    roll = rng.random()
    rare_threshold = rare + off
    rarity = RARE if roll < rare_threshold else UNCOMMON if roll < rare_threshold + uncommon else COMMON
    if mutate:
        growth = 0.005 if ascension >= SCARCITY else 0.01
        offset = CARD_RARITY_BASE_OFFSET if rarity == RARE else min(offset + growth, CARD_RARITY_MAX_OFFSET)
    return rarity, offset


def choose_card(pool: list[str], cards: dict[str, CardSchema], rarity: str, blacklist: list[str],
                rng: random.Random) -> str:
    for r in _FALLBACKS[rarity]:
        avail = [c for c in pool if c not in blacklist and cards[c].rarity == r]
        if avail:
            return rng.choice(avail)
    rest = [c for c in pool if c not in blacklist]
    return rng.choice(rest or pool)


def roll_upgrade(card: CardSchema, act_index: int, ascension: int, rng: random.Random) -> bool:
    roll = rng.random()
    if card.upgrade_of is None:
        return False
    odds = 0.0 if card.rarity == RARE else act_index * (0.125 if ascension >= SCARCITY else 0.25)
    return roll <= odds and odds > 0


def card_reward(cards: dict[str, CardSchema], pool: list[str], room: str, *, ascension: int, act_index: int,
                offset: float, rng: random.Random, size: int = CARD_REWARD_SIZE) -> tuple[list[str], float]:
    """Roll one card reward; returns (card ids, upgraded where rolled; new pity offset)."""

    out: list[str] = []
    picked: list[str] = []
    odds = card_odds(room, ascension)
    for _ in range(size):
        rarity, offset = roll_rarity(offset, odds, rng, ascension=ascension, mutate=True)
        cid = choose_card(pool, cards, rarity, picked, rng)
        picked.append(cid)
        if roll_upgrade(cards[cid], act_index, ascension, rng):
            cid = cards[cid].upgrade_of or cid
        out.append(cid)
    return out, offset


def roll_potion_drop(odds: float, room: str, rng: random.Random) -> tuple[bool, float]:
    bonus = 0.25 * 0.5 if room == "elite" else 0.0
    if rng.random() < odds + bonus:
        return True, odds - POTION_REWARD_STEP
    return False, odds + POTION_REWARD_STEP


def transform_options(cards: dict[str, CardSchema], card_id: str) -> list[str]:
    """RunRewardGenerator.TransformOptionsFor: same pool, Common/Uncommon/Rare, not itself.

    Ancient / Event / Token cards fall back to the Colorless pool; a Status or Curse
    keeps every rarity (a curse transforms into another curse)."""

    orig = cards[card_id]
    base = orig.upgraded_from or orig.card_id
    if orig.rarity in ("ancient", "event", "token"):
        color = "colorless"
    else:
        color = orig.color
    keep_all = orig.rarity in ("status", "curse")
    return sorted(cid for cid, c in cards.items()
                  if c.color == color and not c.upgraded and not c.multiplayer_only and cid != base
                  and cid not in UNSUPPORTED_CARDS
                  and (keep_all or c.rarity in (COMMON, UNCOMMON, RARE)))


def shop_card_price(rarity: str, *, colorless: bool, rng: random.Random) -> int:
    base = {RARE: 150, UNCOMMON: 75}.get(rarity, 50)
    if colorless:
        base = round(base * 1.15)
    return round(base * rng.uniform(0.95, 1.05))


__all__ = [
    "CARD_RARITY_BASE_OFFSET", "card_odds", "card_reward", "choose_card", "combat_gold", "poverty_gold",
    "reward_pool", "roll_potion_drop", "roll_rarity", "roll_upgrade", "shop_card_price", "transform_options",
    "UNSUPPORTED_CARDS",
]
