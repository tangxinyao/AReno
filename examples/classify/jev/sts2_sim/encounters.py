"""Encounter roster builders (the game's EncounterModel.GenerateMonsters).

Fixed rosters come straight from data/encounters.json. Randomised ones are
built here, following r33hab/sts2 `CombatFactory` (which mirrors the game's
per-encounter generators): which slime / raider / filler appears, and the
starting move ("starter") some encounters deal to each monster.

A roster entry is (monster_id, flags) where flags seed MonsterState.flags
(e.g. {"starter": 1}).
"""

from __future__ import annotations

import random
from typing import Any

from .schemas import EncounterSchema


Roster = list[tuple[str, dict[str, Any]]]


def build_roster(enc: EncounterSchema, rng: random.Random) -> Roster:
    if enc.generator is None:
        return [(mid, {}) for mid in enc.monsters]
    return _GENERATORS[enc.generator](rng)


def _slimes_weak(rng: random.Random) -> Roster:
    # SlimesWeak: small, medium, the other small. The first small opens on its
    # attack; a Leaf Slime (S) in the last slot opens on Goop.
    smalls = ["leaf_slime_s", "twig_slime_s"]
    first = rng.choice(smalls)
    smalls.remove(first)
    second = smalls[0]
    medium = rng.choice(["leaf_slime_m", "twig_slime_m"])
    return [(first, {"starter": 0}), (medium, {}), (second, {"starter": 1})]


def _slimes_normal(rng: random.Random) -> Roster:
    leaf_first = rng.random() < 0.5
    first, second = ("leaf_slime_s", "twig_slime_s") if leaf_first else ("twig_slime_s", "leaf_slime_s")
    return [("twig_slime_m", {}), ("leaf_slime_m", {}), (first, {}), (second, {})]


def _flyconid_normal(rng: random.Random) -> Roster:
    return [(rng.choice(["leaf_slime_m", "twig_slime_m"]), {}), ("flyconid", {})]


def _slithering_strangler(rng: random.Random) -> Roster:
    pick = rng.randrange(3)
    if pick == 0:
        out: Roster = [("snapping_jaxfruit", {})]
    elif pick == 1:
        out = [(rng.choice(["leaf_slime_m", "twig_slime_m"]), {})]
    else:
        out = [(rng.choice(["leaf_slime_s", "twig_slime_s"]), {}) for _ in range(2)]
    return out + [("slithering_strangler", {})]


def _ruby_raiders(rng: random.Random) -> Roster:
    pool = ["axe_ruby_raider", "assassin_ruby_raider", "brute_ruby_raider",
            "crossbow_ruby_raider", "tracker_ruby_raider"]
    out: Roster = []
    for _ in range(3):
        out.append((pool.pop(rng.randrange(len(pool))), {}))
    return out


def _corpse_slugs(count: int):
    def gen(rng: random.Random) -> Roster:
        start = rng.randrange(3)
        return [("corpse_slug", {"starter": (start + i) % 3}) for i in range(count)]

    return gen


def _two_tailed_rats(rng: random.Random) -> Roster:
    first = rng.randrange(3)
    # TwoTailedRatsNormal seats its three rats in slots 2-4 of five.
    return [("two_tailed_rat", {"starter": (first + i) % 3, "rat_slot": 2 + i}) for i in range(3)]


_GENERATORS = {
    "slimes_weak": _slimes_weak,
    "slimes_normal": _slimes_normal,
    "flyconid_normal": _flyconid_normal,
    "slithering_strangler": _slithering_strangler,
    "ruby_raiders": _ruby_raiders,
    "corpse_slugs_2": _corpse_slugs(2),
    "corpse_slugs_3": _corpse_slugs(3),
    "two_tailed_rats": _two_tailed_rats,
}


__all__ = ["Roster", "build_roster"]
