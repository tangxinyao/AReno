"""Potion effects, one function per potion, keyed by the game's potion id.

Numbers are the v0.107.1 potion values (Spire Codex text), semantics from
r33hab/sts2 `PotionEffects.cs`. Each function receives the CombatContext
`c` and a `PotionUse` `u` (potion, target). Potions that make the player
choose cards are generators yielding SelectionRequests, like cards.

Fairy in a Bottle is not drunk: combat.py consumes it when HP would hit 0.
"""

from __future__ import annotations

import random
from typing import Callable, Iterator

from .combat import CombatContext, PotionUse, SelectionRequest
from .schemas import PotionSchema
from .state import CardRef, remove_ref


PotionFn = Callable[[CombatContext, PotionUse], object]
POTION_EFFECTS: dict[str, PotionFn] = {}
_RARITY_ROLL = (("rare", 0.1), ("uncommon", 0.35))


def potion(game_id: str):
    def deco(fn: PotionFn) -> PotionFn:
        POTION_EFFECTS[game_id] = fn
        return fn

    return deco


def reward_potions(potions: dict[str, PotionSchema]) -> list[PotionSchema]:
    """The Ironclad pool then the shared one (PotionFactory.GetPotionOptions)."""

    return [p for p in potions.values() if p.pool == "ironclad"] + [p for p in potions.values() if p.pool == "shared"]


def roll_potion_rarity(rng: random.Random) -> str:
    roll = rng.random()
    for rarity, threshold in _RARITY_ROLL:
        if roll <= threshold:
            return rarity
    return "common"


def random_potion(potions: dict[str, PotionSchema], rng: random.Random, *, blacklist: list[str] | None = None) -> str:
    """RunRewardGenerator.NextPotion: roll a rarity, then a potion of it."""

    pool = reward_potions(potions)
    rarity = roll_potion_rarity(rng)
    blocked = set(blacklist or ())
    options = [p.potion_id for p in pool if p.rarity == rarity and p.potion_id not in blocked]
    return rng.choice(options or [p.potion_id for p in pool])


def _choose_generated(c: CombatContext, options: list[str]) -> Iterator[SelectionRequest]:
    """Choose 1 of the options to add to hand, free this turn; the screen can be skipped."""

    if not options:
        return
    chosen = yield SelectionRequest(source="generated", candidates=list(range(len(options))), count=1,
                                    purpose="to_hand", min_count=0, options=options)
    for cid in chosen:
        c.add_to_hand(cid).free_turn = True


@potion("ASHWATER")
def ashwater(c, u) -> Iterator[SelectionRequest]:
    hand = c.player.hand
    if not hand:
        return
    chosen = yield SelectionRequest(source="hand", candidates=list(range(len(hand))), count=len(hand),
                                    purpose="exhaust", min_count=0)
    for ref in chosen:
        c.exhaust_from_hand(ref)


@potion("ATTACK_POTION")
def attack_potion(c, u) -> Iterator[SelectionRequest]:
    yield from _choose_generated(c, c.distinct_cards(c.generation_pool("attack"), 3))


@potion("SKILL_POTION")
def skill_potion(c, u) -> Iterator[SelectionRequest]:
    yield from _choose_generated(c, c.distinct_cards(c.generation_pool("skill"), 3))


@potion("POWER_POTION")
def power_potion(c, u) -> Iterator[SelectionRequest]:
    yield from _choose_generated(c, c.distinct_cards(c.generation_pool("power"), 3))


@potion("COLORLESS_POTION")
def colorless_potion(c, u) -> Iterator[SelectionRequest]:
    yield from _choose_generated(c, c.distinct_cards(c.generation_pool(color="colorless"), 3))


@potion("BEETLE_JUICE")
def beetle_juice(c, u):
    if u.target is not None:
        c.apply_power(u.target, "shrink", 4, source=c.player)


@potion("BLESSING_OF_THE_FORGE")
def blessing_of_the_forge(c, u):
    for i, cid in enumerate(list(c.player.hand)):
        if c.is_upgradable(cid):
            c.upgrade_in_hand(i)


@potion("BLOCK_POTION")
def block_potion(c, u):
    c.gain_block(12, powered=False)


@potion("BLOOD_POTION")
def blood_potion(c, u):
    c.heal_player(int(c.player.max_hp * 20 / 100))


@potion("BOTTLED_POTENTIAL")
def bottled_potential(c, u):
    pl = c.player
    pl.draw_pile.extend(pl.hand)
    pl.hand.clear()
    c.shuffle_discard_into_draw()
    c.draw(5)


@potion("CLARITY")
def clarity(c, u):
    c.draw(1)
    c.apply_power(c.player, "clarity", 3)


@potion("CURE_ALL")
def cure_all(c, u):
    c.gain_energy(1)
    c.draw(2)


@potion("DEXTERITY_POTION")
def dexterity_potion(c, u):
    c.apply_power(c.player, "dexterity", 2)


@potion("DISTILLED_CHAOS")
def distilled_chaos(c, u):
    c.autoplay_top_of_draw(3)


@potion("DROPLET_OF_PRECOGNITION")
def droplet_of_precognition(c, u) -> Iterator[SelectionRequest]:
    pile = c.player.draw_pile
    if not pile:
        return
    chosen = yield SelectionRequest(source="draw", candidates=list(range(len(pile))), count=1, purpose="to_hand")
    for ref in chosen:
        remove_ref(pile, ref)
        c.add_to_hand(ref)


@potion("DUPLICATOR")
def duplicator(c, u):
    c.apply_power(c.player, "duplication", 1)


@potion("ENERGY_POTION")
def energy_potion(c, u):
    c.gain_energy(2)


@potion("ENTROPIC_BREW")
def entropic_brew(c, u):
    stream = c.rng.stream("potion_generation")
    while None in c.player.potions:
        if not c.procure_potion(random_potion(c.potion_defs, stream)):
            break


@potion("EXPLOSIVE_AMPOULE")
def explosive_ampoule(c, u):
    c.damage_all_unpowered(10)


@potion("FIRE_POTION")
def fire_potion(c, u):
    if u.target is not None:
        c.damage_monster_unpowered(u.target, 20)


@potion("FLEX_POTION")
def flex_potion(c, u):
    c.gain_temporary_strength(5)


@potion("FORTIFIER")
def fortifier(c, u):
    c.gain_block(c.player.block * 2, powered=False)


@potion("FRUIT_JUICE")
def fruit_juice(c, u):
    c.gain_max_hp(5)


@potion("FYSH_OIL")
def fysh_oil(c, u):
    c.gain_strength(1)
    c.apply_power(c.player, "dexterity", 1)


@potion("GAMBLERS_BREW")
def gamblers_brew(c, u) -> Iterator[SelectionRequest]:
    yield from discard_any_then_draw(c)


def discard_any_then_draw(c: CombatContext) -> Iterator[SelectionRequest]:
    """Gambler's Brew / Gambling Chip: discard any number of cards, then draw that many."""

    hand = c.player.hand
    if not hand:
        return
    chosen = yield SelectionRequest(source="hand", candidates=list(range(len(hand))), count=len(hand),
                                    purpose="discard", min_count=0)
    for ref in chosen:
        c.discard_from_hand(ref)
    c.draw(len(chosen))


@potion("GIGANTIFICATION_POTION")
def gigantification_potion(c, u):
    c.apply_power(c.player, "gigantification", 1)


@potion("HEART_OF_IRON")
def heart_of_iron(c, u):
    c.apply_power(c.player, "plating", 7)


@potion("LIQUID_BRONZE")
def liquid_bronze(c, u):
    c.apply_power(c.player, "thorns", 3)


@potion("LIQUID_MEMORIES")
def liquid_memories(c, u) -> Iterator[SelectionRequest]:
    pile = c.player.discard_pile
    if not pile:
        return
    chosen = yield SelectionRequest(source="discard", candidates=list(range(len(pile))), count=1, purpose="to_hand")
    for ref in chosen:
        remove_ref(pile, ref)
        if not isinstance(ref, CardRef):
            ref = CardRef(ref)
        ref.free_turn = True
        c.add_to_hand(ref)


@potion("LUCKY_TONIC")
def lucky_tonic(c, u):
    c.apply_power(c.player, "buffer", 1)


@potion("MAZALETHS_GIFT")
def mazaleths_gift(c, u):
    c.apply_power(c.player, "ritual", 1)


@potion("OROBIC_ACID")
def orobic_acid(c, u):
    for ctype in ("attack", "skill", "power"):
        picked = c.distinct_cards(c.generation_pool(ctype), 1)
        if picked:
            c.add_to_hand(picked[0]).free_turn = True


@potion("POTION_OF_BINDING")
def potion_of_binding(c, u):
    for m in c.alive_monsters():
        c.apply_power(m, "weak", 1, source=c.player)
        c.apply_power(m, "vulnerable", 1, source=c.player)


@potion("POTION_SHAPED_ROCK")
def potion_shaped_rock(c, u):
    if u.target is not None:
        c.damage_monster_unpowered(u.target, 15)


@potion("POWDERED_DEMISE")
def powdered_demise(c, u):
    if u.target is not None:
        c.apply_power(u.target, "demise", 9, source=c.player)


@potion("RADIANT_TINCTURE")
def radiant_tincture(c, u):
    c.gain_energy(1)
    c.apply_power(c.player, "radiance", 3)


@potion("REGEN_POTION")
def regen_potion(c, u):
    c.apply_power(c.player, "regen", 5)


@potion("SHACKLING_POTION")
def shackling_potion(c, u):
    for m in c.alive_monsters():
        if c.apply_power(m, "strength", -7, source=c.player):
            m.flags["temporary_strength_loss"] = m.flags.get("temporary_strength_loss", 0) + 7


@potion("SHIP_IN_A_BOTTLE")
def ship_in_a_bottle(c, u):
    c.gain_block(10, powered=False)
    c.apply_power(c.player, "block_next_turn", 10)


@potion("SNECKO_OIL")
def snecko_oil(c, u):
    c.draw(7)
    stream = c.rng.stream("energy_cost")
    for i, cid in enumerate(c.player.hand):
        card = c.cards[cid]
        if card.x_cost or card.unplayable:
            continue
        ref = cid if isinstance(cid, CardRef) else CardRef(cid)
        c.player.hand[i] = ref
        ref.cost_override = stream.randrange(4)
        ref.cost_bump = 0


@potion("SOLDIERS_STEW")
def soldiers_stew(c, u):
    pl = c.player
    for pile in (pl.hand, pl.draw_pile, pl.discard_pile, pl.exhaust_pile):
        for i, cid in enumerate(pile):
            if "strike" in c.cards[cid].tags:
                if not isinstance(cid, CardRef):
                    pile[i] = cid = CardRef(cid)
                cid.replay += 1


@potion("SPEED_POTION")
def speed_potion(c, u):
    c.apply_power(c.player, "dexterity", 5)
    c.player.powers["temporary_dexterity"] = c.player.powers.get("temporary_dexterity", 0) + 5


@potion("STABLE_SERUM")
def stable_serum(c, u):
    c.apply_power(c.player, "retain_hand", 2)


@potion("STRENGTH_POTION")
def strength_potion(c, u):
    c.gain_strength(2)


@potion("SWIFT_POTION")
def swift_potion(c, u):
    c.draw(3)


@potion("TOUCH_OF_INSANITY")
def touch_of_insanity(c, u) -> Iterator[SelectionRequest]:
    hand = c.player.hand
    cands = [i for i, cid in enumerate(hand) if not c.cards[cid].x_cost and c.effective_cost(cid) > 0]
    if not cands:
        return
    chosen = yield SelectionRequest(source="hand", candidates=cands, count=1, purpose="free_for_combat")
    for ref in chosen:
        ref.cost_override = 0
        ref.cost_bump = 0


@potion("VULNERABLE_POTION")
def vulnerable_potion(c, u):
    if u.target is not None:
        c.apply_power(u.target, "vulnerable", 3, source=c.player)


@potion("WEAK_POTION")
def weak_potion(c, u):
    if u.target is not None:
        c.apply_power(u.target, "weak", 3, source=c.player)


@potion("FAIRY_IN_A_BOTTLE")
def fairy_in_a_bottle(c, u):
    del c, u  # automatic: combat._prevent_death


__all__ = ["POTION_EFFECTS", "discard_any_then_draw", "random_potion", "reward_potions", "roll_potion_rarity"]
