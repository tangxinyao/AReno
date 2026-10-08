"""Card play effects, one function per card, keyed by the game's card id.

Each function receives the CombatContext `c` and the `Play` `p` (card,
captured target, X for X-cost cards) and reads every number from the card's
`vars` (already upgraded for "+1" printings). Cards that make the player
choose cards are generators that `yield` a SelectionRequest and receive
the chosen card ids back.

Semantics follow the v0.107.1 card text, cross-checked against r33hab/sts2
`CardEffects.cs`. Cards not listed here (Wound, Dazed, curses ...) have no
on-play effect.
"""

from __future__ import annotations

from typing import Callable, Iterator

from .combat import REND_DEBUFFS, CombatContext, Play, SelectionRequest
from .state import CardRef, index_ref, remove_ref


CardFn = Callable[[CombatContext, Play], object]
CARD_EFFECTS: dict[str, CardFn] = {}


def card(game_id: str):
    def deco(fn: CardFn) -> CardFn:
        CARD_EFFECTS[game_id] = fn
        return fn

    return deco


def _v(p: Play, name: str) -> int:
    return p.card.v(name)


def _hit(c: CombatContext, p: Play, base: int, hits: int = 1) -> int:
    if p.target is None:
        return 0
    return c.attack_monster(p.target, base, hits=hits)


def _strike_count(c: CombatContext) -> int:
    pl = c.player
    n = 0
    for pile in (pl.hand, pl.draw_pile, pl.discard_pile, pl.exhaust_pile):
        n += sum(1 for cid in pile if "Strike" in c.cards[cid].name)
    return n


def _choose_hand(c: CombatContext, purpose: str, *, count: int = 1,
                 only: Callable[[str], bool] | None = None) -> SelectionRequest:
    hand = c.player.hand
    cands = [i for i, cid in enumerate(hand) if only is None or only(cid)]
    return SelectionRequest(source="hand", candidates=cands, count=count, purpose=purpose)


def _exhaust_random_from_hand(c: CombatContext, *, card_type: str | None = None) -> str | None:
    hand = c.player.hand
    cands = [cid for cid in hand if card_type is None or c.cards[cid].card_type == card_type]
    if not cands:
        return None
    pick = c.rng.stream("card_select").choice(cands)
    c.exhaust_from_hand(pick)
    return pick


def _bonus(c: CombatContext, p: Play) -> int:
    del c
    return (p.ref.bonus if p.ref is not None else 0) + p.bonus


# ---------------------------------------------------------------------------
# Basic / token

@card("STRIKE_IRONCLAD")
def strike(c, p):
    _hit(c, p, _v(p, "damage"))


@card("DEFEND_IRONCLAD")
def defend(c, p):
    c.gain_block(_v(p, "block"))


@card("BASH")
def bash(c, p):
    _hit(c, p, _v(p, "damage"))
    if p.target is not None:
        c.apply_power(p.target, "vulnerable", _v(p, "vulnerable"), source=c.player)


@card("GIANT_ROCK")
def giant_rock(c, p):
    _hit(c, p, _v(p, "damage"))


@card("SLIMED")
def slimed(c, p):
    c.draw(_v(p, "cards"))


@card("FRANTIC_ESCAPE")
def frantic_escape(c, p):
    # "Increase Sandpit by 1. Increase the cost of this card by 1."
    pit = next((m for m in c.alive_monsters() if m.powers.get("sandpit", 0) > 0), None)
    if pit is not None:
        pit.powers["sandpit"] += 1
    if p.ref is not None:
        p.ref.cost_bump += 1


# ---------------------------------------------------------------------------
# Attacks

@card("ANGER")
def anger(c, p):
    _hit(c, p, _v(p, "damage"))
    c.player.discard_pile.append(CardRef(p.card.card_id))


@card("ASHEN_STRIKE")
def ashen_strike(c, p):
    _hit(c, p, _v(p, "calculation_base") + _v(p, "extra_damage") * len(c.player.exhaust_pile))


@card("BLUDGEON")
def bludgeon(c, p):
    _hit(c, p, _v(p, "damage"))


@card("BODY_SLAM")
def body_slam(c, p):
    _hit(c, p, c.player.block)


@card("BREAK")
def break_(c, p):
    _hit(c, p, _v(p, "damage"))
    if p.target is not None:
        c.apply_power(p.target, "vulnerable", _v(p, "vulnerable"), source=c.player)


@card("BREAKTHROUGH")
def breakthrough(c, p):
    c.lose_hp(_v(p, "hp_loss"), from_card=True)
    if c.combat.outcome is None:
        c.attack_all(_v(p, "damage"))


@card("BULLY")
def bully(c, p):
    vuln = p.target.powers.get("vulnerable", 0) if p.target is not None else 0
    _hit(c, p, _v(p, "calculation_base") + _v(p, "extra_damage") * vuln)


@card("CINDER")
def cinder(c, p):
    _hit(c, p, _v(p, "damage"))
    _exhaust_random_from_hand(c)


@card("CONFLAGRATION")
def conflagration(c, p):
    c.attack_all(_v(p, "damage"), hits=_v(p, "repeat"))


@card("DISMANTLE")
def dismantle(c, p):
    hits = 2 if p.target is not None and p.target.powers.get("vulnerable", 0) > 0 else 1
    _hit(c, p, _v(p, "damage"), hits)


@card("FEED")
def feed(c, p):
    t = p.target
    if t is None:
        return
    fatal_ok = t.powers.get("minion", 0) <= 0
    _hit(c, p, _v(p, "damage"))
    if fatal_ok and t.hp <= 0:
        c.gain_max_hp(_v(p, "max_hp"))


@card("FIEND_FIRE")
def fiend_fire(c, p):
    hand = list(c.player.hand)
    for cid in hand:
        c.exhaust_from_hand(cid)
    _hit(c, p, _v(p, "damage"), len(hand))


@card("FIGHT_ME")
def fight_me(c, p):
    t = p.target
    _hit(c, p, _v(p, "damage"), _v(p, "repeat"))
    c.gain_strength(_v(p, "strength"))
    if t is not None:
        c.apply_power(t, "strength", _v(p, "enemy_strength"), source=c.player)


@card("HEADBUTT")
def headbutt(c, p) -> Iterator[SelectionRequest]:
    _hit(c, p, _v(p, "damage"))
    pile = c.player.discard_pile
    if pile and c.combat.outcome is None:
        chosen = yield SelectionRequest(source="discard", candidates=list(range(len(pile))),
                                        count=1, purpose="to_draw_top")
        for cid in chosen:
            remove_ref(pile, cid)
            c.player.draw_pile.append(cid)


@card("HEMOKINESIS")
def hemokinesis(c, p):
    c.lose_hp(_v(p, "hp_loss"), from_card=True)
    if c.combat.outcome is None:
        _hit(c, p, _v(p, "damage"))


@card("HOWL_FROM_BEYOND")
def howl_from_beyond(c, p):
    c.attack_all(_v(p, "damage"))


@card("IRON_WAVE")
def iron_wave(c, p):
    c.gain_block(_v(p, "block"))
    _hit(c, p, _v(p, "damage"))


@card("MANGLE")
def mangle(c, p):
    t = p.target
    _hit(c, p, _v(p, "damage"))
    if t is not None and t.alive:
        loss = _v(p, "strength_loss")
        if c.apply_power(t, "strength", -loss, source=c.player):
            t.flags["temporary_strength_loss"] = t.flags.get("temporary_strength_loss", 0) + loss


@card("MOLTEN_FIST")
def molten_fist(c, p):
    t = p.target
    _hit(c, p, _v(p, "damage"))
    if t is not None and t.alive:
        vuln = t.powers.get("vulnerable", 0)
        if vuln > 0:
            c.apply_power(t, "vulnerable", vuln, source=c.player)


@card("PACTS_END")
def pacts_end(c, p):
    if len(c.player.exhaust_pile) >= _v(p, "cards"):
        c.attack_all(_v(p, "damage"))


@card("PERFECTED_STRIKE")
def perfected_strike(c, p):
    count = _strike_count(c) + (1 if "Strike" in p.card.name else 0)
    _hit(c, p, _v(p, "calculation_base") + _v(p, "extra_damage") * count)


@card("PILLAGE")
def pillage(c, p):
    _hit(c, p, _v(p, "damage"))
    while c.combat.outcome is None:
        drawn = c.draw(1)
        if not drawn or c.cards[drawn[0]].card_type != "attack":
            break


@card("POMMEL_STRIKE")
def pommel_strike(c, p):
    _hit(c, p, _v(p, "damage"))
    c.draw(_v(p, "cards"))


@card("RAMPAGE")
def rampage(c, p):
    _hit(c, p, _v(p, "damage") + _bonus(c, p))
    p.bonus += _v(p, "increase")


@card("SETUP_STRIKE")
def setup_strike(c, p):
    _hit(c, p, _v(p, "damage"))
    amount = _v(p, "strength")
    c.gain_strength(amount)
    c.player.powers["temporary_strength"] = c.player.powers.get("temporary_strength", 0) + amount


@card("SPITE")
def spite(c, p):
    hits = _v(p, "repeat") if c.combat.hp_lost_this_turn > 0 else 1
    _hit(c, p, _v(p, "damage"), hits)


@card("STOMP")
def stomp(c, p):
    c.attack_all(_v(p, "damage"))


@card("SWORD_BOOMERANG")
def sword_boomerang(c, p):
    c.attack_random(_v(p, "damage"), hits=_v(p, "repeat"))


@card("TEAR_ASUNDER")
def tear_asunder(c, p):
    _hit(c, p, _v(p, "damage"), _v(p, "repeat") + c.combat.hp_loss_events_this_combat)


@card("THRASH")
def thrash(c, p):
    _hit(c, p, _v(p, "damage") + _bonus(c, p), 2)
    eaten = _exhaust_random_from_hand(c, card_type="attack")
    if eaten is not None:
        ec = c.cards[eaten]
        if ec.game_id == "BODY_SLAM":
            gain = c.player.block
        elif "damage" in ec.vars:
            gain = ec.v("damage")
        else:
            gain = ec.vars.get("calculation_base", 0)
        p.bonus += gain + getattr(eaten, "bonus", 0)


@card("THUNDERCLAP")
def thunderclap(c, p):
    c.attack_all(_v(p, "damage"))
    for m in c.alive_monsters():
        c.apply_power(m, "vulnerable", _v(p, "vulnerable"), source=c.player)


@card("TWIN_STRIKE")
def twin_strike(c, p):
    _hit(c, p, _v(p, "damage"), 2)


@card("UNRELENTING")
def unrelenting(c, p):
    _hit(c, p, _v(p, "damage"))
    c.apply_power(c.player, "free_attack", 1)


@card("UPPERCUT")
def uppercut(c, p):
    t = p.target
    _hit(c, p, _v(p, "damage"))
    if t is not None:
        c.apply_power(t, "weak", _v(p, "power"), source=c.player)
        c.apply_power(t, "vulnerable", _v(p, "power"), source=c.player)


@card("WHIRLWIND")
def whirlwind(c, p):
    c.attack_all(_v(p, "damage"), hits=p.x)


# ---------------------------------------------------------------------------
# Skills

@card("ARMAMENTS")
def armaments(c, p) -> Iterator[SelectionRequest]:
    c.gain_block(_v(p, "block"))
    if p.card.upgraded:
        for i, cid in enumerate(list(c.player.hand)):
            if c.is_upgradable(cid):
                c.upgrade_in_hand(i)
        return
    req = _choose_hand(c, "upgrade", only=c.is_upgradable)
    if req.candidates:
        chosen = yield req
        for cid in chosen:
            c.upgrade_in_hand(index_ref(c.player.hand, cid))


@card("BATTLE_TRANCE")
def battle_trance(c, p):
    c.draw(_v(p, "cards"))
    c.apply_power(c.player, "no_draw", 1)


@card("BLOOD_WALL")
def blood_wall(c, p):
    c.lose_hp(_v(p, "hp_loss"), from_card=True)
    if c.combat.outcome is None:
        c.gain_block(_v(p, "block"))


@card("BLOODLETTING")
def bloodletting(c, p):
    c.lose_hp(_v(p, "hp_loss"), from_card=True)
    if c.combat.outcome is None:
        c.gain_energy(_v(p, "energy"))


@card("BRAND")
def brand(c, p) -> Iterator[SelectionRequest]:
    c.lose_hp(_v(p, "hp_loss"), from_card=True)
    if c.combat.outcome is not None:
        return
    if c.player.hand:
        chosen = yield _choose_hand(c, "exhaust")
        for cid in chosen:
            c.exhaust_from_hand(cid)
    c.gain_strength(_v(p, "strength"))


@card("BURNING_PACT")
def burning_pact(c, p) -> Iterator[SelectionRequest]:
    if c.player.hand:
        chosen = yield _choose_hand(c, "exhaust")
        for cid in chosen:
            c.exhaust_from_hand(cid)
    c.draw(_v(p, "cards"))


@card("CASCADE")
def cascade(c, p):
    count = p.x + (1 if p.card.upgraded else 0)
    pl = c.player
    taken: list[str] = []
    for _ in range(count):
        if not pl.draw_pile:
            if not pl.discard_pile:
                break
            pl.draw_pile = list(pl.discard_pile)
            pl.discard_pile.clear()
            c.rng.stream("combat_shuffle").shuffle(pl.draw_pile)
        taken.append(pl.draw_pile.pop())
    for cid in taken:
        if c.combat.outcome is not None:
            pl.discard_pile.append(cid)
            continue
        c.autoplay(cid)


@card("COLOSSUS")
def colossus(c, p):
    c.gain_block(_v(p, "block"))
    c.apply_power(c.player, "colossus", _v(p, "colossus"))


@card("DOMINATE")
def dominate(c, p):
    t = p.target
    if t is None:
        return
    c.apply_power(t, "vulnerable", _v(p, "vulnerable"), source=c.player)
    gain = _v(p, "strength_per_vulnerable") * t.powers.get("vulnerable", 0)
    if gain:
        c.gain_strength(gain)


@card("DRUM_OF_BATTLE")
def drum_of_battle(c, p):
    c.draw(_v(p, "cards"))


@card("EVIL_EYE")
def evil_eye(c, p):
    c.gain_block(_v(p, "block"))
    if c.combat.cards_exhausted_this_turn > 0:
        c.gain_block(_v(p, "block"))


@card("EXPECT_A_FIGHT")
def expect_a_fight(c, p):
    attacks = sum(1 for cid in c.player.hand if c.cards[cid].card_type == "attack")
    c.gain_energy(attacks)
    c.apply_power(c.player, "no_energy_gain", 1)


@card("FLAME_BARRIER")
def flame_barrier(c, p):
    c.gain_block(_v(p, "block"))
    c.apply_power(c.player, "flame_barrier", _v(p, "damage_back"))


@card("FORGOTTEN_RITUAL")
def forgotten_ritual(c, p):
    if c.combat.cards_exhausted_this_turn > 0:
        c.gain_energy(_v(p, "energy"))


@card("HAVOC")
def havoc(c, p):
    pl = c.player
    if pl.draw_pile:
        c.autoplay(pl.draw_pile.pop(), exhaust=True)


@card("IMPERVIOUS")
def impervious(c, p):
    c.gain_block(_v(p, "block"))


@card("INFERNAL_BLADE")
def infernal_blade(c, p):
    if len(c.player.hand) >= 10:
        return
    pool = c.generation_pool("attack")
    pick = c.rng.stream("card_generation").choice(pool)
    c.add_to_hand(pick).free_turn = True


@card("NOT_YET")
def not_yet(c, p):
    c.heal_player(_v(p, "heal"))


@card("OFFERING")
def offering(c, p):
    c.lose_hp(_v(p, "hp_loss"), from_card=True)
    if c.combat.outcome is None:
        c.gain_energy(_v(p, "energy"))
        c.draw(_v(p, "cards"))


@card("ONE_TWO_PUNCH")
def one_two_punch(c, p):
    c.apply_power(c.player, "one_two_punch", _v(p, "attacks"))


@card("PRIMAL_FORCE")
def primal_force(c, p):
    rock = "giant_rock+1" if p.card.upgraded else "giant_rock"
    hand = c.player.hand
    for i, cid in enumerate(hand):
        if c.cards[cid].card_type == "attack":
            hand[i] = CardRef(rock)


@card("RAGE")
def rage(c, p):
    c.apply_power(c.player, "rage", _v(p, "power"))


@card("SECOND_WIND")
def second_wind(c, p):
    for cid in [x for x in c.player.hand if c.cards[x].card_type != "attack"]:
        c.exhaust_from_hand(cid)
        c.gain_block(_v(p, "block"))


@card("SHRUG_IT_OFF")
def shrug_it_off(c, p):
    c.gain_block(_v(p, "block"))
    c.draw(_v(p, "cards"))


@card("STOKE")
def stoke(c, p):
    hand = list(c.player.hand)
    for cid in hand:
        c.exhaust_from_hand(cid)
    pool = c.generation_pool()
    gen = c.rng.stream("card_generation")
    for _ in hand:
        pick = gen.choice(pool)
        if p.card.upgraded:
            pick = c.cards[pick].upgrade_of or pick
        c.add_to_hand(pick)


@card("TAUNT")
def taunt(c, p):
    c.gain_block(_v(p, "block"))
    if p.target is not None:
        c.apply_power(p.target, "vulnerable", _v(p, "vulnerable"), source=c.player)


@card("TREMBLE")
def tremble(c, p):
    if p.target is not None:
        c.apply_power(p.target, "vulnerable", _v(p, "vulnerable"), source=c.player)


@card("TRUE_GRIT")
def true_grit(c, p) -> Iterator[SelectionRequest]:
    c.gain_block(_v(p, "block"))
    if not c.player.hand:
        return
    if p.card.upgraded:
        chosen = yield _choose_hand(c, "exhaust")
        for cid in chosen:
            c.exhaust_from_hand(cid)
    else:
        _exhaust_random_from_hand(c)


# ---------------------------------------------------------------------------
# Powers

def _power(pid: str, var: str | None = None, amount: int = 1) -> CardFn:
    def fn(c: CombatContext, p: Play) -> None:
        c.apply_power(c.player, pid, _v(p, var) if var else amount)

    return fn


CARD_EFFECTS.update({
    "AGGRESSION": _power("aggression"),
    "BARRICADE": _power("barricade"),
    "CORRUPTION": _power("corruption"),
    "CRUELTY": _power("cruelty", "cruelty"),
    "DARK_EMBRACE": _power("dark_embrace"),
    "DEMON_FORM": _power("demon_form", "strength"),
    "FEEL_NO_PAIN": _power("feel_no_pain", "power"),
    "HELLRAISER": _power("hellraiser"),
    "INFERNO": _power("inferno", "inferno"),
    "JUGGERNAUT": _power("juggernaut", "juggernaut"),
    "JUGGLING": _power("juggling"),
    "PYRE": _power("pyre", "energy"),
    "RUPTURE": _power("rupture", "strength"),
    "STAMPEDE": _power("stampede", "power"),
    "STONE_ARMOR": _power("plating", "plating"),
    "UNMOVABLE": _power("unmovable"),
    "VICIOUS": _power("vicious", "cards"),
})


@card("INFLAME")
def inflame(c, p):
    c.gain_strength(_v(p, "strength"))


@card("CRIMSON_MANTLE")
def crimson_mantle(c, p):
    c.apply_power(c.player, "crimson_mantle", _v(p, "crimson_mantle"))
    # Each copy played adds 1 to the HP lost at the start of the turn.
    c.player.powers["crimson_mantle_hp"] = c.player.powers.get("crimson_mantle_hp", 0) + 1



# ---------------------------------------------------------------------------
# Colorless (r33hab CardEffects `case CL.*`)

@card("ALCHEMIZE")
def alchemize(c, p):
    from .potion_effects import random_potion

    c.procure_potion(random_potion(c.potion_defs, c.rng.stream("potion_generation")))


@card("ANOINTED")
def anointed(c, p):
    pl = c.player
    rares = [ref for ref in pl.draw_pile if c.cards[ref].rarity == "rare"]
    c.rng.stream("card_select").shuffle(rares)
    room = 10 - len(pl.hand)
    for ref in rares[:max(0, room)]:
        remove_ref(pl.draw_pile, ref)
        if not isinstance(ref, CardRef):
            ref = CardRef(ref)
        # Anointed+ only gains Retain itself (its upgrade); the drawn Rares do not.
        pl.hand.append(ref)


@card("BEAT_DOWN")
def beat_down(c, p):
    pl = c.player
    attacks = sorted((ref for ref in pl.discard_pile
                      if c.cards[ref].card_type == "attack" and not c.cards[ref].unplayable), key=str)
    c.rng.stream("combat_shuffle").shuffle(attacks)
    taken = attacks[:_v(p, "cards")]
    for ref in taken:
        remove_ref(pl.discard_pile, ref)
    for ref in taken:
        if c.combat.outcome is not None:
            pl.discard_pile.append(ref)
            continue
        c.autoplay(ref)


@card("BOLAS")
def bolas(c, p):
    _hit(c, p, _v(p, "damage"))


@card("CATASTROPHE")
def catastrophe(c, p):
    pl = c.player
    stream = c.rng.stream("combat_shuffle")
    for _ in range(_v(p, "cards")):
        if not pl.draw_pile or c.combat.outcome is not None:
            break
        playable = [ref for ref in pl.draw_pile if not c.cards[ref].unplayable] or list(pl.draw_pile)
        playable.sort(key=str)
        stream.shuffle(playable)
        ref = playable[0]
        remove_ref(pl.draw_pile, ref)
        c.autoplay(ref)


@card("DARK_SHACKLES")
def dark_shackles(c, p):
    t = p.target
    if t is not None and t.alive:
        loss = _v(p, "strength_loss")
        if c.apply_power(t, "strength", -loss, source=c.player):
            t.flags["temporary_strength_loss"] = t.flags.get("temporary_strength_loss", 0) + loss


@card("DISCOVERY")
def discovery(c, p) -> Iterator[SelectionRequest]:
    options = c.distinct_cards(c.generation_pool(color="colorless"), 3)
    if not options:
        return
    chosen = yield SelectionRequest(source="generated", candidates=list(range(len(options))), count=1,
                                    purpose="to_hand", options=options)
    for cid in chosen:
        c.add_to_hand(cid).free_turn = True


@card("DRAMATIC_ENTRANCE")
def dramatic_entrance(c, p):
    c.attack_all(_v(p, "damage"))


@card("EQUILIBRIUM")
def equilibrium(c, p):
    c.gain_block(_v(p, "block"))
    c.apply_power(c.player, "retain_hand", 1)


@card("FINESSE")
def finesse(c, p):
    c.gain_block(_v(p, "block"))
    c.draw(_v(p, "cards"))


@card("FISTICUFFS")
def fisticuffs(c, p):
    dealt = _hit(c, p, _v(p, "damage"))
    if dealt > 0:
        c.gain_block(dealt)


@card("FLASH_OF_STEEL")
def flash_of_steel(c, p):
    _hit(c, p, _v(p, "damage"))
    c.draw(_v(p, "cards"))


@card("GOLD_AXE")
def gold_axe(c, p):
    _hit(c, p, c.combat.cards_played_this_combat)


@card("HAND_OF_GREED")
def hand_of_greed(c, p):
    t = p.target
    if t is None:
        return
    fatal_ok = t.powers.get("minion", 0) <= 0 and t.powers.get("reattach", 0) <= 0
    _hit(c, p, _v(p, "damage"))
    if fatal_ok and t.hp <= 0:
        c.gain_gold(_v(p, "gold"))


@card("HIDDEN_GEM")
def hidden_gem(c, p):
    pile = c.player.draw_pile
    ok = [i for i, ref in enumerate(pile) if not c.cards[ref].unplayable
          and c.cards[ref].card_type not in ("status", "curse") and getattr(ref, "replay", 0) < 1]
    preferred = [i for i in ok if c.cards[pile[i]].card_type in ("attack", "skill", "power")] or ok
    if preferred:
        i = c.rng.stream("card_select").choice(preferred)
        if not isinstance(pile[i], CardRef):
            pile[i] = CardRef(pile[i])
        pile[i].replay += _v(p, "replay")


@card("IMPATIENCE")
def impatience(c, p):
    if not any(c.cards[ref].card_type == "attack" for ref in c.player.hand):
        c.draw(_v(p, "cards"))


@card("JACK_OF_ALL_TRADES")
def jack_of_all_trades(c, p):
    pool = c.generation_pool(color="colorless")
    stream = c.rng.stream("card_generation")
    for _ in range(_v(p, "cards")):
        if len(c.player.hand) >= 10:
            break
        c.add_to_hand(stream.choice(pool))


@card("JACKPOT")
def jackpot(c, p):
    _hit(c, p, _v(p, "damage"))
    pool = [cid for cid in c.generation_pool() if c.cards[cid].cost == 0 and not c.cards[cid].x_cost]
    stream = c.rng.stream("card_generation")
    for _ in range(_v(p, "cards")):
        if not pool or len(c.player.hand) >= 10:
            break
        cid = stream.choice(pool)
        if p.card.upgraded:
            cid = c.cards[cid].upgrade_of or cid
        c.add_to_hand(cid)


@card("MASTER_OF_STRATEGY")
def master_of_strategy(c, p):
    c.draw(_v(p, "cards"))


@card("MIND_BLAST")
def mind_blast(c, p):
    _hit(c, p, len(c.player.draw_pile))


@card("OMNISLICE")
def omnislice(c, p):
    t = p.target
    if t is None:
        return
    dealt = _hit(c, p, _v(p, "damage"))
    if dealt <= 0:
        return
    for m in c.alive_monsters():
        if m is not t:
            c.damage_monster_unpowered(m, dealt, thorns=True)


@card("PANIC_BUTTON")
def panic_button(c, p):
    c.gain_block(_v(p, "block"))
    c.apply_power(c.player, "no_block", _v(p, "turns"))


@card("PRODUCTION")
def production(c, p):
    c.gain_energy(_v(p, "energy"))


@card("PROLONG")
def prolong(c, p):
    if c.player.block > 0:
        c.apply_power(c.player, "block_next_turn", c.player.block)


@card("PROWESS")
def prowess(c, p):
    c.gain_strength(_v(p, "strength"))
    c.apply_power(c.player, "dexterity", _v(p, "dexterity"))


@card("PURITY")
def purity(c, p) -> Iterator[SelectionRequest]:
    hand = c.player.hand
    if not hand:
        return
    chosen = yield SelectionRequest(source="hand", candidates=list(range(len(hand))), count=_v(p, "cards"),
                                    purpose="exhaust", min_count=0)
    for ref in chosen:
        c.exhaust_from_hand(ref)


@card("REND")
def rend(c, p):
    t = p.target
    debuffs = sum(1 for d in REND_DEBUFFS if t is not None and t.powers.get(d, 0) > 0)
    _hit(c, p, _v(p, "calculation_base") + _v(p, "extra_damage") * debuffs)


@card("RESTLESSNESS")
def restlessness(c, p):
    if not c.player.hand:
        c.draw(_v(p, "cards"))
        c.gain_energy(_v(p, "energy"))


@card("SALVO")
def salvo(c, p):
    _hit(c, p, _v(p, "damage"))
    c.apply_power(c.player, "retain_hand", 1)


@card("SCRAWL")
def scrawl(c, p):
    c.draw(10 - len(c.player.hand))


def _from_draw_pile(c, card_type: str | None):
    pile = c.player.draw_pile
    cands = [i for i, ref in enumerate(pile) if card_type is None or c.cards[ref].card_type == card_type]
    if not cands:
        return
    chosen = yield SelectionRequest(source="draw", candidates=cands, count=1, purpose="to_hand")
    for ref in chosen:
        remove_ref(pile, ref)
        c.add_to_hand(ref)


@card("SECRET_TECHNIQUE")
def secret_technique(c, p) -> Iterator[SelectionRequest]:
    yield from _from_draw_pile(c, "skill")


@card("SECRET_WEAPON")
def secret_weapon(c, p) -> Iterator[SelectionRequest]:
    yield from _from_draw_pile(c, "attack")


@card("SEEKER_STRIKE")
def seeker_strike(c, p) -> Iterator[SelectionRequest]:
    _hit(c, p, _v(p, "damage"))
    pile = c.player.draw_pile
    if not pile or c.combat.outcome is not None:
        return
    order = sorted(range(len(pile)), key=lambda i: str(pile[i]))
    c.rng.stream("card_select").shuffle(order)
    cands = sorted(order[:_v(p, "cards")])
    chosen = yield SelectionRequest(source="draw", candidates=cands, count=1, purpose="to_hand")
    for ref in chosen:
        remove_ref(pile, ref)
        c.add_to_hand(ref)


@card("SHOCKWAVE")
def shockwave(c, p):
    for m in c.alive_monsters():
        c.apply_power(m, "weak", _v(p, "power"), source=c.player)
        c.apply_power(m, "vulnerable", _v(p, "power"), source=c.player)


@card("SPLASH")
def splash(c, p):
    # Offers Attacks from the other characters' pools, which the sim does not model;
    # Splash is kept out of every reward / shop / generation pool (rewards.UNSUPPORTED_CARDS).
    del c, p


@card("THE_BOMB")
def the_bomb(c, p):
    c.apply_power(c.player, "the_bomb", _v(p, "turns"))
    c._s.extra["the_bomb_damage"] = _v(p, "bomb_damage")


@card("THE_GAMBIT")
def the_gambit(c, p):
    c.gain_block(_v(p, "block"))
    c.apply_power(c.player, "the_gambit", 1)


@card("THINKING_AHEAD")
def thinking_ahead(c, p) -> Iterator[SelectionRequest]:
    c.draw(_v(p, "cards"))
    hand = c.player.hand
    if not hand:
        return
    chosen = yield SelectionRequest(source="hand", candidates=list(range(len(hand))), count=1, purpose="to_draw_top")
    for ref in chosen:
        remove_ref(hand, ref)
        c.player.draw_pile.append(ref)


@card("THRUMMING_HATCHET")
def thrumming_hatchet(c, p):
    _hit(c, p, _v(p, "damage"))


@card("ULTIMATE_DEFEND")
def ultimate_defend(c, p):
    c.gain_block(_v(p, "block"))


@card("ULTIMATE_STRIKE")
def ultimate_strike(c, p):
    _hit(c, p, _v(p, "damage"))


@card("VOLLEY")
def volley(c, p):
    c.attack_random(_v(p, "damage"), hits=p.x)


CARD_EFFECTS.update({
    "AUTOMATION": _power("automation", "energy"),
    "CALAMITY": _power("calamity"),
    "ENTROPY": _power("entropy", "cards"),
    "ETERNAL_ARMOR": _power("plating", "plating"),
    "FASTEN": _power("fasten", "extra_block"),
    "MAYHEM": _power("mayhem"),
    "NOSTALGIA": _power("nostalgia"),
    "PANACHE": _power("panache", "panache_damage"),
    "PREP_TIME": _power("prep_time", "prep_time"),
    "ROLLING_BOULDER": _power("rolling_boulder", "rolling_boulder"),
    "STRATAGEM": _power("stratagem"),
})


__all__ = ["CARD_EFFECTS"]
