"""Normal (non-ancient) events.

Each handler is `fn(loop) -> list[EventOption]` returning the choices shown
at the EVENT screen. The runtime (`run.py`) rolls one event per EVENT room
from the act's pool, caches its labels in `state.event_option_labels`, and
calls `apply(loop)` on the picked option.

An event with no handler falls back to a "move on" option so the room is
always traversable. Handlers mutate state through `loop` only; they must
not open screens that persist after the event closes (ok to push a card
reward / deck select -- the runtime will carry it over to the map).

Pools are rough mappings of codex `act` fields to the three Ironclad acts
plus the Act 1 Underdocks variant. "Shared" events (codex act = None) are
reachable from any act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .state import CardRef, RewardItem

if TYPE_CHECKING:  # pragma: no cover - type hint only
    from .run import RunLoop


@dataclass
class EventOption:
    id: str
    label: str
    apply: Callable[["RunLoop"], None]
    locked: bool = False  # shown but not selectable


Handler = Callable[["RunLoop"], list[EventOption]]
EVENT_HANDLERS: dict[str, Handler] = {}


def event(event_id: str) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        EVENT_HANDLERS[event_id] = fn
        return fn
    return deco


# Pools by act. The runtime passes an act name ("overgrowth", "underdocks",
# "hive", "glory"); events listed under "shared" are legal in any act.
EVENT_POOLS: dict[str, tuple[str, ...]] = {
    "overgrowth": (
        "AROMA_OF_CHAOS", "BYRDONIS_NEST", "DENSE_VEGETATION", "JUNGLE_MAZE_ADVENTURE",
        "LUMINOUS_CHOIR", "MORPHIC_GROVE", "SAPPHIRE_SEED", "TABLET_OF_TRUTH",
        "UNREST_SITE", "WELLSPRING", "WHISPERING_HOLLOW", "WOOD_CARVINGS",
    ),
    "underdocks": (
        "ABYSSAL_BATHS", "DOORS_OF_LIGHT_AND_DARK", "DROWNING_BEACON", "ENDLESS_CONVEYOR",
        "PUNCH_OFF", "SPIRALING_WHIRLPOOL", "SUNKEN_STATUE", "SUNKEN_TREASURY",
        "TRASH_HEAP", "WATERLOGGED_SCRIPTORIUM",
    ),
    "hive": (
        "AMALGAMATOR", "BUGSLAYER", "COLORFUL_PHILOSOPHERS", "COLOSSAL_FLOWER",
        "FIELD_OF_MAN_SIZED_HOLES", "INFESTED_AUTOMATON", "LOST_WISP", "SPIRIT_GRAFTER",
        "THE_LANTERN_KEY", "ZEN_WEAVER",
    ),
    "glory": (
        "BATTLEWORN_DUMMY", "GRAVE_OF_THE_FORGOTTEN", "HUNGRY_FOR_MUSHROOMS",
        "REFLECTIONS", "ROUND_TEA_PARTY", "TINKER_TIME", "TRIAL",
    ),
    "shared": (
        "BRAIN_LEECH", "DOLL_ROOM", "FAKE_MERCHANT", "POTION_COURIER", "RANWID_THE_ELDER",
        "RELIC_TRADER", "ROOM_FULL_OF_CHEESE", "SELF_HELP_BOOK", "SLIPPERY_BRIDGE",
        "STONE_OF_ALL_TIME", "SYMBIOTE", "TEA_MASTER", "THE_ARCHITECT",
        "THE_FUTURE_OF_POTIONS", "THE_LEGENDS_WERE_TRUE", "THIS_OR_THAT",
        "WAR_HISTORIAN_REPY", "WELCOME_TO_WONGOS", "CRYSTAL_SPHERE",
    ),
}


def act_pool(act: str) -> list[str]:
    return list(EVENT_POOLS.get(act, ())) + list(EVENT_POOLS.get("shared", ()))


def _default_options(event_id: str) -> list[EventOption]:
    return [EventOption(id="leave", label=f"leave ({event_id})", apply=lambda _l: None)]


def options_for(loop: "RunLoop", event_id: str) -> list[EventOption]:
    fn = EVENT_HANDLERS.get(event_id)
    return fn(loop) if fn is not None else _default_options(event_id)


# ---------------------------------------------------------------------------
# Helpers used across events

def _p(loop):
    return loop.state.player


def _heal(amount: int):
    def apply(loop):
        p = _p(loop)
        p.hp = min(p.max_hp, p.hp + amount)
    return apply


def _damage(amount: int):
    def apply(loop):
        p = _p(loop)
        p.hp = max(1, p.hp - amount)  # events can't kill
    return apply


def _max_hp(delta: int):
    def apply(loop):
        p = _p(loop)
        if delta >= 0:
            p.max_hp += delta
            p.hp += delta
        else:
            p.max_hp = max(1, p.max_hp + delta)
            p.hp = min(p.hp, p.max_hp)
    return apply


def _gold(delta: int):
    def apply(loop):
        p = _p(loop)
        if delta >= 0:
            p.gold += delta
        else:
            p.gold = max(0, p.gold + delta)
    return apply


def _add_curse(curse_id: str):
    def apply(loop):
        if curse_id in loop.cards:
            _p(loop).deck.append(CardRef(curse_id))
    return apply


def _remove_random_card():
    def apply(loop):
        p = _p(loop)
        unremovable = {"ascenders_bane", "curse_of_the_bell"}
        cands = [i for i, c in enumerate(p.deck) if c not in unremovable]
        if cands:
            stream = loop.rng.stream("relic_pickup")
            p.deck.pop(stream.choice(cands))
    return apply


def _upgrade_random_card():
    def apply(loop):
        p = _p(loop)
        cards = loop.cards
        cands = [i for i, c in enumerate(p.deck) if cards[c].upgrade_of is not None]
        if cands:
            stream = loop.rng.stream("relic_pickup")
            idx = stream.choice(cands)
            cur = p.deck[idx]
            p.deck[idx] = CardRef(cards[cur].upgrade_of, like=cur)
    return apply


def _add_random_potion():
    def apply(loop):
        p = _p(loop)
        if None not in p.potions:
            return
        from .potion_effects import random_potion
        stream = loop.rng.stream("relic_pickup")
        p.potions[p.potions.index(None)] = random_potion(loop.potion_defs, stream)
    return apply


# ---------------------------------------------------------------------------
# Act 1 - Overgrowth

@event("WELLSPRING")
def _wellspring(loop) -> list[EventOption]:
    return [
        EventOption(id="drink", label="Drink: heal 20 HP.", apply=_heal(20)),
        EventOption(id="leave", label="Leave the wellspring untouched.", apply=lambda _l: None),
    ]


@event("MORPHIC_GROVE")
def _morphic_grove(loop) -> list[EventOption]:
    # "Trade a card for a random card of the same type" vs "leave".
    def swap(loop):
        p = _p(loop)
        unremovable = {"ascenders_bane", "curse_of_the_bell"}
        cands = [i for i, c in enumerate(p.deck) if c not in unremovable]
        if not cands:
            return
        stream = loop.rng.stream("relic_pickup")
        idx = stream.choice(cands)
        src = loop.cards[p.deck[idx]]
        pool = [cid for cid in loop._reward_pool
                if loop.cards[cid].card_type == src.card_type and cid != p.deck[idx]]
        if pool:
            p.deck[idx] = CardRef(stream.choice(pool))

    return [
        EventOption(id="offer", label="Transform a random card into another of its type.", apply=swap),
        EventOption(id="leave", label="Walk past the grove.", apply=lambda _l: None),
    ]


@event("WHISPERING_HOLLOW")
def _whispering_hollow(loop) -> list[EventOption]:
    # Lose 1 Max HP; upgrade a random card. Or walk away.
    def apply_upgrade(loop):
        _max_hp(-1)(loop)
        _upgrade_random_card()(loop)
    return [
        EventOption(id="listen", label="Listen: -1 Max HP, upgrade a random card.",
                    apply=apply_upgrade),
        EventOption(id="leave", label="Leave the hollow in silence.", apply=lambda _l: None),
    ]


@event("UNREST_SITE")
def _unrest_site(loop) -> list[EventOption]:
    # Rest 25 HP vs 50 gold.
    return [
        EventOption(id="rest", label="Rest here: heal 25 HP.", apply=_heal(25)),
        EventOption(id="loot", label="Loot the camp: gain 50 Gold.", apply=_gold(50)),
    ]


@event("AROMA_OF_CHAOS")
def _aroma_of_chaos(loop) -> list[EventOption]:
    # Shuffle a random Curse into the deck for a Rare card.
    def inhale(loop):
        _add_curse("decay")(loop)
        stream = loop.rng.stream("relic_pickup")
        rares = [cid for cid in loop._reward_pool if loop.cards[cid].rarity == "rare"]
        if rares:
            _p(loop).deck.append(CardRef(stream.choice(rares)))
    return [
        EventOption(id="inhale", label="Inhale: gain a random Rare card, gain a Curse.", apply=inhale),
        EventOption(id="leave", label="Hold your breath and leave.", apply=lambda _l: None),
    ]


# ---------------------------------------------------------------------------
# Act 1 - Underdocks

@event("TRASH_HEAP")
def _trash_heap(loop) -> list[EventOption]:
    # Dig: gain 30 gold and a Curse. Leave.
    def dig(loop):
        _gold(30)(loop)
        _add_curse("regret")(loop)
    return [
        EventOption(id="dig", label="Dig: gain 30 Gold, gain a Curse.", apply=dig),
        EventOption(id="leave", label="Walk past the heap.", apply=lambda _l: None),
    ]


@event("SUNKEN_TREASURY")
def _sunken_treasury(loop) -> list[EventOption]:
    # Pay 75 gold for a Rare card, or leave.
    def pay(loop):
        p = _p(loop)
        if p.gold < 75:
            return
        _gold(-75)(loop)
        stream = loop.rng.stream("relic_pickup")
        rares = [cid for cid in loop._reward_pool if loop.cards[cid].rarity == "rare"]
        if rares:
            p.deck.append(CardRef(stream.choice(rares)))
    locked = _p(loop).gold < 75
    return [
        EventOption(id="pay", label="Pay 75 Gold for a random Rare card.", apply=pay, locked=locked),
        EventOption(id="leave", label="Leave the treasury.", apply=lambda _l: None),
    ]


@event("ABYSSAL_BATHS")
def _abyssal_baths(loop) -> list[EventOption]:
    # Immerse: +2 Max HP, take 3 damage. Abstain: heal 10 HP.
    def immerse(loop):
        _max_hp(2)(loop)
        _damage(3)(loop)
    return [
        EventOption(id="immerse", label="Immerse: +2 Max HP, take 3 damage.", apply=immerse),
        EventOption(id="abstain", label="Abstain: heal 10 HP.", apply=_heal(10)),
    ]


# ---------------------------------------------------------------------------
# Act 2 - Hive

@event("LOST_WISP")
def _lost_wisp(loop) -> list[EventOption]:
    # Chase: heal 15 HP. Ignore: nothing.
    return [
        EventOption(id="chase", label="Chase the wisp: heal 15 HP.", apply=_heal(15)),
        EventOption(id="ignore", label="Ignore the wisp.", apply=lambda _l: None),
    ]


@event("ZEN_WEAVER")
def _zen_weaver(loop) -> list[EventOption]:
    # Three choices: upgrade a card / remove a card / get a potion.
    def remove(loop):
        _remove_random_card()(loop)
    return [
        EventOption(id="upgrade", label="Weave strength: upgrade a random card.",
                    apply=_upgrade_random_card()),
        EventOption(id="remove", label="Weave release: remove a random card.", apply=remove),
        EventOption(id="brew", label="Weave brew: gain a random potion.", apply=_add_random_potion()),
    ]


# ---------------------------------------------------------------------------
# Act 3 - Glory

@event("HUNGRY_FOR_MUSHROOMS")
def _hungry_for_mushrooms(loop) -> list[EventOption]:
    # Eat: heal 15 HP, 50% chance gain Decay.
    def eat(loop):
        _heal(15)(loop)
        if loop.rng.stream("relic_pickup").random() < 0.5:
            _add_curse("decay")(loop)
    return [
        EventOption(id="eat", label="Eat the mushrooms: heal 15 HP (risk Curse).", apply=eat),
        EventOption(id="leave", label="Leave the mushrooms alone.", apply=lambda _l: None),
    ]


@event("TRIAL")
def _trial(loop) -> list[EventOption]:
    # Accept: lose 15% max HP, upgrade a random card. Decline: nothing.
    def accept(loop):
        p = _p(loop)
        loss = max(1, p.max_hp * 15 // 100)
        p.hp = max(1, p.hp - loss)
        _upgrade_random_card()(loop)
    return [
        EventOption(id="accept", label="Accept the Trial: lose 15% HP, upgrade a random card.",
                    apply=accept),
        EventOption(id="decline", label="Decline.", apply=lambda _l: None),
    ]


# ---------------------------------------------------------------------------
# Shared

@event("POTION_COURIER")
def _potion_courier(loop) -> list[EventOption]:
    # Buy a random potion for 50 gold, or leave.
    def buy(loop):
        p = _p(loop)
        if p.gold < 50:
            return
        _gold(-50)(loop)
        _add_random_potion()(loop)
    locked = _p(loop).gold < 50
    return [
        EventOption(id="buy", label="Buy a random potion for 50 Gold.", apply=buy, locked=locked),
        EventOption(id="leave", label="Thank the courier and leave.", apply=lambda _l: None),
    ]


@event("RELIC_TRADER")
def _relic_trader(loop) -> list[EventOption]:
    # Trade 100 gold for a shared-pool relic.
    def trade(loop):
        p = _p(loop)
        if p.gold < 100:
            return
        _gold(-100)(loop)
        rarity = loop._roll_relic_rarity(loop.rng.stream("relic_pickup"))
        relic = loop._pull_relic(loop.state.shared_relic_bag, rarity)
        if relic:
            loop._obtain_relic(relic, return_to=loop.state.screen)
    locked = _p(loop).gold < 100
    return [
        EventOption(id="trade", label="Trade 100 Gold for a shared Relic.", apply=trade, locked=locked),
        EventOption(id="leave", label="Decline the trade.", apply=lambda _l: None),
    ]


@event("DOLL_ROOM")
def _doll_room(loop) -> list[EventOption]:
    # Three options: lose 5 HP + draw card; +5 Max HP; leave.
    def hug(loop):
        _damage(5)(loop)
        stream = loop.rng.stream("relic_pickup")
        if loop._reward_pool:
            _p(loop).deck.append(CardRef(stream.choice(loop._reward_pool)))
    def mend(loop):
        _max_hp(5)(loop)
    return [
        EventOption(id="hug", label="Hug a doll: -5 HP, gain a random card.", apply=hug),
        EventOption(id="mend", label="Mend a doll: +5 Max HP.", apply=mend),
        EventOption(id="leave", label="Leave the room.", apply=lambda _l: None),
    ]


# ---------------------------------------------------------------------------
# Helpers for the second batch
#
# Public game data for most of these events only lists titles and option
# names, not the exact numeric effects. The handlers below approximate each
# event with decision-affecting branches of the right shape (option count,
# cost / benefit polarity) so RL rollouts have meaningful choice signal; the
# precise numbers are chosen to be plausible but are explicitly not canon.


def _upgrade_random_of_type(card_type: str):
    def apply(loop):
        p = _p(loop)
        cards = loop.cards
        cands = [i for i, c in enumerate(p.deck)
                 if cards[c].upgrade_of is not None and cards[c].card_type == card_type]
        if cands:
            stream = loop.rng.stream("relic_pickup")
            idx = stream.choice(cands)
            cur = p.deck[idx]
            p.deck[idx] = CardRef(cards[cur].upgrade_of, like=cur)
    return apply


def _add_random_from(pool_attr: str, rarity: str | None = None):
    def apply(loop):
        p = _p(loop)
        pool = getattr(loop, pool_attr, None) or []
        if rarity is not None:
            pool = [c for c in pool if loop.cards[c].rarity == rarity]
        if pool:
            stream = loop.rng.stream("relic_pickup")
            p.deck.append(CardRef(stream.choice(pool)))
    return apply


def _combo(*fns):
    def apply(loop):
        for fn in fns:
            fn(loop)
    return apply


# ---------------------------------------------------------------------------
# Act 1 - Overgrowth (remaining)

@event("TABLET_OF_TRUTH")
def _tablet_of_truth(loop) -> list[EventOption]:
    return [
        EventOption(id="truth", label="Speak truth: upgrade a random card, lose 5 HP.",
                    apply=_combo(_damage(5), _upgrade_random_card())),
        EventOption(id="doubt", label="Doubt the tablet: gain a Curse.", apply=_add_curse("regret")),
        EventOption(id="leave", label="Leave the tablet alone.", apply=lambda _l: None),
    ]


@event("SAPPHIRE_SEED")
def _sapphire_seed(loop) -> list[EventOption]:
    locked = _p(loop).gold < 30
    return [
        EventOption(id="plant", label="Plant the seed: pay 30 Gold, gain 5 Max HP.",
                    apply=_combo(_gold(-30), _max_hp(5)), locked=locked),
        EventOption(id="crush", label="Crush the seed: gain 20 Gold.", apply=_gold(20)),
    ]


@event("DENSE_VEGETATION")
def _dense_vegetation(loop) -> list[EventOption]:
    return [
        EventOption(id="hack", label="Hack through: lose 8 HP, gain a random card.",
                    apply=_combo(_damage(8), _add_random_from("_reward_pool"))),
        EventOption(id="navigate", label="Navigate around: gain 30 Gold.", apply=_gold(30)),
    ]


@event("JUNGLE_MAZE_ADVENTURE")
def _jungle_maze(loop) -> list[EventOption]:
    def delve(loop):
        p = _p(loop)
        loss = max(1, p.max_hp * 15 // 100)
        p.hp = max(1, p.hp - loss)
        _upgrade_random_card()(loop)
    return [
        EventOption(id="delve", label="Delve deeper: lose 15% HP, upgrade a random card.", apply=delve),
        EventOption(id="escape", label="Find a safe path out.", apply=lambda _l: None),
    ]


@event("LUMINOUS_CHOIR")
def _luminous_choir(loop) -> list[EventOption]:
    return [
        EventOption(id="sing", label="Join the choir: heal 20 HP, gain a Curse.",
                    apply=_combo(_heal(20), _add_curse("decay"))),
        EventOption(id="silence", label="Keep silent and move on.", apply=lambda _l: None),
    ]


@event("BYRDONIS_NEST")
def _byrdonis_nest(loop) -> list[EventOption]:
    locked = _p(loop).gold < 25
    return [
        EventOption(id="steal", label="Steal an egg: lose 10 HP, gain a Colorless card.",
                    apply=_combo(_damage(10), _add_random_from("_colorless_pool"))),
        EventOption(id="feed", label="Feed the birds: pay 25 Gold, gain a potion.",
                    apply=_combo(_gold(-25), _add_random_potion()), locked=locked),
        EventOption(id="leave", label="Leave the nest in peace.", apply=lambda _l: None),
    ]


@event("WOOD_CARVINGS")
def _wood_carvings(loop) -> list[EventOption]:
    return [
        EventOption(id="study", label="Study the carvings: upgrade a random card.",
                    apply=_upgrade_random_card()),
        EventOption(id="smash", label="Smash the carvings: gain 40 Gold, gain a Curse.",
                    apply=_combo(_gold(40), _add_curse("regret"))),
        EventOption(id="leave", label="Walk past.", apply=lambda _l: None),
    ]


# ---------------------------------------------------------------------------
# Act 1 - Underdocks (remaining)

@event("DROWNING_BEACON")
def _drowning_beacon(loop) -> list[EventOption]:
    return [
        EventOption(id="light", label="Light the beacon: gain 50 Gold, gain a Curse.",
                    apply=_combo(_gold(50), _add_curse("decay"))),
        EventOption(id="extinguish", label="Extinguish it: upgrade a random card.",
                    apply=_upgrade_random_card()),
    ]


@event("ENDLESS_CONVEYOR")
def _endless_conveyor(loop) -> list[EventOption]:
    return [
        EventOption(id="grab", label="Grab an item: gain a random card.",
                    apply=_add_random_from("_reward_pool")),
        EventOption(id="observe", label="Watch it pass: heal 10 HP.", apply=_heal(10)),
    ]


@event("PUNCH_OFF")
def _punch_off(loop) -> list[EventOption]:
    def fight(loop):
        _damage(10)(loop)
        _upgrade_random_of_type("attack")(loop)
    return [
        EventOption(id="fight", label="Enter the punch-off: lose 10 HP, upgrade a random Attack.",
                    apply=fight),
        EventOption(id="watch", label="Watch from the sidelines.", apply=lambda _l: None),
    ]


@event("SPIRALING_WHIRLPOOL")
def _spiraling_whirlpool(loop) -> list[EventOption]:
    return [
        EventOption(id="jump", label="Jump in: gain a random card, lose 8 HP.",
                    apply=_combo(_damage(8), _add_random_from("_reward_pool"))),
        EventOption(id="swim", label="Swim away: heal 10 HP.", apply=_heal(10)),
    ]


@event("SUNKEN_STATUE")
def _sunken_statue(loop) -> list[EventOption]:
    return [
        EventOption(id="pray", label="Pray before the statue: gain 5 Max HP.", apply=_max_hp(5)),
        EventOption(id="examine", label="Examine the statue: gain 40 Gold.", apply=_gold(40)),
    ]


@event("DOORS_OF_LIGHT_AND_DARK")
def _doors_of_light_and_dark(loop) -> list[EventOption]:
    def dark(loop):
        _damage(10)(loop)
        _add_random_from("_reward_pool", rarity="rare")(loop)
    return [
        EventOption(id="light", label="Pass through the Light Door: heal 15 HP, remove a random card.",
                    apply=_combo(_heal(15), _remove_random_card())),
        EventOption(id="dark", label="Pass through the Dark Door: lose 10 HP, gain a random Rare card.",
                    apply=dark),
    ]


@event("WATERLOGGED_SCRIPTORIUM")
def _waterlogged_scriptorium(loop) -> list[EventOption]:
    return [
        EventOption(id="read", label="Read the sodden tomes: lose 3 Max HP, upgrade a random card.",
                    apply=_combo(_max_hp(-3), _upgrade_random_card())),
        EventOption(id="leave", label="Leave the ruined library.", apply=lambda _l: None),
    ]


# ---------------------------------------------------------------------------
# Act 2 - Hive (remaining)

@event("AMALGAMATOR")
def _amalgamator(loop) -> list[EventOption]:
    def fuse(loop):
        p = _p(loop)
        cards = loop.cards
        cands = [i for i, c in enumerate(p.deck) if cards[c].upgrade_of is not None]
        if len(cands) < 2:
            return
        stream = loop.rng.stream("relic_pickup")
        a, b = stream.sample(cands, 2) if hasattr(stream, "sample") else (cands[0], cands[1])
        for idx in sorted((a, b), reverse=True):
            if idx < len(p.deck):
                p.deck.pop(idx)
        if loop._reward_pool:
            p.deck.append(CardRef(stream.choice(loop._reward_pool)))
    return [
        EventOption(id="fuse", label="Amalgamate: remove two cards, gain a random card.", apply=fuse),
        EventOption(id="leave", label="Decline the fusion.", apply=lambda _l: None),
    ]


@event("BUGSLAYER")
def _bugslayer(loop) -> list[EventOption]:
    def duel(loop):
        p = _p(loop)
        loss = max(1, p.max_hp * 15 // 100)
        p.hp = max(1, p.hp - loss)
        _gold(40)(loop)
    return [
        EventOption(id="duel", label="Duel the Bugslayer: lose 15% HP, gain 40 Gold.", apply=duel),
        EventOption(id="run", label="Run away.", apply=lambda _l: None),
    ]


@event("COLORFUL_PHILOSOPHERS")
def _colorful_philosophers(loop) -> list[EventOption]:
    return [
        EventOption(id="red", label="Red: upgrade a random Attack.", apply=_upgrade_random_of_type("attack")),
        EventOption(id="blue", label="Blue: upgrade a random Skill.", apply=_upgrade_random_of_type("skill")),
        EventOption(id="green", label="Green: upgrade a random Power.", apply=_upgrade_random_of_type("power")),
    ]


@event("COLOSSAL_FLOWER")
def _colossal_flower(loop) -> list[EventOption]:
    return [
        EventOption(id="sniff", label="Sniff deeply: heal 20 HP, gain a Curse.",
                    apply=_combo(_heal(20), _add_curse("decay"))),
        EventOption(id="leave", label="Hold your breath and leave.", apply=lambda _l: None),
    ]


@event("FIELD_OF_MAN_SIZED_HOLES")
def _field_of_man_sized_holes(loop) -> list[EventOption]:
    def jump(loop):
        _damage(8)(loop)
        _gold(40)(loop)
        _add_random_from("_reward_pool")(loop)
    return [
        EventOption(id="jump", label="Climb into a hole: lose 8 HP, gain 40 Gold and a random card.",
                    apply=jump),
        EventOption(id="step", label="Step carefully around them.", apply=lambda _l: None),
    ]


@event("INFESTED_AUTOMATON")
def _infested_automaton(loop) -> list[EventOption]:
    locked = _p(loop).gold < 30
    return [
        EventOption(id="repair", label="Repair it: pay 30 Gold, gain 5 Max HP.",
                    apply=_combo(_gold(-30), _max_hp(5)), locked=locked),
        EventOption(id="scrap", label="Scrap it: gain a random Colorless card.",
                    apply=_add_random_from("_colorless_pool")),
    ]


@event("SPIRIT_GRAFTER")
def _spirit_grafter(loop) -> list[EventOption]:
    return [
        EventOption(id="graft", label="Accept the graft: upgrade a random card, gain a Curse.",
                    apply=_combo(_upgrade_random_card(), _add_curse("decay"))),
        EventOption(id="leave", label="Refuse the graft.", apply=lambda _l: None),
    ]


@event("THE_LANTERN_KEY")
def _the_lantern_key(loop) -> list[EventOption]:
    locked = _p(loop).gold < 50
    return [
        EventOption(id="unlock", label="Unlock the chest: pay 50 Gold, gain a Rare card.",
                    apply=_combo(_gold(-50), _add_random_from("_reward_pool", rarity="rare")),
                    locked=locked),
        EventOption(id="smash", label="Smash the chest: gain 25 Gold.", apply=_gold(25)),
    ]


# ---------------------------------------------------------------------------
# Act 3 - Glory (remaining)

@event("BATTLEWORN_DUMMY")
def _battleworn_dummy(loop) -> list[EventOption]:
    return [
        EventOption(id="attack", label="Train Attacks: upgrade a random Attack.",
                    apply=_upgrade_random_of_type("attack")),
        EventOption(id="defense", label="Train Defense: upgrade a random Skill.",
                    apply=_upgrade_random_of_type("skill")),
        EventOption(id="rest", label="Rest: heal 15 HP.", apply=_heal(15)),
    ]


@event("GRAVE_OF_THE_FORGOTTEN")
def _grave_of_the_forgotten(loop) -> list[EventOption]:
    return [
        EventOption(id="desecrate", label="Desecrate the grave: gain 50 Gold, gain a Curse.",
                    apply=_combo(_gold(50), _add_curse("regret"))),
        EventOption(id="honor", label="Honor the dead: heal 15 HP.", apply=_heal(15)),
    ]


@event("REFLECTIONS")
def _reflections(loop) -> list[EventOption]:
    def look(loop):
        p = _p(loop)
        loss = max(1, p.max_hp * 10 // 100)
        p.hp = max(1, p.hp - loss)
        _upgrade_random_card()(loop)
        _upgrade_random_card()(loop)
    return [
        EventOption(id="look", label="Look into the mirror: lose 10% HP, upgrade 2 random cards.",
                    apply=look),
        EventOption(id="turn", label="Turn away from the mirror.", apply=lambda _l: None),
    ]


@event("ROUND_TEA_PARTY")
def _round_tea_party(loop) -> list[EventOption]:
    return [
        EventOption(id="black", label="Black tea: heal 15 HP.", apply=_heal(15)),
        EventOption(id="green", label="Green tea: gain 5 Max HP.", apply=_max_hp(5)),
        EventOption(id="red", label="Red tea: gain 40 Gold.", apply=_gold(40)),
    ]


@event("TINKER_TIME")
def _tinker_time(loop) -> list[EventOption]:
    return [
        EventOption(id="tinker", label="Tinker with a card: lose 2 Max HP, upgrade a random card.",
                    apply=_combo(_max_hp(-2), _upgrade_random_card())),
        EventOption(id="leave", label="Leave the workshop.", apply=lambda _l: None),
    ]


# ---------------------------------------------------------------------------
# Shared (remaining)

@event("BRAIN_LEECH")
def _brain_leech(loop) -> list[EventOption]:
    return [
        EventOption(id="submit", label="Submit to the leech: lose 8 HP, upgrade a random card.",
                    apply=_combo(_damage(8), _upgrade_random_card())),
        EventOption(id="resist", label="Fight it off: lose 3 Max HP.", apply=_max_hp(-3)),
    ]


@event("FAKE_MERCHANT")
def _fake_merchant(loop) -> list[EventOption]:
    locked = _p(loop).gold < 50
    return [
        EventOption(id="buy", label="Buy: pay 50 Gold, gain a random card.",
                    apply=_combo(_gold(-50), _add_random_from("_reward_pool")), locked=locked),
        EventOption(id="haggle", label="Haggle: lose 10 HP, gain a random card.",
                    apply=_combo(_damage(10), _add_random_from("_reward_pool"))),
    ]


@event("RANWID_THE_ELDER")
def _ranwid_the_elder(loop) -> list[EventOption]:
    return [
        EventOption(id="wisdom", label="Ask for wisdom: upgrade a random card.",
                    apply=_upgrade_random_card()),
        EventOption(id="might", label="Ask for might: gain 5 Max HP.", apply=_max_hp(5)),
        EventOption(id="wealth", label="Ask for wealth: gain 50 Gold.", apply=_gold(50)),
    ]


@event("ROOM_FULL_OF_CHEESE")
def _room_full_of_cheese(loop) -> list[EventOption]:
    return [
        EventOption(id="eat", label="Eat the cheese: heal 20 HP, gain 10 Gold.",
                    apply=_combo(_heal(20), _gold(10))),
        EventOption(id="leave", label="Leave the room.", apply=lambda _l: None),
    ]


@event("SELF_HELP_BOOK")
def _self_help_book(loop) -> list[EventOption]:
    return [
        EventOption(id="read", label="Read the book: lose 2 Max HP, upgrade a random card.",
                    apply=_combo(_max_hp(-2), _upgrade_random_card())),
        EventOption(id="leave", label="Shelve the book.", apply=lambda _l: None),
    ]


@event("SLIPPERY_BRIDGE")
def _slippery_bridge(loop) -> list[EventOption]:
    return [
        EventOption(id="run", label="Sprint across: lose 10 HP, gain 50 Gold.",
                    apply=_combo(_damage(10), _gold(50))),
        EventOption(id="creep", label="Creep carefully across.", apply=lambda _l: None),
    ]


@event("STONE_OF_ALL_TIME")
def _stone_of_all_time(loop) -> list[EventOption]:
    return [
        EventOption(id="past", label="Touch the Past: gain a random Uncommon card.",
                    apply=_add_random_from("_reward_pool", rarity="uncommon")),
        EventOption(id="present", label="Touch the Present: heal 15 HP.", apply=_heal(15)),
        EventOption(id="future", label="Touch the Future: gain a random potion.", apply=_add_random_potion()),
    ]


@event("SYMBIOTE")
def _symbiote(loop) -> list[EventOption]:
    return [
        EventOption(id="accept", label="Accept the symbiote: gain 8 Max HP, gain a Curse.",
                    apply=_combo(_max_hp(8), _add_curse("decay"))),
        EventOption(id="reject", label="Reject the symbiote.", apply=lambda _l: None),
    ]


@event("TEA_MASTER")
def _tea_master(loop) -> list[EventOption]:
    return [
        EventOption(id="focus", label="Tea of Focus: upgrade a random card.",
                    apply=_upgrade_random_card()),
        EventOption(id="vigor", label="Tea of Vigor: heal 20 HP.", apply=_heal(20)),
        EventOption(id="wealth", label="Tea of Wealth: gain 40 Gold.", apply=_gold(40)),
    ]


@event("THE_ARCHITECT")
def _the_architect(loop) -> list[EventOption]:
    return [
        EventOption(id="learn", label="Learn from the Architect: upgrade a random card, gain a Curse.",
                    apply=_combo(_upgrade_random_card(), _add_curse("regret"))),
        EventOption(id="leave", label="Leave without a lesson.", apply=lambda _l: None),
    ]


@event("THE_FUTURE_OF_POTIONS")
def _the_future_of_potions(loop) -> list[EventOption]:
    return [
        EventOption(id="taste", label="Taste the new potion: lose 5 HP, gain a random potion.",
                    apply=_combo(_damage(5), _add_random_potion())),
        EventOption(id="leave", label="Pass on the sample.", apply=lambda _l: None),
    ]


@event("THE_LEGENDS_WERE_TRUE")
def _the_legends_were_true(loop) -> list[EventOption]:
    locked = _p(loop).gold < 40
    return [
        EventOption(id="believe", label="Pay 40 Gold tribute: gain 10 Max HP.",
                    apply=_combo(_gold(-40), _max_hp(10)), locked=locked),
        EventOption(id="scoff", label="Scoff and move on.", apply=lambda _l: None),
    ]


@event("THIS_OR_THAT")
def _this_or_that(loop) -> list[EventOption]:
    return [
        EventOption(id="this", label="This: gain 30 Gold.", apply=_gold(30)),
        EventOption(id="that", label="That: upgrade a random card.", apply=_upgrade_random_card()),
    ]


@event("WAR_HISTORIAN_REPY")
def _war_historian_repy(loop) -> list[EventOption]:
    return [
        EventOption(id="listen", label="Listen to Repy: lose 15 HP, upgrade a random card.",
                    apply=_combo(_damage(15), _upgrade_random_card())),
        EventOption(id="leave", label="Leave the storyteller.", apply=lambda _l: None),
    ]


@event("WELCOME_TO_WONGOS")
def _welcome_to_wongos(loop) -> list[EventOption]:
    return [
        EventOption(id="enter", label="Enter Wongos: heal 15 HP.", apply=_heal(15)),
        EventOption(id="decline", label="Decline the welcome.", apply=lambda _l: None),
    ]


@event("CRYSTAL_SPHERE")
def _crystal_sphere(loop) -> list[EventOption]:
    return [
        EventOption(id="peer", label="Peer into the sphere: lose 10 HP, gain a random Rare card.",
                    apply=_combo(_damage(10), _add_random_from("_reward_pool", rarity="rare"))),
        EventOption(id="leave", label="Look away.", apply=lambda _l: None),
    ]


__all__ = ["EventOption", "EVENT_HANDLERS", "EVENT_POOLS", "act_pool", "options_for", "event"]
