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


__all__ = ["EventOption", "EVENT_HANDLERS", "EVENT_POOLS", "act_pool", "options_for", "event"]
