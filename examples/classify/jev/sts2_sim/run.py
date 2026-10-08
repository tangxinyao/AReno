"""RunLoop: a whole Ironclad run -- Neow, three acts of map, rooms, rewards.

Run structure follows r33hab/sts2's `RunEngine` / `RunMapGenerator` /
`RunRewardGenerator` / `RunNonCombatEffects` (v0.107.1):

  * Act 1 is Overgrowth or Underdocks, then the Hive, then Glory. Every act's
    encounters are rolled up front: its weak fights (3 in Act 1, 2 later),
    then normal fights, fifteen elites and a boss, never repeating an
    encounter tag back to back. Ascension 10 adds a second, different boss
    to the last act.
  * Each act has a 7-column map (mapgen.py). Monster rooms take the next
    normal encounter, Elites the next elite; Unknown rooms roll Monster 10% /
    Treasure 2% / Shop 3% / Event otherwise, the odds of the types not
    rolled growing by their base each time.
  * Winning a fight opens the rewards screen: gold, maybe a potion, an elite's
    relic, a card reward (rewards.py; relics can add more). Rest sites heal
    30% of max HP or upgrade a card (plus Train / Dig with Girya / Shovel).
    Shops sell five Ironclad cards (one on sale), two Colorless cards, three
    relics, three potions and a card removal. Treasure rooms hold gold and a
    relic. Beating a boss moves to the next act, whose Ancient heals the
    player first (80% of missing HP from Ascension 2).
  * Relics come from per-run grab bags shuffled at the start: the player's bag
    (shared + Ironclad pools) for elites and shops, the shared bag for chests.

`encounter=` pins a single fight instead (Neow -> that combat -> game over),
the scope the earlier one-combat episodes used.

Action ids use STS2MCP action names (see `actions.py`):
  neow / ancient:  "choose_event_option:{i}"
  map:             "choose_map_node:{i}"          (travelable nodes by column)
  combat:          "play_card:{card_id}[:{enemy_pos}]", "use_potion:{slot}[:{enemy_pos}]",
                   "end_turn"
  hand_select:     "combat_select_card:{i}" / "combat_confirm_selection"
  card_select:     "select_card:{i}" / "confirm_selection" / "cancel_selection"
  rewards:         "claim_reward:{i}" / "discard_potion:{slot}" / "proceed"
  card_reward:     "select_card_reward:{i}" / "skip_card_reward"
  rest_site:       "choose_rest_option:{i}" / "proceed"
  shop:            "shop_purchase:{i}" / "proceed"
  treasure:        "claim_treasure_relic:{i}" / "proceed"
  event:           "choose_event_option:{i}"
  game_over:       "menu_select:main_menu"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import actions, mapgen, rewards
from .combat import CombatContext, CombatError
from .effects import EffectQueue
from .encounters import build_roster
from .enums import Character, CombatPhase, DecisionPoint, Outcome, Screen
from .hooks import HookBus
from .loader import load_all, load_ancient_event_pools, load_encounters, load_potions, load_relics
from .potion_effects import random_potion
from .relics import RelicEngine, gold_gained
from .rng import Rng
from .schemas import CardSchema, EncounterSchema, MonsterSchema, PotionSchema, PowerSchema, RelicSchema
from .state import (
    ActRooms,
    CardRef,
    DeckSelection,
    PlayerState,
    RewardItem,
    RunState,
    ShopItem,
)


_IRONCLAD_MAX_HP = 80
_IRONCLAD_START_GOLD = 99
_START_ENERGY = 3
_POTION_SLOTS = 3
_ELITE_SEQUENCE = 15
_WEAK_POOL = "weak"
ASC_WEARY_TRAVELER = 2
ASC_TIGHT_BELT = 4
ASC_ASCENDERS_BANE = 5
ASC_INFLATION = 6
ASC_DOUBLE_BOSS = 10
LATE_RELIC_FLOOR = 41  # RelicGrabBag.AllowedInSoloRun: after the Act 3 chest
GIRYA_LIFTS = 3

IRONCLAD_STARTING_DECK: tuple[str, ...] = (
    "strike_ironclad", "strike_ironclad", "strike_ironclad", "strike_ironclad", "strike_ironclad",
    "defend_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad",
    "bash",
)
IRONCLAD_STARTING_RELIC = "burning_blood"
ACT1_VARIANTS: tuple[str, ...] = ("overgrowth", "underdocks")
LATER_ACTS: tuple[str, ...] = ("hive", "glory")
# Which ancient event rolls at each act's Ancient room.
# From codex `acts.json` -> `ancients` field.
_ACT_ANCIENTS: dict[str, tuple[str, ...]] = {
    "overgrowth": ("NEOW",),
    "underdocks": ("NEOW",),
    "hive": ("OROBAS", "PAEL", "TEZCATARA"),
    "glory": ("NONUPEIPE", "TANX", "VAKUU"),
}
FIRST_COMBAT_POOL = _WEAK_POOL
_RELIC_RARITIES = ("common", "uncommon", "rare", "shop")
_UNREMOVABLE = frozenset({"ascenders_bane", "curse_of_the_bell"})

# Unknown map point: (room, base odds). Elite is never rolled (-1) without a relic.
_UNKNOWN_BASE = {"monster": 0.1, "elite": -1.0, "treasure": 0.02, "shop": 0.03}
_ROOM_OF_KIND = {
    mapgen.MONSTER: "monster", mapgen.ELITE: "elite", mapgen.BOSS: "boss", mapgen.REST: "rest",
    mapgen.SHOP: "shop", mapgen.TREASURE: "treasure", mapgen.UNKNOWN: "unknown", mapgen.ANCIENT: "ancient",
}
# Rest site options (rest_site_ui OPTION_*): id, name, description.
_REST_TEXT = {
    "HEAL": ("Rest", "Heal for 30% of your Max HP ({heal})."),
    "SMITH": ("Smith", "Upgrade a card in your Deck."),
    "LIFT": ("Train", "Start battles with +1 Strength. ({left} Left)"),
    "DIG": ("Dig", "Dig for a Relic."),
}
# Enchantments relics hand out on pickup: (enchant, amount, count, may pick fewer, card types).
_PICKUP_ENCHANTS = {
    "gnarled_hammer": ("sharp", 3, 3, True, ("attack",)),
    "kifuda": ("adroit", 3, 3, True, ("attack", "skill", "power")),
    "punch_dagger": ("momentum", 5, 1, False, ("attack",)),
    "royal_stamp": ("royally_approved", 0, 1, False, ("attack", "skill")),
}
_ENCHANT_NAMES = {"sharp": "Sharp", "adroit": "Adroit", "momentum": "Momentum",
                  "royally_approved": "Royally Approved", "swift": "Swift"}


@dataclass
class Decision:
    id: str
    text: str


class RunLoopError(Exception):
    pass


class RunLoop:
    """Owns one run's state. Not thread-safe; one RunLoop per worker.

    `encounter` pins a single fight (an encounter id from data/encounters.json,
    or a list of monster ids): the episode is Neow -> that combat -> game over.
    """

    def __init__(
        self,
        *,
        character: str = Character.IRONCLAD,
        ascension: int = 0,
        seed: int,
        max_steps: int = 20000,
        encounter: str | list[str] | None = None,
    ) -> None:
        if character != Character.IRONCLAD:
            raise RunLoopError(f"character {character!r} not supported")
        if not 0 <= ascension <= 10:
            raise RunLoopError("ascension must be in 0..10")

        self._character = character
        self._ascension = int(ascension)
        self._master_seed = int(seed)
        self._max_steps = int(max_steps)
        self._encounter_pin = encounter

        self._rng: Rng | None = None
        self._state: RunState | None = None
        self._effects: EffectQueue | None = None
        self._hooks: HookBus | None = None
        self._combat_ctx: CombatContext | None = None
        self._cards: dict[str, CardSchema] | None = None
        self._monsters: dict[str, MonsterSchema] | None = None
        self._powers: dict[str, PowerSchema] | None = None
        self._encounters: dict[str, EncounterSchema] | None = None
        self._potions: dict[str, PotionSchema] = {}
        self._relics: dict[str, RelicSchema] = {}
        self._reward_pool: list[str] = []
        self._colorless_pool: list[str] = []

    # ------------------------------------------------------------------
    # Public surface

    @property
    def single_combat(self) -> bool:
        return self._encounter_pin is not None

    @property
    def rng(self) -> Rng:
        self._require_started()
        assert self._rng is not None
        return self._rng

    @property
    def state(self) -> RunState:
        self._require_started()
        assert self._state is not None
        return self._state

    @property
    def hooks(self) -> HookBus:
        self._require_started()
        assert self._hooks is not None
        return self._hooks

    @property
    def effects(self) -> EffectQueue:
        self._require_started()
        assert self._effects is not None
        return self._effects

    @property
    def combat_ctx(self) -> CombatContext:
        self._require_started()
        assert self._combat_ctx is not None
        return self._combat_ctx

    @property
    def cards(self) -> dict[str, CardSchema]:
        assert self._cards is not None
        return self._cards

    @property
    def monsters(self) -> dict[str, MonsterSchema]:
        assert self._monsters is not None
        return self._monsters

    @property
    def powers(self) -> dict[str, PowerSchema]:
        assert self._powers is not None
        return self._powers

    @property
    def encounters(self) -> dict[str, EncounterSchema]:
        assert self._encounters is not None
        return self._encounters

    @property
    def potion_defs(self) -> dict[str, PotionSchema]:
        return self._potions

    @property
    def relics(self) -> dict[str, RelicSchema]:
        return self._relics

    def reset(self) -> dict[str, Any]:
        self._rng = Rng(self._master_seed)
        self._effects = EffectQueue()
        self._hooks = HookBus()
        if self._cards is None:
            self._powers, self._cards, self._monsters = load_all()
            self._encounters = load_encounters(monster_ids=set(self._monsters))
            self._potions = load_potions()
            self._relics = load_relics()
            self._ancient_pools = load_ancient_event_pools()
            self._gid_to_rid = {r.game_id: rid for rid, r in self._relics.items()}
            self._reward_pool = rewards.reward_pool(self._cards)
            self._colorless_pool = rewards.reward_pool(self._cards, "colorless")
        asc = self._ascension
        hp = _IRONCLAD_MAX_HP if asc < ASC_WEARY_TRAVELER else int(_IRONCLAD_MAX_HP * 0.8)
        deck = [CardRef(c) for c in IRONCLAD_STARTING_DECK]
        if asc >= ASC_ASCENDERS_BANE:
            deck.append(CardRef("ascenders_bane"))
        self._state = RunState(
            character=self._character,
            ascension=asc,
            seed=self._master_seed,
            floor=1,
            player=PlayerState(
                hp=hp,
                max_hp=_IRONCLAD_MAX_HP,
                gold=_IRONCLAD_START_GOLD,
                max_energy=_START_ENERGY,
                energy=_START_ENERGY,
                deck=deck,
                relics=[IRONCLAD_STARTING_RELIC],
                potions=[None] * (_POTION_SLOTS - 1 if asc >= ASC_TIGHT_BELT else _POTION_SLOTS),
            ),
            unknown_odds=dict(_UNKNOWN_BASE),
            card_rarity_offset=rewards.CARD_RARITY_BASE_OFFSET,
        )
        assert self._cards is not None and self._monsters is not None and self._powers is not None
        self._combat_ctx = CombatContext(
            run=self._state,
            cards=self._cards,
            monsters=self._monsters,
            powers=self._powers,
            rng=self._rng,
            hooks=self._hooks,
            effects=self._effects,
            encounters=self._encounters,
            potions=self._potions,
        )
        if not self.single_combat:
            self._fill_relic_bags()
            self._generate_acts()
            self._enter_act_map(0)
            self._open_ancient_event()
        self._state.screen = Screen.NEOW
        self._state.room = "ancient"
        return self._packet()

    def step(self, action_id: str) -> dict[str, Any]:
        self._require_started()
        state = self.state
        if state.is_terminal():
            raise RunLoopError("run already terminal; call reset()")
        if state.steps >= self._max_steps:
            raise RunLoopError("max_steps exceeded")

        legal = {d.id for d in self._decisions()}
        if action_id not in legal:
            raise RunLoopError(f"illegal action {action_id!r} at screen {state.screen!r}")

        name, args = actions.parse_action(action_id)
        screen = state.screen
        if screen == Screen.NEOW:
            if name == actions.SELECT_RELIC:
                idx = int(args[0])
                rid = state.ancient_choices[idx]
                state.ancient_choices = []
                state.ancient_event = None
                self._obtain_relic(rid, return_to=Screen.MAP)
                if state.screen == Screen.NEOW:
                    self._leave_neow()
            else:
                state.ancient_choices = []
                state.ancient_event = None
                self._leave_neow()
        elif screen == Screen.ANCIENT:
            self._step_ancient(name, args)
        elif screen == Screen.MAP:
            self._choose_map_node(int(args[0]))
        elif screen == Screen.COMBAT:
            self._handle_combat_action(name, args)
            self._maybe_finalize_combat()
        elif screen == Screen.REWARDS:
            self._step_rewards(name, args)
        elif screen == Screen.CARD_REWARD:
            self._step_card_reward(name, args)
        elif screen == Screen.CARD_SELECT:
            self._step_deck_select(name, args)
        elif screen == Screen.REST:
            self._step_rest(name, args)
        elif screen == Screen.SHOP:
            self._step_shop(name, args)
        elif screen == Screen.TREASURE:
            self._step_treasure(name, args)
        elif screen == Screen.EVENT:
            self._enter_map_screen()
        else:  # pragma: no cover - gated by the legal set
            raise RunLoopError(f"unhandled action {action_id!r} at screen {screen!r}")

        state.steps += 1
        return self._packet()

    def close(self) -> None:
        self._rng = None
        self._state = None
        self._effects = None
        self._hooks = None
        self._combat_ctx = None

    # ------------------------------------------------------------------
    # Run generation

    def _require_started(self) -> None:
        if self._state is None:
            raise RunLoopError("RunLoop not started; call reset() first")

    def _fill_relic_bags(self) -> None:
        """RunManager.InitializeNewRun: the shared bag, then the player's (shared + Ironclad)."""

        state = self.state
        stream = self.rng.stream("up_front")
        shared = [r for r in self._relics.values() if r.pool == "shared" and r.rarity in _RELIC_RARITIES]
        ironclad = [r for r in self._relics.values() if r.pool == "ironclad" and r.rarity in _RELIC_RARITIES]
        for bag, pool in ((state.shared_relic_bag, shared), (state.relic_bag, shared + ironclad)):
            for rarity in _RELIC_RARITIES:
                ids = sorted(r.relic_id for r in pool if r.rarity == rarity)
                stream.shuffle(ids)
                bag[rarity] = ids

    def _generate_acts(self) -> None:
        """RunManager.GenerateRooms: every act's encounter sequence up front."""

        state = self.state
        act1 = self.rng.stream("act_selection").choice(ACT1_VARIANTS)
        stream = self.rng.stream("up_front")
        state.acts = [self._generate_rooms(act, stream) for act in (act1, *LATER_ACTS)]
        if state.ascension >= ASC_DOUBLE_BOSS:
            last = state.acts[-1]
            others = [e for e in self._pool(last.act, "boss") if e != last.boss]
            last.second_boss = stream.choice(others)

    def _pool(self, act: str, pool: str) -> list[str]:
        return sorted(e.encounter_id for e in self.encounters.values() if e.act == act and e.pool == pool)

    def _generate_rooms(self, act: str, stream) -> ActRooms:
        weak_count, room_count = mapgen.ACT_ROOMS[act]
        normal: list[str] = []
        last: str | None = None
        bag: list[str] = []
        for _ in range(weak_count):
            if not bag:
                bag = self._pool(act, "weak")
            last = self._grab(bag, last, stream)
            normal.append(last)
        bag = []
        for _ in range(weak_count, room_count):
            if not bag:
                bag = self._pool(act, "normal")
            last = self._grab(bag, last, stream)
            normal.append(last)
        elites: list[str] = []
        bag = []
        last_elite: str | None = None
        for _ in range(_ELITE_SEQUENCE):
            if not bag:
                bag = self._pool(act, "elite")
            last_elite = self._grab(bag, last_elite, stream)
            elites.append(last_elite)
        boss = stream.choice(self._pool(act, "boss"))
        return ActRooms(act=act, normal=normal, elite=elites, boss=boss)

    def _grab(self, bag: list[str], last: str | None, stream) -> str:
        """GrabWithoutRepeatingTags: never the last encounter or one sharing its tags."""

        last_tags = set(self.encounters[last].tags) if last else set()

        def ok(e: str) -> bool:
            return e != last and not (set(self.encounters[e].tags) & last_tags)

        any_valid = any(ok(e) for e in bag)
        while True:
            i = stream.randrange(len(bag))
            if not any_valid or ok(bag[i]):
                return bag.pop(i)

    def _enter_act_map(self, act_index: int) -> None:
        state = self.state
        state.act_index = act_index
        state.act = act_index + 1
        rooms = state.acts[act_index]
        second = rooms.second_boss is not None
        state.map = mapgen.generate_act_map(rooms.act, state.ascension,
                                            self.rng.stream(f"act_{act_index + 1}_map"), second_boss=second)
        state.map_coord = None
        state.normal_visited = 0
        state.elite_visited = 0
        state.unknown_odds = dict(_UNKNOWN_BASE)

    # ------------------------------------------------------------------
    # Relics, gold, deck

    def _relic_allowed(self, relic_id: str) -> bool:
        r = self._relics[relic_id]
        return relic_id not in self.state.player.relics and not (
            r.stops_after_act3_chest and self.state.floor >= LATE_RELIC_FLOOR)

    def _pull_relic(self, bag: dict[str, list[str]], rarity: str, *, front: bool = True,
                    shop: bool = False) -> str | None:
        """RelicGrabBag.Pull: the rarity's deque, falling back Shop -> Common -> Uncommon -> Rare."""

        chain = {"shop": "common", "common": "uncommon", "uncommon": "rare", "rare": None}
        r: str | None = rarity
        while r is not None:
            deque = bag.get(r, [])
            deque[:] = [x for x in deque if self._relic_allowed(x)]
            order = deque if front else list(reversed(deque))
            pick = next((x for x in order if not shop or self._relics[x].in_shops), None)
            if pick is not None:
                deque.remove(pick)
                return pick
            r = chain[r]
        return None

    def _roll_relic_rarity(self, stream) -> str:
        roll = stream.random()
        return "common" if roll < 0.5 else "uncommon" if roll < 0.83 else "rare"

    def _next_relic(self, stream) -> str:
        """RunRewardGenerator.NextRelic: a rolled rarity from the front of the player's bag."""

        relic = self._pull_relic(self.state.relic_bag, self._roll_relic_rarity(stream))
        if relic is None:
            return "circlet"
        self._drop_from_bags(relic)
        return relic

    def _drop_from_bags(self, relic: str) -> None:
        for bag in (self.state.relic_bag, self.state.shared_relic_bag):
            for deque in bag.values():
                if relic in deque:
                    deque.remove(relic)

    def _gain_gold(self, amount: int) -> None:
        p = self.state.player
        p.gold += gold_gained(p, amount)

    def _heal(self, amount: int) -> None:
        p = self.state.player
        p.hp = min(p.max_hp, p.hp + max(0, amount))

    def _gain_max_hp(self, amount: int) -> None:
        p = self.state.player
        p.max_hp += amount
        p.hp += amount

    def _add_card_to_deck(self, card: str) -> None:
        """AddCardToDeck: egg upgrades, Lucky Fysh, Book of Five Rings."""

        p = self.state.player
        ref = card if isinstance(card, CardRef) else CardRef(card)
        cdef = self.cards[ref]
        egg = {"power": "frozen_egg", "attack": "molten_egg", "skill": "toxic_egg"}.get(cdef.card_type)
        if egg in p.relics and cdef.upgrade_of is not None:
            ref = CardRef(cdef.upgrade_of, like=ref)
        p.deck.append(ref)
        if "lucky_fysh" in p.relics:
            self._gain_gold(15)
        if "book_of_five_rings" in p.relics:
            n = p.relic_state.get("book_of_five_rings", 0) + 1
            p.relic_state["book_of_five_rings"] = n % 5
            if n % 5 == 0:
                self._heal(20)

    def _upgrade_random(self, card_type: str, count: int) -> None:
        p = self.state.player
        cands = [i for i, c in enumerate(p.deck) if self.cards[c].card_type == card_type
                 and self.cards[c].upgrade_of is not None]
        stream = self.rng.stream("relic_pickup")
        stream.shuffle(cands)
        for i in cands[:count]:
            p.deck[i] = CardRef(self.cards[p.deck[i]].upgrade_of, like=p.deck[i])

    def _obtain_relic(self, relic_id: str, *, return_to: str) -> None:
        """ApplyRelicPickup. May open a deck selection or extra rewards."""

        state = self.state
        p = state.player
        if relic_id != "circlet" and relic_id in p.relics:
            return
        p.relics.append(relic_id)
        self._drop_from_bags(relic_id)
        if relic_id == "potion_belt":
            p.potions.extend([None, None])
        elif relic_id == "war_paint":
            self._upgrade_random("skill", 2)
        elif relic_id == "whetstone":
            self._upgrade_random("attack", 2)
        elif relic_id == "strawberry":
            self._gain_max_hp(7)
        elif relic_id == "pear":
            self._gain_max_hp(10)
        elif relic_id == "mango":
            self._gain_max_hp(14)
        elif relic_id == "lees_waffle":
            self._gain_max_hp(7)
            p.hp = p.max_hp
        elif relic_id == "old_coin":
            self._gain_gold(300)
        elif relic_id == "orrery":
            for _ in range(5):
                self._push_reward(self._card_reward_item("monster"), return_to)
        elif relic_id == "cauldron":
            stream = self.rng.stream("rewards")
            for _ in range(5):
                self._push_reward(RewardItem(kind="potion", potion=random_potion(self._potions, stream)), return_to)
        elif relic_id == "dollys_mirror":
            self._open_deck_select("duplicate", list(range(len(p.deck))), 1, return_to)
        elif relic_id in _PICKUP_ENCHANTS:
            enchant, amount, count, fewer, types = _PICKUP_ENCHANTS[relic_id]
            cands = [i for i, c in enumerate(p.deck) if self.cards[c].card_type in types
                     and getattr(c, "enchant", None) is None and not self.cards[c].unplayable]
            if cands:
                self._open_deck_select("enchant", cands, count, return_to, min_count=0 if fewer else None,
                                       enchant=enchant, enchant_amount=amount)
        elif relic_id == "nutritious_oyster":
            self._gain_max_hp(11)
        elif relic_id == "looming_fruit":
            self._gain_max_hp(31)
        elif relic_id == "golden_pearl":
            self._gain_gold(150)
        elif relic_id == "signet_ring":
            self._gain_gold(999)
        elif relic_id == "cursed_pearl":
            self._add_card_if_exists(p, "greed")
            self._gain_gold(333)
        elif relic_id == "neows_torment":
            self._add_card_if_exists(p, "neows_fury")
        elif relic_id == "storybook":
            self._add_card_if_exists(p, "brightest_flame")
        elif relic_id == "tanxs_whistle":
            self._add_card_if_exists(p, "whistle")
        elif relic_id == "jewelry_box":
            self._add_card_if_exists(p, "apotheosis")
        elif relic_id == "distinguished_cape":
            self._lose_max_hp(9)
            for _ in range(3):
                self._add_card_if_exists(p, "apparition")
        elif relic_id == "blood_soaked_rose":
            self._add_card_if_exists(p, "enthralled")
        elif relic_id == "preserved_fog":
            self._add_card_if_exists(p, "folly")
        elif relic_id == "sere_talon":
            stream = self.rng.stream("relic_pickup")
            curses = sorted(c.card_id for c in self.cards.values() if c.card_type == "curse"
                            and c.card_id not in _UNREMOVABLE)
            if curses:
                stream.shuffle(curses)
                for cid in curses[:2]:
                    p.deck.append(CardRef(cid))
        elif relic_id == "small_capsule":
            stream = self.rng.stream("relic_pickup")
            self._offer_random_shared_relic(stream, return_to)
        elif relic_id == "large_capsule":
            stream = self.rng.stream("relic_pickup")
            for _ in range(2):
                self._offer_random_shared_relic(stream, return_to)
            p.deck.append(CardRef("strike_ironclad"))
            p.deck.append(CardRef("defend_ironclad"))
        elif relic_id == "pomander":
            self._upgrade_random_any(1)
        elif relic_id == "yummy_cookie":
            self._upgrade_random_any(4)
        elif relic_id == "sand_castle":
            self._upgrade_random_any(6)
        elif relic_id == "precarious_shears":
            self._lose_hp(13)
            cands = [i for i, c in enumerate(p.deck) if c not in _UNREMOVABLE]
            if cands:
                self._open_deck_select("remove", cands, 2, return_to)
        elif relic_id == "precise_scissors":
            cands = [i for i, c in enumerate(p.deck) if c not in _UNREMOVABLE]
            if cands:
                self._open_deck_select("remove", cands, 1, return_to)
        elif relic_id == "empty_cage":
            cands = [i for i, c in enumerate(p.deck) if c not in _UNREMOVABLE]
            if cands:
                self._open_deck_select("remove", cands, 2, return_to)
        elif relic_id == "stone_humidifier":
            # Resting raises Max HP: handled at rest time via relic_state flag.
            p.relic_state.setdefault("stone_humidifier", True)

    def _add_card_if_exists(self, player, card_id: str) -> None:
        if card_id in self.cards:
            player.deck.append(CardRef(card_id))

    def _upgrade_random_any(self, count: int) -> None:
        p = self.state.player
        cands = [i for i, c in enumerate(p.deck) if self.cards[c].upgrade_of is not None]
        if not cands:
            return
        stream = self.rng.stream("relic_pickup")
        stream.shuffle(cands)
        for i in cands[:count]:
            p.deck[i] = CardRef(self.cards[p.deck[i]].upgrade_of, like=p.deck[i])

    def _offer_random_shared_relic(self, stream, return_to: str) -> None:
        p = self.state.player
        pool = sorted(r.relic_id for r in self._relics.values()
                      if r.pool in ("shared", "ironclad")
                      and r.rarity in _RELIC_RARITIES
                      and r.relic_id not in p.relics)
        if not pool:
            return
        rid = stream.choice(pool)
        self._obtain_relic(rid, return_to=return_to)

    def _lose_hp(self, amount: int) -> None:
        p = self.state.player
        if amount <= 0 or p is None:
            return
        p.hp = max(1, p.hp - amount)

    def _lose_max_hp(self, amount: int) -> None:
        p = self.state.player
        if amount <= 0 or p is None:
            return
        p.max_hp = max(1, p.max_hp - amount)
        p.hp = min(p.hp, p.max_hp)

    def _push_reward(self, item: RewardItem, return_to: str) -> None:
        state = self.state
        state.rewards.append(item)
        if state.screen != Screen.REWARDS and state.deck_select is None:
            state.rewards_return = return_to
            state.screen = Screen.REWARDS

    def _open_deck_select(self, purpose: str, cands: list[int], count: int, return_to: str, *,
                          min_count: int | None = None, enchant: str | None = None, enchant_amount: int = 0,
                          price: int = 0, cancelable: bool = False) -> None:
        state = self.state
        state.deck_select = DeckSelection(purpose=purpose, candidates=cands, count=min(count, len(cands)),
                                          source=return_to, min_count=min_count, enchant=enchant,
                                          enchant_amount=enchant_amount, price=price, cancelable=cancelable)
        state.screen = Screen.CARD_SELECT

    # ------------------------------------------------------------------
    # Neow / Ancient / map

    def _leave_neow(self) -> None:
        state = self.state
        if self.single_combat:
            state.screen = Screen.COMBAT
            monster_ids, flags = self._pick_pinned_encounter()
            self._start_combat(monster_ids, flags, room="monster")
            return
        # Act 1's Ancient is Neow; the run stands on it, so the map opens on row 1.
        state.map_coord = state.map.start
        self._enter_map_screen()

    def _step_ancient(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        if name == actions.SELECT_RELIC:
            idx = int(args[0])
            rid = state.ancient_choices[idx]
            state.ancient_choices = []
            state.ancient_event = None
            self._obtain_relic(rid, return_to=Screen.MAP)
            if state.screen == Screen.ANCIENT:
                self._enter_map_screen()
            return
        if name == actions.CHOOSE_EVENT_OPTION:
            # Legacy "proceed" path for old traces.
            state.ancient_choices = []
            state.ancient_event = None
            self._enter_map_screen()
            return
        state.ancient_choices = []
        state.ancient_event = None
        self._enter_map_screen()

    def _enter_map_screen(self) -> None:
        self.state.screen = Screen.MAP
        self.state.room = None

    def _map_options(self) -> list[mapgen.MapNode]:
        state = self.state
        m = state.map
        if state.map_coord is None:
            return [m.node(m.start)]
        return sorted((m.node(c) for c in m.node(state.map_coord).children), key=lambda n: (n.col, n.row))

    def _choose_map_node(self, index: int) -> None:
        state = self.state
        p = state.player
        node = self._map_options()[index]
        state.map_coord = node.coord
        state.floor += 1
        room = _ROOM_OF_KIND[node.kind]
        if room == "unknown":
            room = self._roll_unknown()
            if "planisphere" in p.relics:
                self._heal(5)
        state.room = room
        if room == "ancient":
            self._ancient_heal()
            self._open_ancient_event()
            state.screen = Screen.ANCIENT
            return
        if room in ("monster", "elite", "boss"):
            if room == "boss" and "pantograph" in p.relics:
                self._heal(25)
            enc_id = self._next_encounter(room, node)
            self._start_encounter(enc_id, room)
        elif room == "rest":
            self._enter_rest()
        elif room == "shop":
            self._enter_shop()
        elif room == "treasure":
            self._enter_treasure()
        else:
            state.screen = Screen.EVENT
        state.last_room = room

    def _next_encounter(self, room: str, node: mapgen.MapNode) -> str:
        state = self.state
        rooms = state.acts[state.act_index]
        if room == "monster":
            enc = rooms.normal[state.normal_visited % len(rooms.normal)]
            state.normal_visited += 1
            return enc
        if room == "elite":
            enc = rooms.elite[state.elite_visited % len(rooms.elite)]
            state.elite_visited += 1
            return enc
        second = any(state.map.node(p).kind == mapgen.BOSS for p in node.parents)
        return rooms.second_boss if second and rooms.second_boss else rooms.boss

    def _roll_unknown(self) -> str:
        """RunEngine.RollUnknownMapPointNodeType."""

        state = self.state
        allowed = {"monster", "elite", "treasure", "shop", "event"}
        options = self._map_options()
        if state.last_room == "shop" or (options and all(n.kind == mapgen.SHOP for n in options)):
            allowed.discard("shop")
        if "juzu_bracelet" in state.player.relics:
            allowed.discard("monster")
        roll = self.rng.stream("unknown_map_point").random()
        rolled = "event"
        acc = 0.0
        for room in ("monster", "elite", "treasure", "shop"):
            odds = state.unknown_odds[room]
            if room not in allowed or odds < 0:
                continue
            acc += odds
            if roll <= acc:
                rolled = room
                break
        for room, base in _UNKNOWN_BASE.items():
            if rolled == room:
                state.unknown_odds[room] = base
            elif room in allowed:
                state.unknown_odds[room] += base
        return rolled

    def _ancient_heal(self) -> None:
        p = self.state.player
        assert p is not None
        missing = p.max_hp - p.hp
        if missing > 0:
            p.hp += int(missing * 0.8) if self.state.ascension >= ASC_WEARY_TRAVELER else missing

    def _open_ancient_event(self) -> None:
        """Pick the act's ancient and roll 3 unowned relics from its pool."""

        state = self.state
        p = state.player
        assert p is not None
        act = state.acts[state.act_index].act
        stream = self.rng.stream("ancient_event")
        event_id = stream.choice(_ACT_ANCIENTS[act])
        state.ancient_event = event_id
        pool_gids = self._ancient_pools.get(event_id, ())
        candidates = []
        for gid in pool_gids:
            rid = self._gid_to_rid.get(gid)
            if rid is None or rid in p.relics:
                continue
            candidates.append(rid)
        stream.shuffle(candidates)
        state.ancient_choices = candidates[:3]

    # ------------------------------------------------------------------
    # Combat

    def _pick_pinned_encounter(self) -> tuple[list[str], list[dict[str, Any]]]:
        state = self.state
        pin = self._encounter_pin
        stream = self.rng.stream("encounter")
        if isinstance(pin, list):
            state.encounter_id = None
            return list(pin), [{} for _ in pin]
        assert isinstance(pin, str)
        enc = self.encounters[pin]
        state.encounter_id = enc.encounter_id
        roster = build_roster(enc, stream)
        return [mid for mid, _ in roster], [flags for _, flags in roster]

    def _start_encounter(self, enc_id: str, room: str) -> None:
        enc = self.encounters[enc_id]
        self.state.encounter_id = enc_id
        roster = build_roster(enc, self.rng.stream("encounter"))
        self._start_combat([mid for mid, _ in roster], [flags for _, flags in roster], room=room)

    def _start_combat(self, monster_ids: list[str], flags: list[dict[str, Any]], *, room: str) -> None:
        state = self.state
        state.screen = Screen.COMBAT
        state.room = room
        p = state.player
        assert p is not None
        ctx = self.combat_ctx
        ctx.relics = RelicEngine(ctx)
        ctx.start_combat(monster_ids, list(p.deck), monster_flags=flags, room=room)
        self._maybe_finalize_combat()

    def _handle_combat_action(self, name: str, args: tuple[str, ...]) -> None:
        ctx = self.combat_ctx
        try:
            sel = ctx.combat.pending_selection
            if name == actions.END_TURN and not args:
                ctx.end_turn()
            elif name == actions.PLAY_CARD and len(args) == 1:
                ctx.play_card(args[0])
            elif name == actions.PLAY_CARD and len(args) == 2:
                ctx.play_card(args[0], target_slot=self._alive_monsters()[int(args[1])].slot)
            elif name == actions.USE_POTION:
                target = self._alive_monsters()[int(args[1])].slot if len(args) == 2 else None
                ctx.use_potion(int(args[0]), target_slot=target)
            elif name in (actions.COMBAT_SELECT_CARD, actions.SELECT_CARD) and sel is not None:
                ctx.toggle_selection(sel.candidates[int(args[0])])
            elif name in (actions.COMBAT_CONFIRM_SELECTION, actions.CONFIRM_SELECTION):
                ctx.confirm_selection()
            elif name == actions.CANCEL_SELECTION:
                ctx.confirm_selection(skip=True)
            else:
                raise RunLoopError(f"unknown combat action {name!r}")
        except CombatError as exc:
            raise RunLoopError(f"combat engine rejected {name}:{':'.join(args)}: {exc}") from exc

    def _maybe_finalize_combat(self) -> None:
        state = self.state
        combat = state.combat
        assert combat is not None
        if combat.outcome is None:
            return
        p = state.player
        assert p is not None
        # A Thieving Hopper that got away keeps the card for the rest of the run.
        for _, card in combat.stolen:
            if card in p.deck:
                p.deck.remove(card)
        combat.stolen.clear()
        p.block = 0
        p.energy = p.max_energy
        p.powers = {}
        if combat.outcome != "victory":
            self._game_over(Outcome.DEATH)
            return
        if self.single_combat:
            self._game_over(Outcome.VICTORY)
            return
        # GenerateCombatRewards: the healing relics act first.
        if "burning_blood" in p.relics:
            self._heal(6)
        if "black_blood" in p.relics:
            self._heal(12)
        if "meat_on_the_bone" in p.relics and p.hp <= p.max_hp // 2:
            self._heal(12)
        room = state.room
        if room == "boss" and state.act_index >= len(state.acts) - 1:
            if not self._boss_remaining():
                self._game_over(Outcome.VICTORY)
                return
            # A10's first boss of the last act: straight on to the second.
            self._enter_map_screen()
            return
        self._open_combat_rewards(room or "monster", took_damage=combat.took_unblocked)

    def _boss_remaining(self) -> bool:
        state = self.state
        node = state.map.node(state.map_coord)
        return any(state.map.node(c).kind == mapgen.BOSS for c in node.children)

    def _game_over(self, outcome: str) -> None:
        self.state.screen = Screen.GAME_OVER
        self.state.outcome = outcome

    # ------------------------------------------------------------------
    # Rewards

    def _card_reward_item(self, room: str, *, took_damage: bool = True, rare: bool = False) -> RewardItem:
        state = self.state
        p = state.player
        pool = self._reward_pool + (self._colorless_pool if "dingy_rug" in p.relics else [])
        stream = self.rng.stream("rewards")
        cards, state.card_rarity_offset = rewards.card_reward(
            self.cards, pool, "boss" if rare else room, ascension=state.ascension, act_index=state.act_index,
            offset=state.card_rarity_offset, rng=stream)
        refs = [CardRef(c) for c in cards]
        if "lava_lamp" in p.relics and not took_damage:
            refs = [CardRef(self.cards[c].upgrade_of, like=c) if self.cards[c].upgrade_of else c for c in refs]
        if "lasting_candy" in p.relics and p.relic_state.get("lasting_candy", 0) % 2 == 0 \
                and p.relic_state.get("lasting_candy", 0) > 0:
            powers = [c for c in pool if self.cards[c].card_type == "power" and c not in cards]
            if powers:
                refs.append(CardRef(stream.choice(powers)))
        if "wing_charm" in p.relics:
            target = stream.choice(refs)
            target.enchant, target.enchant_amount = "swift", 1
        return RewardItem(kind="card", cards=refs)

    def _open_combat_rewards(self, room: str, *, took_damage: bool) -> None:
        state = self.state
        p = state.player
        stream = self.rng.stream("rewards")
        if "lasting_candy" in p.relics:
            p.relic_state["lasting_candy"] = p.relic_state.get("lasting_candy", 0) + 1
        gold = rewards.combat_gold(room, state.ascension, stream)
        if "amethyst_aubergine" in p.relics:
            gold += 15
        items: list[RewardItem] = [RewardItem(kind="gold", gold=gold)]
        dropped, state.potion_odds = rewards.roll_potion_drop(state.potion_odds, room, stream)
        if dropped or "white_beast_statue" in p.relics:
            items.append(RewardItem(kind="potion", potion=random_potion(self._potions, stream)))
        if room == "elite":
            items.append(RewardItem(kind="relic", relic=self._next_relic(stream)))
            # Ancient: black_star drops an extra relic from elites.
            if "black_star" in p.relics:
                items.append(RewardItem(kind="relic", relic=self._next_relic(stream)))
        if room == "boss" and state.act_index == 0 and "lava_rock" in p.relics:
            # Ancient: Act 1 boss drops 2 relics.
            items.append(RewardItem(kind="relic", relic=self._next_relic(stream)))
            items.append(RewardItem(kind="relic", relic=self._next_relic(stream)))
        items.append(self._card_reward_item(room, took_damage=took_damage))
        if room == "monster" and "prayer_wheel" in p.relics:
            items.append(self._card_reward_item(room, took_damage=took_damage))
        if room == "elite" and "white_star" in p.relics:
            items.append(self._card_reward_item(room, took_damage=took_damage, rare=True))
        state.rewards = items
        state.rewards_return = None
        state.screen = Screen.REWARDS

    def _step_rewards(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        p = state.player
        assert p is not None
        if name == actions.PROCEED:
            self._leave_rewards()
            return
        if name == actions.DISCARD_POTION:
            p.potions[int(args[0])] = None
            return
        index = int(args[0])
        item = state.rewards[index]
        if item.kind == "gold":
            self._gain_gold(item.gold)
        elif item.kind == "potion":
            p.potions[p.potions.index(None)] = item.potion
        elif item.kind == "relic":
            state.rewards.pop(index)
            self._obtain_relic(item.relic, return_to=Screen.REWARDS)
            return
        elif item.kind == "card":
            state.card_reward = list(item.cards)
            state.card_reward_item = index
            state.screen = Screen.CARD_REWARD
            return
        state.rewards.pop(index)

    def _step_card_reward(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        assert state.card_reward is not None and state.card_reward_item is not None
        if name == actions.SELECT_CARD_REWARD:
            self._add_card_to_deck(state.card_reward[int(args[0])])
        state.rewards.pop(state.card_reward_item)
        state.card_reward = None
        state.card_reward_item = None
        state.screen = Screen.REWARDS

    def _leave_rewards(self) -> None:
        state = self.state
        state.rewards = []
        back = state.rewards_return
        state.rewards_return = None
        if back is not None:
            state.screen = back
            return
        if state.room == "boss":
            self._enter_act_map(state.act_index + 1)
        self._enter_map_screen()

    # ------------------------------------------------------------------
    # Rest site

    def _enter_rest(self) -> None:
        state = self.state
        p = state.player
        p.relic_state.pop("rest_taken", None)
        if "venerable_tea_set" in p.relics:
            p.relic_state["venerable_tea_set"] = 1
        if "eternal_feather" in p.relics:
            self._heal(len(p.deck) // 5 * 3)
        state.rest_used = False
        state.screen = Screen.REST

    def _rest_heal(self) -> int:
        p = self.state.player
        return max(1, int(p.max_hp * 0.3)) + (15 if "regal_pillow" in p.relics else 0)

    def _upgradable_deck(self) -> list[int]:
        cards = self.cards
        return [i for i, c in enumerate(self.state.player.deck)
                if cards[c].upgrade_of is not None and cards[c].card_type not in ("status", "curse")]

    def _rest_options(self) -> list[tuple[str, str, str]]:
        p = self.state.player
        taken = p.relic_state.get("rest_taken", ())
        out = []
        if "HEAL" not in taken:
            out.append(("HEAL", _REST_TEXT["HEAL"][0], _REST_TEXT["HEAL"][1].format(heal=self._rest_heal())))
        if "SMITH" not in taken and self._upgradable_deck():
            out.append(("SMITH",) + _REST_TEXT["SMITH"])
        lifts = p.relic_state.get("girya", 0)
        if "girya" in p.relics and lifts < GIRYA_LIFTS and "LIFT" not in taken:
            out.append(("LIFT", _REST_TEXT["LIFT"][0], _REST_TEXT["LIFT"][1].format(left=GIRYA_LIFTS - lifts)))
        if "shovel" in p.relics and "DIG" not in taken:
            out.append(("DIG",) + _REST_TEXT["DIG"])
        return out

    def _finish_rest_option(self, option: str) -> None:
        p = self.state.player
        p.relic_state["rest_taken"] = tuple(p.relic_state.get("rest_taken", ())) + (option,)
        if "miniature_tent" not in p.relics or not self._rest_options():
            self.state.rest_used = True

    def _step_rest(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        if name == actions.PROCEED:
            self._enter_map_screen()
            return
        option = self._rest_options()[int(args[0])][0]
        p = state.player
        assert p is not None
        if option == "HEAL":
            self._heal(self._rest_heal())
            if "tiny_mailbox" in p.relics:
                stream = self.rng.stream("rewards")
                for _ in range(2):
                    if None in p.potions:
                        p.potions[p.potions.index(None)] = random_potion(self._potions, stream)
            # Ancient: stone_humidifier raises Max HP by 5 whenever you Rest.
            if "stone_humidifier" in p.relics:
                self._gain_max_hp(5)
            self._finish_rest_option(option)
        elif option == "SMITH":
            self._open_deck_select("upgrade", self._upgradable_deck(), 1, Screen.REST, cancelable=True)
        elif option == "LIFT":
            p.relic_state["girya"] = p.relic_state.get("girya", 0) + 1
            self._finish_rest_option(option)
        elif option == "DIG":
            self._finish_rest_option(option)
            self._obtain_relic(self._next_relic(self.rng.stream("rewards")), return_to=Screen.REST)

    # ------------------------------------------------------------------
    # Deck card selection (Smith, card removal, relic pickups)

    def _step_deck_select(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        sel = state.deck_select
        assert sel is not None
        if name == actions.SELECT_CARD:
            idx = sel.candidates[int(args[0])]
            if idx in sel.selected:
                sel.selected.remove(idx)
            else:
                sel.selected.append(idx)
            return
        state.deck_select = None
        state.screen = sel.source
        if name == actions.CANCEL_SELECTION:
            return
        deck = state.player.deck
        if sel.purpose == "upgrade":
            for idx in sel.selected:
                deck[idx] = CardRef(self.cards[deck[idx]].upgrade_of or deck[idx], like=deck[idx])
            self._finish_rest_option("SMITH")
        elif sel.purpose == "remove":
            for idx in sorted(sel.selected, reverse=True):
                deck.pop(idx)
            state.player.gold -= sel.price
            state.removals_used += 1
            for item in state.shop:
                if item.category == "card_removal":
                    item.stocked = False
        elif sel.purpose == "duplicate":
            for idx in sel.selected:
                self._add_card_to_deck(CardRef(str(deck[idx]), like=deck[idx]))
        elif sel.purpose == "enchant":
            for idx in sel.selected:
                ref = deck[idx] if isinstance(deck[idx], CardRef) else CardRef(deck[idx])
                ref.enchant, ref.enchant_amount = sel.enchant, sel.enchant_amount
                deck[idx] = ref
        if state.rewards and state.screen != Screen.REWARDS and state.rewards_return is None:
            state.rewards_return = state.screen
            state.screen = Screen.REWARDS

    # ------------------------------------------------------------------
    # Shop

    def _price(self, base: int) -> int:
        p = self.state.player
        if "the_courier" in p.relics:
            base = int(base * 0.8)
        if "membership_card" in p.relics:
            base //= 2
        return base

    def _shop_card(self, ctype: str | None, picked: list[str], stream, *, colorless_rarity: str | None = None
                   ) -> ShopItem:
        state = self.state
        cards = self.cards
        if colorless_rarity is not None:
            cid = rewards.choose_card(self._colorless_pool, cards, colorless_rarity, picked, stream)
            price = rewards.shop_card_price(cards[cid].rarity, colorless=True, rng=stream)
        else:
            typed = [c for c in self._reward_pool if cards[c].card_type == ctype]
            rarity, _ = rewards.roll_rarity(state.card_rarity_offset, rewards.card_odds("shop", state.ascension),
                                            stream, ascension=state.ascension, mutate=False)
            cid = rewards.choose_card(typed, cards, rarity, picked, stream)
            price = rewards.shop_card_price(cards[cid].rarity, colorless=False, rng=stream)
        picked.append(cid)
        return ShopItem(category="card", item=cid, price=price)

    def _shop_relic(self, rarity: str, stream) -> ShopItem:
        relic = self._pull_relic(self.state.relic_bag, rarity, front=False, shop=True) or "circlet"
        self._drop_from_bags(relic)
        base = {"common": 175, "uncommon": 225, "rare": 275, "shop": 200}.get(self._relics[relic].rarity, 1)
        return ShopItem(category="relic", item=relic, price=round(base * stream.uniform(0.85, 1.15)))

    def _shop_potion(self, picked: list[str], stream) -> ShopItem:
        pid = random_potion(self._potions, stream, blacklist=picked)
        picked.append(pid)
        base = {"rare": 100, "uncommon": 75}.get(self._potions[pid].rarity, 50)
        return ShopItem(category="potion", item=pid, price=int(base * stream.uniform(0.95, 1.05) + 0.5))

    def _enter_shop(self) -> None:
        state = self.state
        p = state.player
        if "meal_ticket" in p.relics:
            self._heal(15)
        stream = self.rng.stream("shop")
        sale = stream.randrange(5)
        items: list[ShopItem] = []
        picked: list[str] = []
        for i, ctype in enumerate(("attack", "attack", "skill", "skill", "power")):
            item = self._shop_card(ctype, picked, stream)
            if i == sale:
                item.price //= 2
                item.on_sale = True
            items.append(item)
        for rarity in ("uncommon", "rare"):
            items.append(self._shop_card(None, picked, stream, colorless_rarity=rarity))
        for rarity in (self._roll_relic_rarity(stream), self._roll_relic_rarity(stream), "shop"):
            items.append(self._shop_relic(rarity, stream))
        potions: list[str] = []
        for _ in range(3):
            items.append(self._shop_potion(potions, stream))
        inflation = state.ascension >= ASC_INFLATION
        removal = (100 if inflation else 75) + (50 if inflation else 25) * state.removals_used
        items.append(ShopItem(category="card_removal", item=None, price=removal))
        for item in items:
            item.price = self._price(item.price)
        state.shop = items
        state.screen = Screen.SHOP

    def _step_shop(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        if name == actions.PROCEED:
            state.shop = []
            self._enter_map_screen()
            return
        item = self._shop_items()[int(args[0])]
        p = state.player
        assert p is not None
        if item.category == "card_removal":
            self._open_deck_select("remove", [i for i, c in enumerate(p.deck) if c not in _UNREMOVABLE], 1,
                                   Screen.SHOP, price=item.price, cancelable=True)
            return
        p.gold -= item.price
        item.stocked = False
        if "the_courier" in p.relics:
            self._restock(item)
        if item.category == "card":
            self._add_card_to_deck(item.item)
        elif item.category == "potion":
            p.potions[p.potions.index(None)] = item.item
        elif item.category == "relic":
            self._obtain_relic(item.item, return_to=Screen.SHOP)

    def _restock(self, sold: ShopItem) -> None:
        """The Courier: the merchant never runs out."""

        stream = self.rng.stream("shop")
        if sold.category == "card":
            cdef = self.cards[sold.item]
            new = self._shop_card(None if cdef.color == "colorless" else cdef.card_type, [sold.item], stream,
                                  colorless_rarity=cdef.rarity if cdef.color == "colorless" else None)
        elif sold.category == "relic":
            new = self._shop_relic(self._roll_relic_rarity(stream), stream)
        else:
            new = self._shop_potion([sold.item], stream)
        new.price = self._price(new.price)
        self.state.shop[self.state.shop.index(sold)] = new

    def _shop_items(self) -> list[ShopItem]:
        p = self.state.player
        assert p is not None
        out = []
        for i in self.state.shop:
            if not i.stocked or i.price > p.gold:
                continue
            if i.category == "card_removal" and not any(c not in _UNREMOVABLE for c in p.deck):
                continue
            if i.category == "potion" and None not in p.potions:
                continue
            out.append(i)
        return out

    # ------------------------------------------------------------------
    # Treasure

    def _enter_treasure(self) -> None:
        """EnterTreasureRoom: the chest's gold, paid as it opens, and a relic from the shared bag."""

        state = self.state
        stream = self.rng.stream("treasure")
        self._gain_gold(rewards.poverty_gold(state.ascension, stream.randint(42, 52)))
        relic = self._pull_relic(state.shared_relic_bag, self._roll_relic_rarity(stream))
        state.treasure_relics = [relic or "circlet"]
        state.screen = Screen.TREASURE

    def _step_treasure(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        if name == actions.CLAIM_TREASURE_RELIC:
            relic = state.treasure_relics.pop(int(args[0]))
            state.treasure_relics = []
            self._obtain_relic(relic, return_to=Screen.TREASURE)
            return
        self._enter_map_screen()

    # ------------------------------------------------------------------
    # Decisions

    def _decisions(self) -> list[Decision]:
        state = self.state
        screen = state.screen
        if screen == Screen.NEOW:
            out = [Decision(id=actions.format_action(actions.SELECT_RELIC, i),
                            text=f"take {self._relic_text(r)}")
                   for i, r in enumerate(state.ancient_choices)]
            out.append(Decision(id=actions.NEOW_SKIP, text="skip Neow"))
            return out
        if screen == Screen.ANCIENT:
            out = [Decision(id=actions.format_action(actions.SELECT_RELIC, i),
                            text=f"take {self._relic_text(r)}")
                   for i, r in enumerate(state.ancient_choices)]
            out.append(Decision(id=actions.SKIP_RELIC_SELECTION, text="leave ancient"))
            return out
        if screen == Screen.MAP:
            return self._map_decisions()
        if screen == Screen.COMBAT:
            return self._combat_decisions()
        if screen == Screen.REWARDS:
            return self._reward_decisions()
        if screen == Screen.CARD_REWARD:
            out = [Decision(id=actions.format_action(actions.SELECT_CARD_REWARD, i), text=f"take {self.deck_card_label(c)}")
                   for i, c in enumerate(state.card_reward or [])]
            out.append(Decision(id=actions.SKIP_CARD_REWARD, text="skip card reward"))
            return out
        if screen == Screen.CARD_SELECT:
            return self._deck_select_decisions()
        if screen == Screen.REST:
            out = [] if state.rest_used else [
                Decision(id=actions.format_action(actions.CHOOSE_REST_OPTION, i), text=f"{name}: {desc}")
                for i, (_, name, desc) in enumerate(self._rest_options())]
            if state.rest_used or state.player.relic_state.get("rest_taken"):
                out.append(Decision(id=actions.PROCEED, text="leave rest site"))
            return out
        if screen == Screen.SHOP:
            out = [Decision(id=actions.format_action(actions.SHOP_PURCHASE, i), text=self._shop_item_text(item))
                   for i, item in enumerate(self._shop_items())]
            out.append(Decision(id=actions.PROCEED, text="leave"))
            return out
        if screen == Screen.TREASURE:
            out = [Decision(id=actions.format_action(actions.CLAIM_TREASURE_RELIC, i),
                            text=f"take relic {self._relic_text(r)}") for i, r in enumerate(state.treasure_relics)]
            out.append(Decision(id=actions.PROCEED, text="leave treasure room"))
            return out
        if screen == Screen.EVENT:
            return [Decision(id=actions.format_action(actions.CHOOSE_EVENT_OPTION, 0),
                             text="proceed (events are not simulated yet)")]
        if screen == Screen.GAME_OVER:
            return [Decision(id=actions.GAME_OVER_MAIN_MENU, text="<game_over>")]
        raise RunLoopError(f"no decision table for screen {screen!r}")

    def _relic_text(self, relic_id: str) -> str:
        r = self._relics[relic_id]
        return f"{r.name}: {r.description}"

    def _potion_text(self, potion_id: str) -> str:
        p = self._potions[potion_id]
        return f"{p.name}: {p.description}"

    def _map_decisions(self) -> list[Decision]:
        out = []
        m = self.state.map
        for i, n in enumerate(self._map_options()):
            ahead = ", ".join(c.kind for c in sorted((m.node(x) for x in n.children), key=lambda x: x.col))
            out.append(Decision(id=actions.format_action(actions.CHOOSE_MAP_NODE, i),
                                text=f"go to {n.kind} (row {n.row}, col {n.col}) leads to [{ahead}]"))
        return out

    def _reward_decisions(self) -> list[Decision]:
        p = self.state.player
        assert p is not None
        out = []
        potion_waiting = False
        for i, item in enumerate(self.state.rewards):
            if item.kind == "potion" and None not in p.potions:
                potion_waiting = True
                continue
            out.append(Decision(id=actions.format_action(actions.CLAIM_REWARD, i),
                                text=f"claim {item.kind}: {self._reward_description(item)}"))
        if potion_waiting:
            for slot, pid in enumerate(p.potions):
                if pid is not None:
                    out.append(Decision(id=actions.format_action(actions.DISCARD_POTION, slot),
                                        text=f"discard potion {self._potions[pid].name}"))
        out.append(Decision(id=actions.PROCEED, text="proceed"))
        return out

    def _reward_description(self, item: RewardItem) -> str:
        if item.kind == "gold":
            return f"{item.gold} Gold"
        if item.kind == "card":
            return "Add a card to your deck."
        if item.kind == "potion" and item.potion is not None:
            return self._potion_text(item.potion)
        if item.kind == "relic" and item.relic is not None:
            return self._relic_text(item.relic)
        return ""

    def _deck_select_decisions(self) -> list[Decision]:
        sel = self.state.deck_select
        assert sel is not None
        deck = self.state.player.deck
        out = []
        if len(sel.selected) < sel.count:
            for pos, idx in enumerate(sel.candidates):
                if idx not in sel.selected:
                    out.append(Decision(id=actions.format_action(actions.SELECT_CARD, pos),
                                        text=f"select {self.deck_card_label(deck[idx])}"))
        low = sel.count if sel.min_count is None else sel.min_count
        if low <= len(sel.selected) <= sel.count:
            out.append(Decision(id=actions.CONFIRM_SELECTION, text="confirm selection"))
        if sel.cancelable:
            out.append(Decision(id=actions.CANCEL_SELECTION, text="cancel / skip"))
        return out

    def _shop_item_text(self, item: ShopItem) -> str:
        if item.category == "card":
            card = self.cards[item.item]
            what = f"card {card.name}: {card.description}"
        elif item.category == "relic":
            what = f"relic {self._relic_text(item.item)}"
        elif item.category == "potion":
            what = f"potion {self._potion_text(item.item)}"
        else:
            what = item.category
        return f"buy {what} for {item.price} gold"

    def deck_card_label(self, card_id: str) -> str:
        card = self.cards[card_id]
        up = "+" if card.upgraded else ""
        cost = "X" if card.x_cost else card.cost
        name = card.name[:-1] if card.upgraded and card.name.endswith("+") else card.name
        enchant = getattr(card_id, "enchant", None)
        extra = ""
        if enchant is not None:
            amount = getattr(card_id, "enchant_amount", 0)
            extra = f" [{_ENCHANT_NAMES.get(enchant, enchant)}{(' ' + str(amount)) if amount else ''}]"
        return f"{name}{up}{extra} ({card.card_type}, cost {cost}): {card.description}"

    def card_label(self, card_id: str) -> str:
        card = self.cards[card_id]
        cost = "X" if card.x_cost else self.combat_ctx.effective_cost(card_id)
        return f"{card.name} ({card.card_type}, cost {cost}): {card.description}".rstrip(": ")

    def _generated_label(self, sel, idx: int) -> str:
        card = self.cards[sel.options[idx]]
        desc = card.description
        if sel.purpose == "curse_of_knowledge" and card.card_id == "disintegration":
            combat = self.state.combat
            if combat is not None and combat.curse_choice is not None:
                desc = f"At the end of your turn, take {combat.curse_choice[1]} damage."
        return f"{card.name} ({card.card_type}): {desc}"

    def _combat_decisions(self) -> list[Decision]:
        state = self.state
        assert state.combat is not None
        ctx = self.combat_ctx
        combat = state.combat
        player = state.player
        assert player is not None
        if combat.phase != CombatPhase.PLAYER or combat.outcome is not None:
            raise RunLoopError(
                f"combat_decisions requested in phase {combat.phase!r} (outcome={combat.outcome!r})"
            )

        sel = combat.pending_selection
        if sel is not None:
            return self._selection_decisions(sel)

        decisions: list[Decision] = []
        seen: set[str] = set()
        alive = self._alive_monsters()
        for card_id in player.hand:
            if card_id in seen:
                continue
            seen.add(card_id)
            if not ctx.can_play(card_id):
                continue
            card = self.cards[card_id]
            cost = "X" if card.x_cost else ctx.effective_cost(card_id)
            label = f"play {card.name} (cost {cost}): {card.description}".rstrip(": ")
            if card.target == "single_enemy":
                for pos, monster in enumerate(alive):
                    decisions.append(Decision(
                        id=actions.format_action(actions.PLAY_CARD, card_id, pos),
                        text=f"{label} -> {monster.name}#{pos}[{monster.hp}/{monster.max_hp}]",
                    ))
            else:
                decisions.append(Decision(id=actions.format_action(actions.PLAY_CARD, card_id), text=label))
        for slot, pid in enumerate(player.potions):
            if pid is None or not ctx.can_use_potion(slot):
                continue
            potion = self._potions[pid]
            label = f"use potion {potion.name}: {potion.description}"
            if potion.target == "single_enemy":
                for pos, monster in enumerate(alive):
                    decisions.append(Decision(
                        id=actions.format_action(actions.USE_POTION, slot, pos),
                        text=f"{label} -> {monster.name}#{pos}[{monster.hp}/{monster.max_hp}]",
                    ))
            else:
                decisions.append(Decision(id=actions.format_action(actions.USE_POTION, slot), text=label))
        decisions.append(Decision(id=actions.END_TURN, text="end turn"))
        return decisions

    def _selection_decisions(self, sel) -> list[Decision]:
        ctx = self.combat_ctx
        player = self.state.player
        in_hand = sel.source == "hand"
        pick, confirm = ((actions.COMBAT_SELECT_CARD, actions.COMBAT_CONFIRM_SELECTION) if in_hand
                         else (actions.SELECT_CARD, actions.CONFIRM_SELECTION))
        out = []
        if len(sel.selected) < sel.max_count:
            for pos, idx in enumerate(sel.candidates):
                if idx in sel.selected:
                    continue
                if sel.source == "generated":
                    text = f"choose {self._generated_label(sel, idx)}"
                else:
                    pile = {"hand": player.hand, "discard": player.discard_pile,
                            "draw": player.draw_pile}[sel.source]
                    text = f"select {self.card_label(pile[idx])}"
                out.append(Decision(id=actions.format_action(pick, pos), text=text))
        if ctx.can_confirm_selection() and (sel.selected or sel.min_count > 0 or in_hand):
            out.append(Decision(id=confirm, text="confirm selection"))
        if sel.min_count == 0 and not in_hand:
            out.append(Decision(id=actions.CANCEL_SELECTION, text="skip"))
        return out

    def _alive_monsters(self) -> list:
        state = self.state
        assert state.combat is not None
        return [m for m in state.combat.monsters if m.alive]

    def _packet(self) -> dict[str, Any]:
        state = self.state
        decisions = self._decisions()
        return {
            "screen": state.screen,
            "decision_point": self._decision_point(),
            "step": state.steps,
            "done": state.is_terminal(),
            "outcome": state.outcome,
            "floor": state.floor,
            "act": state.act,
            "candidates": [{"id": d.id, "text": d.text} for d in decisions],
        }

    def _decision_point(self) -> str:
        state = self.state
        if state.screen == Screen.COMBAT and state.combat is not None:
            sel = state.combat.pending_selection
            if sel is not None:
                return DecisionPoint.HAND_SELECT if sel.source == "hand" else DecisionPoint.CARD_SELECT
        return {
            Screen.NEOW: DecisionPoint.NEOW_BONUS,
            Screen.ANCIENT: DecisionPoint.EVENT_CHOICE,
            Screen.MAP: DecisionPoint.MAP_SELECT,
            Screen.COMBAT: DecisionPoint.COMBAT_PLAY,
            Screen.REWARDS: DecisionPoint.REWARDS,
            Screen.CARD_REWARD: DecisionPoint.CARD_REWARD,
            Screen.CARD_SELECT: DecisionPoint.CARD_SELECT,
            Screen.REST: DecisionPoint.REST_SITE,
            Screen.SHOP: DecisionPoint.SHOP,
            Screen.TREASURE: DecisionPoint.TREASURE,
            Screen.EVENT: DecisionPoint.EVENT_CHOICE,
            Screen.BOSS_RELIC: DecisionPoint.BOSS_RELIC,
            Screen.GAME_OVER: DecisionPoint.GAME_OVER,
        }[state.screen]
