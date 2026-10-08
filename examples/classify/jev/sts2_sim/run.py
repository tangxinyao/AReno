"""RunLoop: a whole Ironclad run -- Neow, three acts of map, rooms, rewards.

Run structure follows r33hab/sts2's `RunEngine` / `RunMapGenerator` /
`RunRewardGenerator` (v0.107.1):

  * Act 1 is Overgrowth or Underdocks, then the Hive, then Glory. Every act's
    encounters are rolled up front: its weak fights (3 in Act 1, 2 later),
    then normal fights, fifteen elites and a boss, never repeating an
    encounter tag back to back. Ascension 10 adds a second, different boss
    to the last act.
  * Each act has a 7-column map (mapgen.py). Monster rooms take the next
    normal encounter, Elites the next elite; Unknown rooms roll Monster 10% /
    Treasure 2% / Shop 3% / Event otherwise, the odds of the types not
    rolled growing by their base each time.
  * Winning a fight opens the rewards screen (gold, maybe a potion, an elite
    relic, a card reward -- see rewards.py). Rest sites heal 30% of max HP or
    upgrade a card. Beating a boss moves to the next act, whose Ancient heals
    the player first (80% of missing HP from Ascension 2).

`encounter=` pins a single fight instead (Neow -> that combat -> game over),
the scope the earlier one-combat episodes used.

Action ids use STS2MCP action names (see `actions.py`):
  neow / ancient:  "choose_event_option:{i}"
  map:             "choose_map_node:{i}"          (travelable nodes by column)
  combat:          "play_card:{card_id}[:{enemy_pos}]", "end_turn"
  hand_select:     "combat_select_card:{i}" / "combat_confirm_selection"
  card_select:     "select_card:{i}" / "confirm_selection" / "cancel_selection"
  rewards:         "claim_reward:{i}" / "proceed"
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
from .loader import load_all, load_encounters
from .rng import Rng
from .schemas import CardSchema, EncounterSchema, MonsterSchema, PowerSchema
from .state import (
    ActRooms,
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

IRONCLAD_STARTING_DECK: tuple[str, ...] = (
    "strike_ironclad", "strike_ironclad", "strike_ironclad", "strike_ironclad", "strike_ironclad",
    "defend_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad",
    "bash",
)
IRONCLAD_STARTING_RELIC = "burning_blood"
ACT1_VARIANTS: tuple[str, ...] = ("overgrowth", "underdocks")
LATER_ACTS: tuple[str, ...] = ("hive", "glory")
FIRST_COMBAT_POOL = _WEAK_POOL

# Unknown map point: (room, base odds). Elite is never rolled (-1) without a relic.
_UNKNOWN_BASE = {"monster": 0.1, "elite": -1.0, "treasure": 0.02, "shop": 0.03}
_ROOM_OF_KIND = {
    mapgen.MONSTER: "monster", mapgen.ELITE: "elite", mapgen.BOSS: "boss", mapgen.REST: "rest",
    mapgen.SHOP: "shop", mapgen.TREASURE: "treasure", mapgen.UNKNOWN: "unknown", mapgen.ANCIENT: "ancient",
}
_REST_HEAL = ("HEAL", "Rest", "Heal for 30% of your Max HP ({heal}).")
_REST_SMITH = ("SMITH", "Smith", "Upgrade a card in your Deck.")


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
        self._reward_pool: list[str] = []

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

    def reset(self) -> dict[str, Any]:
        self._rng = Rng(self._master_seed)
        self._effects = EffectQueue()
        self._hooks = HookBus()
        if self._cards is None:
            self._powers, self._cards, self._monsters = load_all()
            self._encounters = load_encounters(monster_ids=set(self._monsters))
            self._reward_pool = rewards.reward_pool(self._cards)
        asc = self._ascension
        hp = _IRONCLAD_MAX_HP if asc < ASC_WEARY_TRAVELER else int(_IRONCLAD_MAX_HP * 0.8)
        deck = list(IRONCLAD_STARTING_DECK)
        if asc >= ASC_ASCENDERS_BANE:
            deck.append("ascenders_bane")
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
        )
        if not self.single_combat:
            self._generate_acts()
            self._enter_act_map(0)
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
            self._leave_neow()
        elif screen == Screen.ANCIENT:
            self._enter_map_screen()
        elif screen == Screen.MAP:
            self._choose_map_node(int(args[0]))
        elif screen == Screen.COMBAT:
            self._handle_combat_action(action_id)
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
        node = self._map_options()[index]
        state.map_coord = node.coord
        state.floor += 1
        room = _ROOM_OF_KIND[node.kind]
        if room == "unknown":
            room = self._roll_unknown()
        state.room = room
        if room == "ancient":
            self._ancient_heal()
            state.screen = Screen.ANCIENT
            return
        if room in ("monster", "elite", "boss"):
            enc_id = self._next_encounter(room, node)
            self._start_encounter(enc_id, room)
        elif room == "rest":
            state.rest_used = False
            state.screen = Screen.REST
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
        self.combat_ctx.start_combat(monster_ids, list(p.deck), monster_flags=flags)
        self._maybe_finalize_combat()

    def _handle_combat_action(self, action_id: str) -> None:
        ctx = self.combat_ctx
        try:
            name, args = actions.parse_action(action_id)
            sel = ctx.combat.pending_selection
            if name == actions.END_TURN and not args:
                ctx.end_turn()
            elif name == actions.PLAY_CARD and len(args) == 1:
                ctx.play_card(args[0])
            elif name == actions.PLAY_CARD and len(args) == 2:
                ctx.play_card(args[0], target_slot=self._alive_monsters()[int(args[1])].slot)
            elif name in (actions.COMBAT_SELECT_CARD, actions.SELECT_CARD) and sel is not None:
                ctx.toggle_selection(sel.candidates[int(args[0])])
            elif name in (actions.COMBAT_CONFIRM_SELECTION, actions.CONFIRM_SELECTION):
                ctx.confirm_selection()
            else:
                raise RunLoopError(f"unknown combat action {action_id!r}")
        except CombatError as exc:
            raise RunLoopError(f"combat engine rejected {action_id!r}: {exc}") from exc

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
        if "burning_blood" in p.relics:
            p.hp = min(p.max_hp, p.hp + 6)
        room = state.room
        if room == "boss" and state.act_index >= len(state.acts) - 1:
            if not self._boss_remaining():
                self._game_over(Outcome.VICTORY)
                return
            # A10's first boss of the last act: straight on to the second.
            self._enter_map_screen()
            return
        self._open_combat_rewards(room or "monster")

    def _boss_remaining(self) -> bool:
        state = self.state
        node = state.map.node(state.map_coord)
        return any(state.map.node(c).kind == mapgen.BOSS for c in node.children)

    def _game_over(self, outcome: str) -> None:
        self.state.screen = Screen.GAME_OVER
        self.state.outcome = outcome

    # ------------------------------------------------------------------
    # Rewards

    def _open_combat_rewards(self, room: str) -> None:
        state = self.state
        stream = self.rng.stream("rewards")
        items: list[RewardItem] = [RewardItem(kind="gold", gold=rewards.combat_gold(room, state.ascension, stream))]
        dropped, state.potion_odds = rewards.roll_potion_drop(state.potion_odds, room, stream)
        if dropped:
            potion = self._roll_potion(stream)
            if potion is not None:
                items.append(RewardItem(kind="potion", potion=potion))
        if room == "elite":
            relic = self._roll_relic(stream)
            if relic is not None:
                items.append(RewardItem(kind="relic", relic=relic))
        cards, state.card_rarity_offset = rewards.card_reward(
            self.cards, self._reward_pool, room, ascension=state.ascension, act_index=state.act_index,
            offset=state.card_rarity_offset, rng=stream)
        items.append(RewardItem(kind="card", cards=cards))
        state.rewards = items
        state.screen = Screen.REWARDS

    def _roll_potion(self, stream) -> str | None:
        del stream
        return None  # potions arrive with potions.py

    def _roll_relic(self, stream) -> str | None:
        del stream
        return None  # relic rewards arrive with relics.py

    def _step_rewards(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        if name == actions.PROCEED:
            self._leave_rewards()
            return
        index = int(args[0])
        item = state.rewards[index]
        p = state.player
        assert p is not None
        if item.kind == "gold":
            p.gold += item.gold
        elif item.kind == "potion":
            slot = p.potions.index(None)
            p.potions[slot] = item.potion
        elif item.kind == "relic":
            p.relics.append(item.relic)
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
            state.player.deck.append(state.card_reward[int(args[0])])
        state.rewards.pop(state.card_reward_item)
        state.card_reward = None
        state.card_reward_item = None
        state.screen = Screen.REWARDS

    def _leave_rewards(self) -> None:
        state = self.state
        state.rewards = []
        if state.room == "boss":
            self._enter_act_map(state.act_index + 1)
        self._enter_map_screen()

    # ------------------------------------------------------------------
    # Rest site

    def _rest_heal(self) -> int:
        return max(1, int(self.state.player.max_hp * 0.3))

    def _upgradable_deck(self) -> list[int]:
        cards = self.cards
        return [i for i, c in enumerate(self.state.player.deck)
                if cards[c].upgrade_of is not None and cards[c].card_type not in ("status", "curse")]

    def _rest_options(self) -> list[tuple[str, str, str, bool]]:
        heal = self._rest_heal()
        return [
            (_REST_HEAL[0], _REST_HEAL[1], _REST_HEAL[2].format(heal=heal), True),
            (_REST_SMITH[0], _REST_SMITH[1], _REST_SMITH[2], bool(self._upgradable_deck())),
        ]

    def _step_rest(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        if name == actions.PROCEED:
            self._enter_map_screen()
            return
        enabled = [o for o in self._rest_options() if o[3]]
        option = enabled[int(args[0])][0]
        p = state.player
        assert p is not None
        if option == "HEAL":
            p.hp = min(p.max_hp, p.hp + self._rest_heal())
            state.rest_used = True
        elif option == "SMITH":
            state.deck_select = DeckSelection(purpose="upgrade", candidates=self._upgradable_deck(), count=1,
                                              source=Screen.REST)
            state.screen = Screen.CARD_SELECT

    # ------------------------------------------------------------------
    # Deck card selection (Smith, card removal)

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
                deck[idx] = self.cards[deck[idx]].upgrade_of or deck[idx]
            state.rest_used = True
        elif sel.purpose == "remove":
            for idx in sorted(sel.selected, reverse=True):
                deck.pop(idx)
            state.player.gold -= sel.price
            state.removals_used += 1
            for item in state.shop:
                if item.category == "card_removal":
                    item.stocked = False

    # ------------------------------------------------------------------
    # Shop

    def _enter_shop(self) -> None:
        state = self.state
        stream = self.rng.stream("shop")
        cards = self.cards
        pool = self._reward_pool
        sale = stream.randrange(5)
        items: list[ShopItem] = []
        picked: list[str] = []
        for i, ctype in enumerate(("attack", "attack", "skill", "skill", "power")):
            typed = [c for c in pool if cards[c].card_type == ctype]
            rarity, _ = rewards.roll_rarity(state.card_rarity_offset, rewards.card_odds("shop", state.ascension),
                                            stream, ascension=state.ascension, mutate=False)
            cid = rewards.choose_card(typed, cards, rarity, picked, stream)
            picked.append(cid)
            price = rewards.shop_card_price(cards[cid].rarity, colorless=False, rng=stream)
            if i == sale:
                price //= 2
            items.append(ShopItem(category="card", item=cid, price=price, on_sale=i == sale))
        inflation = state.ascension >= ASC_INFLATION
        removal = (100 if inflation else 75) + (50 if inflation else 25) * state.removals_used
        items.append(ShopItem(category="card_removal", item=None, price=removal))
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
            state.deck_select = DeckSelection(
                purpose="remove", candidates=[i for i, c in enumerate(p.deck) if self._removable(c)],
                count=1, source=Screen.SHOP, price=item.price)
            state.screen = Screen.CARD_SELECT
            return
        p.gold -= item.price
        item.stocked = False
        if item.category == "card":
            p.deck.append(item.item)

    def _removable(self, card_id: str) -> bool:
        return card_id not in ("ascenders_bane", "curse_of_the_bell", "necronomicurse")

    def _shop_items(self) -> list[ShopItem]:
        p = self.state.player
        assert p is not None
        return [i for i in self.state.shop if i.stocked and i.price <= p.gold
                and (i.category != "card_removal" or any(self._removable(c) for c in p.deck))]

    # ------------------------------------------------------------------
    # Treasure

    def _enter_treasure(self) -> None:
        state = self.state
        stream = self.rng.stream("treasure")
        state.player.gold += rewards.poverty_gold(state.ascension, stream.randint(42, 52))
        state.treasure_relics = []
        state.screen = Screen.TREASURE

    def _step_treasure(self, name: str, args: tuple[str, ...]) -> None:
        state = self.state
        if name == actions.CLAIM_TREASURE_RELIC:
            state.player.relics.append(state.treasure_relics.pop(int(args[0])))
            state.treasure_relics = []
            return
        self._enter_map_screen()

    # ------------------------------------------------------------------
    # Decisions

    def _decisions(self) -> list[Decision]:
        state = self.state
        screen = state.screen
        if screen == Screen.NEOW:
            return [Decision(id=actions.NEOW_SKIP, text="skip Neow (stub)")]
        if screen == Screen.ANCIENT:
            return [Decision(id=actions.format_action(actions.CHOOSE_EVENT_OPTION, 0), text="proceed")]
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
            if state.rest_used:
                return [Decision(id=actions.PROCEED, text="leave rest site")]
            return [Decision(id=actions.format_action(actions.CHOOSE_REST_OPTION, i), text=f"{name}: {desc}")
                    for i, (_, name, desc, on) in enumerate(o for o in self._rest_options() if o[3])]
        if screen == Screen.SHOP:
            out = [Decision(id=actions.format_action(actions.SHOP_PURCHASE, i), text=self._shop_item_text(item))
                   for i, item in enumerate(self._shop_items())]
            out.append(Decision(id=actions.PROCEED, text="leave"))
            return out
        if screen == Screen.TREASURE:
            return [Decision(id=actions.PROCEED, text="leave treasure room")]
        if screen == Screen.EVENT:
            return [Decision(id=actions.format_action(actions.CHOOSE_EVENT_OPTION, 0), text="proceed (events are not simulated yet)")]
        if screen == Screen.GAME_OVER:
            return [Decision(id=actions.GAME_OVER_MAIN_MENU, text="<game_over>")]
        raise RunLoopError(f"no decision table for screen {screen!r}")

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
        for i, item in enumerate(self.state.rewards):
            if item.kind == "potion" and None not in p.potions:
                continue
            out.append(Decision(id=actions.format_action(actions.CLAIM_REWARD, i),
                                text=f"claim {item.kind}: {self._reward_description(item)}"))
        out.append(Decision(id=actions.PROCEED, text="proceed"))
        return out

    def _reward_description(self, item: RewardItem) -> str:
        if item.kind == "gold":
            return f"{item.gold} Gold"
        if item.kind == "card":
            return "Add a card to your deck."
        return item.potion or item.relic or ""

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
        if len(sel.selected) == sel.count:
            out.append(Decision(id=actions.CONFIRM_SELECTION, text="confirm selection"))
        if sel.cancelable:
            out.append(Decision(id=actions.CANCEL_SELECTION, text="cancel / skip"))
        return out

    def _shop_item_text(self, item: ShopItem) -> str:
        if item.category == "card":
            card = self.cards[item.item]
            what = f"card {card.name}: {card.description}"
        else:
            what = item.category
        return f"buy {what} for {item.price} gold"

    def deck_card_label(self, card_id: str) -> str:
        card = self.cards[card_id]
        up = "+" if card.upgraded else ""
        cost = "X" if card.x_cost else card.cost
        name = card.name[:-1] if card.upgraded and card.name.endswith("+") else card.name
        return f"{name}{up} ({card.card_type}, cost {cost}): {card.description}"

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
        return f"{card.name}: {desc}"

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
        if sel is not None and sel.source == "generated":
            out = []
            if len(sel.selected) < sel.max_count:
                for pos, idx in enumerate(sel.candidates):
                    if idx not in sel.selected:
                        out.append(Decision(id=actions.format_action(actions.SELECT_CARD, pos),
                                            text=f"choose {self._generated_label(sel, idx)}"))
            if ctx.can_confirm_selection():
                out.append(Decision(id=actions.CONFIRM_SELECTION, text="confirm selection"))
            return out
        if sel is not None:
            in_hand = sel.source == "hand"
            pile = player.hand if in_hand else player.discard_pile
            pick, confirm = (
                (actions.COMBAT_SELECT_CARD, actions.COMBAT_CONFIRM_SELECTION) if in_hand
                else (actions.SELECT_CARD, actions.CONFIRM_SELECTION)
            )
            out = []
            if len(sel.selected) < sel.max_count:
                for pos, idx in enumerate(sel.candidates):
                    if idx in sel.selected:
                        continue
                    out.append(Decision(id=actions.format_action(pick, pos),
                                        text=f"select {self.card_label(pile[idx])}"))
            if ctx.can_confirm_selection():
                out.append(Decision(id=confirm, text="confirm selection"))
            return out

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
        decisions.append(Decision(id=actions.END_TURN, text="end turn"))
        return decisions

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
