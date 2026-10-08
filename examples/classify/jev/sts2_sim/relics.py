"""Relics: the combat-side hooks (RelicEngine) and run-side helpers.

Semantics follow r33hab/sts2 `RelicEffects.cs` (v0.107.1), numbers the
relic text. Counters that the emulator keeps per combat (Pen Nib, Nunchaku,
Happy Flower ...) live on the engine and reset each fight; run-long state
(Lizard Tail used, Girya lifts, Venerable Tea Set armed, Book of Five Rings,
Lasting Candy) lives in PlayerState.relic_state.

Relics without a combat hook (pickup effects, rest sites, shops, rewards,
map) are applied by run.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .combat import RelicHooks
from .schemas import CardSchema, PotionSchema
from .state import CardRef, MonsterState

if TYPE_CHECKING:
    from .combat import CombatContext


STARTER_RELIC = "burning_blood"
# Relics that only ever act outside a fight (or only when picked up).
RUN_ONLY = frozenset({
    "amethyst_aubergine", "black_blood", "book_of_five_rings", "bowler_hat", "burning_blood", "cauldron",
    "dingy_rug", "dollys_mirror", "dragon_fruit", "eternal_feather", "frozen_egg", "girya", "gnarled_hammer",
    "juzu_bracelet", "kifuda", "lasting_candy", "lava_lamp", "lees_waffle", "lucky_fysh", "mango",
    "meal_ticket", "meat_on_the_bone", "membership_card", "miniature_tent", "molten_egg", "old_coin", "orrery",
    "pantograph", "pear", "planisphere", "potion_belt", "prayer_wheel", "punch_dagger", "regal_pillow",
    "royal_stamp", "shovel", "strawberry", "the_courier", "tiny_mailbox", "toxic_egg", "venerable_tea_set",
    "war_paint", "whetstone", "white_beast_statue", "white_star", "wing_charm",
})


class RelicEngine(RelicHooks):
    """The relics the player holds, acting on one combat."""

    def __init__(self, ctx: "CombatContext") -> None:
        self.c = ctx
        self.p = ctx.player
        self.held = set(ctx.player.relics)
        self.run_state = ctx.player.relic_state
        self.n: dict[str, int] = {}  # per-combat counters / once-per-combat flags
        self.pen_nib_armed = False
        self.lamp_card = False
        self.vambrace_card = False
        self.ethereal_exhausts = 0
        self.rainbow = set()

    # -- helpers -------------------------------------------------------------

    def has(self, relic_id: str) -> bool:
        return relic_id in self.held

    def _count(self, relic_id: str, period: int) -> bool:
        """CountTowards: true every `period`-th time."""

        if relic_id not in self.held:
            return False
        seen = (self.n.get(relic_id, 0) + 1) % period
        self.n[relic_id] = seen
        return seen == 0

    def _once(self, relic_id: str) -> bool:
        if relic_id not in self.held or self.n.get(relic_id):
            return False
        self.n[relic_id] = 1
        return True

    def _block(self, amount: int) -> None:
        self.c.gain_block(amount, powered=False)

    # -- queries -------------------------------------------------------------

    def conserves_energy(self) -> bool:
        return self.has("ice_cream")

    def keeps_block(self) -> bool:
        return self.has("sturdy_clamp")

    def skips_first_flush(self) -> bool:
        return self.has("ringing_triangle")

    def makes_ethereal(self, card: CardSchema) -> bool:
        return self.has("ghost_seed") and card.rarity == "basic" and bool({"strike", "defend"} & card.tags)

    def upgrades_played(self, card: CardSchema) -> bool:
        return self.has("razor_tooth") and card.card_type in ("attack", "skill")

    def prevent_death(self) -> bool:
        if self.has("lizard_tail") and not self.run_state.get("lizard_tail_used"):
            self.run_state["lizard_tail_used"] = True
            self.p.hp = max(1, self.p.max_hp // 2)
            return True
        return False

    # -- modifiers -------------------------------------------------------------

    def card_damage_bonus(self, card: CardSchema) -> int:
        if card.card_type != "attack":
            return 0
        bonus = 0
        if card.upgraded and self.has("miniature_cannon"):
            bonus += 3
        if "strike" in card.tags and self.has("strike_dummy"):
            bonus += 3
        return bonus

    def card_damage_multiplier(self, card: CardSchema) -> float:
        return 2.0 if card.card_type == "attack" and self.pen_nib_armed else 1.0

    def enchanted_damage_bonus(self) -> int:
        return 9 if self.has("mystic_lighter") else 0

    def vulnerable_bonus(self) -> float:
        return 0.25 if self.has("paper_phrog") else 0.0

    def turn_draw_extra(self, turn: int) -> int:
        extra = 0
        if turn == 1 and self.has("bag_of_preparation"):
            extra += 2
        if self.n.get("pocketwatch_due"):
            extra += 3
        return extra

    def modify_max_energy(self, energy: int, turn: int) -> int:
        return energy + (1 if turn > 1 and self.has("bread") else 0)

    def modify_hp_loss(self, amount: int) -> int:
        if self.has("tungsten_rod"):
            amount = max(0, amount - 1)
        if self.has("beating_remnant"):
            lost = self.c.combat.hp_lost_this_turn if self.c.combat.phase == "player" else self.n.get("enemy_hp_lost", 0)
            amount = max(0, min(amount, 20 - lost))
            if self.c.combat.phase != "player":
                self.n["enemy_hp_lost"] = self.n.get("enemy_hp_lost", 0) + amount
        return amount

    def modify_card_block(self, amount: int) -> int:
        if self.has("vambrace") and not self.n.get("vambrace") and amount > 0:
            self.vambrace_card = True
            return amount * 2
        return amount

    def modify_strength_gain(self, amount: int) -> int:
        return amount * 2 if self._once("ruined_helmet") else amount

    def modify_enemy_debuff(self, power_id: str, amount: int) -> int:
        if not self.has("unsettling_lamp") or self.n.get("unsettling_lamp") or self.c.playing is None:
            return amount
        self.lamp_card = True
        return amount * 2

    def modify_gold(self, amount: int) -> int:
        return gold_gained(self.p, amount)

    def adjust_x(self, x: int) -> int:
        return x + 2 if self.has("chemical_x") else x

    # -- combat flow -----------------------------------------------------------

    def before_opening_hand(self) -> None:
        if self.has("stone_cracker"):
            pile = self.c.player.draw_pile
            done = 0
            for i in range(len(pile) - 1, -1, -1):
                if done >= 2:
                    break
                if self.c.is_upgradable(pile[i]):
                    self.c.upgrade_in_pile(pile, i)
                    done += 1

    def combat_start(self) -> None:
        c = self.c
        if c.room == "elite" and self.has("sling_of_courage"):
            c.gain_strength(2)
        if self.has("bread"):
            self.p.energy = max(0, self.p.energy - 2)
        lifts = self.run_state.get("girya", 0)
        if lifts > 0:
            c.gain_strength(lifts)
        if self.has("petrified_toad"):
            c.procure_potion("potion_shaped_rock")
        if self.has("blood_vial"):
            c.heal_player(2)
        if self.has("anchor"):
            self._block(10)
        if self.has("vajra"):
            c.gain_strength(1)
        if self.has("oddly_smooth_stone"):
            c.apply_power(self.p, "dexterity", 1)
        if self.has("gorget"):
            c.apply_power(self.p, "plating", 4)
        if self.has("bronze_scales"):
            c.apply_power(self.p, "thorns", 3)
        self.potions_changed()
        self.hp_changed()

    def turn_start(self, turn: int, *, attacks_last_turn: int, cards_last_turn: int) -> None:
        c = self.c
        for rid in ("kunai", "kusarigama", "shuriken", "ornamental_fan", "letter_opener", "demon_tongue"):
            self.n.pop(rid, None)
        self.rainbow = set()
        self.n.pop("rainbow_ring_paid", None)
        self.n.pop("enemy_hp_lost", None)
        if turn == 1:
            if self.has("lantern"):
                c.gain_energy(1)
            if self.has("venerable_tea_set") and self.run_state.get("venerable_tea_set"):
                self.run_state["venerable_tea_set"] = 0
                c.gain_energy(2)
            if self.has("akabeko"):
                c.apply_power(self.p, "vigor", 8)
            for m in c.alive_monsters():
                if self.has("bag_of_marbles"):
                    c.apply_power(m, "vulnerable", 1, source=self.p)
                if self.has("red_mask"):
                    c.apply_power(m, "weak", 1, source=self.p)
            if self.has("festive_popper"):
                c.damage_all_unpowered(9)
        if self.has("brimstone"):
            c.gain_strength(2)
            for m in c.alive_monsters():
                c._change_power(m, "strength", 1, allow_negative=True)
        if turn == 2 and self.has("horn_cleat"):
            self._block(14)
        if turn == 3 and self.has("captains_wheel"):
            self._block(18)
        if self._count("happy_flower", 3):
            c.gain_energy(1)
        if self._count("pendulum", 3):
            c.draw(1)
        if turn > 1 and self.has("art_of_war") and attacks_last_turn == 0:
            c.gain_energy(1)
        self.n["pocketwatch_due"] = 1 if self.has("pocketwatch") and turn > 1 and cards_last_turn <= 3 else 0
        if self.has("mercury_hourglass"):
            c.damage_all_unpowered(3)
        if turn == 2 and self.has("candelabra"):
            c.gain_energy(2)
        if turn == 3 and self.has("chandelier"):
            c.gain_energy(3)

    def after_block_cleared(self, turn: int) -> None:
        if turn == 3 and self.has("sparkling_rouge"):
            self.c.gain_strength(1)
            self.c.apply_power(self.p, "dexterity", 1)

    def after_hand_drawn(self, turn: int) -> None:
        if turn != 1:
            return
        c = self.c
        if self.has("bellows"):
            for i in range(len(self.p.hand)):
                c.upgrade_in_pile(self.p.hand, i)
        if self.has("vexing_puzzlebox"):
            pool = c.generation_pool()
            if pool:
                c.add_to_hand(c.rng.stream("card_generation").choice(pool)).free_turn = True
        if self.has("toolbox"):
            c.queue_followup(self._toolbox)
        if self.has("gambling_chip"):
            c.queue_followup(self._gambling_chip)

    def _toolbox(self):
        c = self.c
        options = c.distinct_cards(c.generation_pool(color="colorless"), 3)
        return _toolbox_pick(c, options)

    def _gambling_chip(self):
        from .potion_effects import discard_any_then_draw

        return discard_any_then_draw(self.c) if self.p.hand else None

    def before_card_played(self, card: CardSchema, ref: Any, energy_spent: int) -> None:
        if card.card_type == "attack" and self.has("pen_nib"):
            self.pen_nib_armed = self._count("pen_nib", 10)
        else:
            self.pen_nib_armed = False
        if energy_spent >= 2 and self.has("intimidating_helmet"):
            self._block(4)

    def after_card_resolved(self, card: CardSchema) -> None:
        if card.card_type == "power" and self.has("game_piece"):
            self.c.draw(1)

    def after_card_played(self, card: CardSchema, ref: Any, energy_spent: int) -> None:
        c = self.c
        self.pen_nib_armed = False
        if self.lamp_card:
            self.lamp_card = False
            self.n["unsettling_lamp"] = 1
        if self.vambrace_card:
            self.vambrace_card = False
            self.n["vambrace"] = 1
        if card.card_type == "attack":
            if self._count("shuriken", 3):
                c.gain_strength(1)
            if self._count("kunai", 3):
                c.apply_power(self.p, "dexterity", 1)
            if self._count("ornamental_fan", 3):
                self._block(4)
            if self._count("kusarigama", 3):
                t = c.random_monster()
                if t is not None:
                    c.damage_monster_unpowered(t, 6)
            if self._count("nunchaku", 10):
                c.gain_energy(1)
        elif card.card_type == "skill":
            if self._count("letter_opener", 3):
                c.damage_all_unpowered(5)
            if self._count("tuning_fork", 10):
                self._block(7)
        elif card.card_type == "power":
            if self._once("permafrost"):
                self._block(7)
            if self.has("mummified_hand"):
                self._mummified_hand()
        if self.has("rainbow_ring") and not self.n.get("rainbow_ring_paid"):
            self.rainbow.add(card.card_type)
            if {"attack", "skill", "power"} <= self.rainbow:
                self.n["rainbow_ring_paid"] = 1
                c.gain_strength(1)
                c.apply_power(self.p, "dexterity", 1)

    def _mummified_hand(self) -> None:
        c = self.c
        hand = self.p.hand
        cands = [i for i, cid in enumerate(hand) if not c.cards[cid].x_cost and c.effective_cost(cid) > 0]
        if not cands:
            return
        i = c.rng.stream("card_select").choice(cands)
        if not isinstance(hand[i], CardRef):
            hand[i] = CardRef(hand[i])
        hand[i].free_turn = True

    def after_exhaust(self, ref: Any, *, ethereal: bool) -> None:
        c = self.c
        if self.has("charons_ashes"):
            c.damage_all_unpowered(3)
        if self.has("joss_paper"):
            if ethereal:
                self.ethereal_exhausts += 1
            else:
                self._joss(1)
        if self.has("burning_sticks") and not self.n.get("burning_sticks") and c.cards[ref].card_type == "skill":
            self.n["burning_sticks"] = 1
            if len(self.p.hand) < 10:
                self.p.hand.append(CardRef(str(ref), like=ref if isinstance(ref, CardRef) else None))

    def _joss(self, count: int) -> None:
        total = self.n.get("joss_paper", 0) + count
        self.n["joss_paper"] = total % 5
        if total // 5:
            self.c.draw(total // 5)

    def after_shuffle(self) -> None:
        if self.has("the_abacus"):
            self._block(6)

    def after_hp_lost(self, amount: int, *, attack: bool, player_turn: bool) -> None:
        c = self.c
        if self.has("self_forming_clay"):
            c.apply_power(self.p, "block_next_turn", 3)
        if self._once("centennial_puzzle"):
            c.draw(3)
        if player_turn and self.has("demon_tongue") and not self.n.get("demon_tongue"):
            self.n["demon_tongue"] = 1
            c.heal_player(amount)
        self.hp_changed()

    def hp_changed(self) -> None:
        if not self.has("red_skull"):
            return
        active = self.p.hp <= self.p.max_hp // 2
        if active == bool(self.n.get("red_skull_active")):
            return
        self.n["red_skull_active"] = 1 if active else 0
        self.c._change_power(self.p, "strength", 3 if active else -3, allow_negative=True)

    def potions_changed(self) -> None:
        if not self.has("belt_buckle"):
            return
        should = not any(self.p.potions)
        if should == bool(self.n.get("belt_buckle_on")):
            return
        self.n["belt_buckle_on"] = 1 if should else 0
        self.c._change_power(self.p, "dexterity", 2 if should else -2, allow_negative=True)

    def after_potion_used(self, potion: PotionSchema) -> None:
        if self.has("reptile_trinket"):
            self.c.gain_temporary_strength(3)
        self.potions_changed()

    def monster_died(self, monster: MonsterState) -> None:
        if self.has("gremlin_horn") and self.c.combat.outcome is None:
            self.c.gain_energy(1)
            self.c.draw(1)

    def hand_emptied(self) -> None:
        if self.has("unceasing_top") and not self.p.hand:
            self.c.draw(1)

    def turn_end(self, turn: int, *, attacks: int) -> None:
        c = self.c
        no_block = self.c._s.extra.get("block_before_end", self.p.block) == 0
        if self.has("orichalcum") and no_block:
            self._block(6)
        if self.has("cloak_clasp") and self.p.hand:
            self._block(len(self.p.hand))
        if self.has("screaming_flagon") and not self.p.hand:
            c.damage_all_unpowered(20)
        if self.has("stone_calendar") and turn == 7:
            c.damage_all_unpowered(52)
        if self.has("parrying_shield") and self.p.block >= 10:
            t = c.random_monster()
            if t is not None:
                c.damage_monster_unpowered(t, 6)
        if self.has("ripple_basin") and attacks == 0:
            self._block(4)

    def after_flush(self) -> None:
        if self.ethereal_exhausts and self.has("joss_paper"):
            n, self.ethereal_exhausts = self.ethereal_exhausts, 0
            self._joss(n)


def _toolbox_pick(c, options):
    if not options:
        return
    from .combat import SelectionRequest

    chosen = yield SelectionRequest(source="generated", candidates=list(range(len(options))), count=1,
                                    purpose="to_hand", options=options)
    for cid in chosen:
        c.add_to_hand(cid)


# ---------------------------------------------------------------------------
# Run-side helpers


def gold_gained(player: Any, amount: int) -> int:
    """Bowler Hat (+25%) and Dragon Fruit (+1 Max HP per gain)."""

    if amount <= 0:
        return amount
    if "bowler_hat" in player.relics:
        amount = int(amount * 1.25)
    if "dragon_fruit" in player.relics:
        player.max_hp += 1
        player.hp += 1
    return amount


__all__ = ["RUN_ONLY", "RelicEngine", "STARTER_RELIC", "gold_gained"]
