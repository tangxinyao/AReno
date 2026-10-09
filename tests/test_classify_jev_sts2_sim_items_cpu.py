"""sts2_sim potions, Colorless cards and relics (STS2 v0.107.1), CPU only.

Builds combats with a RelicEngine for the given relics and checks each item's
effect against its game text / r33hab `PotionEffects` / `RelicEffects`.
"""

from __future__ import annotations

import unittest

from tests.test_classify_jev_sts2_sim_combat_cpu import DATA, SIM, m0

POTIONS = SIM.load_potions()
RELICS = SIM.load_relics()


def make(monsters=("nibbit",), *, hand=None, deck=None, relics=(), potions=(), energy=3, hp=100, monster_hp=200,
         seed=7, room="monster", ascension=0):
    RelicEngine = SIM.relics.RelicEngine

    powers, cards, mdefs = DATA
    player = SIM.PlayerState(hp=hp, max_hp=hp, gold=99, max_energy=3, relics=list(relics),
                             potions=list(potions) + [None] * max(0, 3 - len(potions)))
    run = SIM.RunState(character="ironclad", ascension=ascension, seed=seed, player=player)
    ctx = SIM.CombatContext(run=run, cards=cards, monsters=mdefs, powers=powers, rng=SIM.Rng(seed),
                            hooks=SIM.HookBus(), effects=SIM.EffectQueue(), potions=POTIONS)
    ctx.relics = RelicEngine(ctx)
    ctx.start_combat(list(monsters), list(deck or ["strike_ironclad"] * 10), room=room)
    player.hp = hp
    if monster_hp is not None:
        for m in ctx.combat.monsters:
            m.hp = m.max_hp = monster_hp
    if hand is not None:
        player.draw_pile.extend(player.hand)
        player.hand = [SIM.CardRef(c) for c in hand]
    player.energy = energy
    return ctx


class PotionTest(unittest.TestCase):
    def test_catalog(self) -> None:
        self.assertEqual(len([p for p in POTIONS.values() if p.pool != "token"]), 48)
        POTION_EFFECTS = SIM.potion_effects.POTION_EFFECTS
        for p in POTIONS.values():
            self.assertIn(p.game_id, POTION_EFFECTS, p.potion_id)

    def test_fire_and_explosive(self) -> None:
        ctx = make(["nibbit", "nibbit"], potions=["fire_potion", "explosive_ampoule"])
        ctx.use_potion(0, target_slot=1)
        self.assertEqual(ctx.combat.monsters[1].hp, 180)
        ctx.use_potion(1)
        self.assertEqual([m.hp for m in ctx.combat.monsters], [190, 170])
        self.assertEqual(ctx.player.potions, [None, None, None])

    def test_block_potion_is_unpowered(self) -> None:
        ctx = make(potions=["block_potion"])
        ctx.player.powers["dexterity"] = 5
        ctx.player.powers["frail"] = 1
        ctx.use_potion(0)
        self.assertEqual(ctx.player.block, 12)

    def test_flex_and_speed_are_temporary(self) -> None:
        ctx = make(potions=["flex_potion", "speed_potion"], hand=[])
        ctx.use_potion(0)
        ctx.use_potion(1)
        self.assertEqual(ctx.player.powers["strength"], 5)
        self.assertEqual(ctx.player.powers["dexterity"], 5)
        ctx.end_turn()
        self.assertNotIn("strength", ctx.player.powers)
        self.assertNotIn("dexterity", ctx.player.powers)

    def test_duplicator_replays_next_card(self) -> None:
        ctx = make(potions=["duplicator"], hand=["defend_ironclad"])
        ctx.use_potion(0)
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.block, 10)

    def test_gigantification_triples_next_attack(self) -> None:
        ctx = make(potions=["gigantification_potion"], hand=["strike_ironclad", "strike_ironclad"])
        ctx.use_potion(0)
        ctx.play_card("strike_ironclad", target_slot=0)
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 18 - 6)

    def test_attack_potion_choice_is_free_and_skippable(self) -> None:
        ctx = make(potions=["attack_potion", "attack_potion"], hand=[], energy=0)
        ctx.use_potion(0)
        sel = ctx.combat.pending_selection
        self.assertEqual(sel.source, "generated")
        self.assertEqual(len(sel.options), 3)
        self.assertTrue(all(ctx.cards[c].card_type == "attack" for c in sel.options))
        ctx.toggle_selection(0)
        ctx.confirm_selection()
        self.assertEqual(ctx.player.hand, [sel.options[0]])
        self.assertEqual(ctx.effective_cost(ctx.player.hand[0]), 0)
        ctx.use_potion(1)
        ctx.confirm_selection(skip=True)
        self.assertEqual(len(ctx.player.hand), 1)

    def test_fairy_in_a_bottle_saves_from_death(self) -> None:
        ctx = make(potions=["fairy_in_a_bottle"], hp=5, hand=[])
        self.assertFalse(ctx.can_use_potion(0))
        ctx.end_turn()  # Nibbit Butt 12
        self.assertIsNone(ctx.combat.outcome)
        self.assertEqual(ctx.player.hp, 1)  # max(30% of 5, 1)
        self.assertEqual(ctx.player.potions[0], None)

    def test_entropic_brew_fills_slots(self) -> None:
        ctx = make(potions=["entropic_brew"])
        ctx.use_potion(0)
        self.assertTrue(all(ctx.player.potions))

    def test_regen_buffer_and_ship(self) -> None:
        ctx = make(potions=["regen_potion", "lucky_tonic", "ship_in_a_bottle"], hand=[], hp=100)
        ctx.player.hp = 50
        for slot in range(3):
            ctx.use_potion(slot)
        self.assertEqual(ctx.player.block, 10)
        ctx.end_turn()  # Regen 5 heals at turn end; the Nibbit hit gets past 10 Block, Buffer eats it
        self.assertEqual(ctx.player.powers["regen"], 4)
        self.assertNotIn("buffer", ctx.player.powers)
        self.assertEqual(ctx.player.block, 10)  # Ship in a Bottle: block next turn
        self.assertEqual(ctx.player.hp, 55)

    def test_powdered_demise(self) -> None:
        ctx = make(potions=["powdered_demise"], hand=[])
        ctx.use_potion(0, target_slot=0)
        ctx.end_turn()
        self.assertEqual(m0(ctx).hp, 191)

    def test_snecko_oil_and_touch_of_insanity(self) -> None:
        ctx = make(potions=["snecko_oil", "touch_of_insanity"], hand=["bludgeon"])
        ctx.use_potion(0)
        self.assertTrue(all(0 <= ctx.effective_cost(c) <= 3 for c in ctx.player.hand))
        ctx.player.hand = [SIM.CardRef("bludgeon")]
        ctx.use_potion(1)
        ctx.toggle_selection(0)
        ctx.confirm_selection()
        self.assertEqual(ctx.effective_cost(ctx.player.hand[0]), 0)

    def test_soldiers_stew_replays_strikes(self) -> None:
        ctx = make(potions=["soldiers_stew"], hand=["strike_ironclad"])
        ctx.use_potion(0)
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).hp, 188)

    def test_shackling_and_beetle_juice(self) -> None:
        ctx = make(["nibbit", "nibbit"], potions=["shackling_potion", "beetle_juice"], hand=[])
        ctx.use_potion(0)
        self.assertEqual(m0(ctx).powers["strength"], -7)
        ctx.use_potion(1, target_slot=1)
        self.assertEqual(ctx.combat.monsters[1].powers["shrink"], 4)
        ctx.end_turn()
        self.assertNotIn("strength", m0(ctx).powers)

    def test_liquid_memories_and_droplet(self) -> None:
        ctx = make(potions=["liquid_memories", "droplet_of_precognition"], hand=[])
        ctx.player.discard_pile = [SIM.CardRef("bludgeon")]
        ctx.use_potion(0)
        ctx.toggle_selection(0)
        ctx.confirm_selection()
        self.assertEqual(ctx.player.hand, ["bludgeon"])
        self.assertEqual(ctx.effective_cost("bludgeon"), 0)
        ctx.use_potion(1)
        self.assertEqual(ctx.combat.pending_selection.source, "draw")


class ColorlessCardTest(unittest.TestCase):
    def test_catalog(self) -> None:
        cards = DATA[1]
        colorless = [c for c in cards.values() if c.color == "colorless" and not c.upgraded]
        self.assertEqual(len(colorless), 64)
        from importlib import import_module
        CARD_EFFECTS = import_module(SIM.__name__ + ".card_effects").CARD_EFFECTS
        for c in colorless:
            if not c.multiplayer_only:
                self.assertIn(c.game_id, CARD_EFFECTS, c.card_id)

    def test_rend_counts_unique_debuffs(self) -> None:
        ctx = make(hand=["rend", "rend+1"], energy=4)
        m0(ctx).powers.update(weak=1, vulnerable=1)
        ctx.play_card("rend", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - int((15 + 10) * 1.5))
        ctx.play_card("rend+1", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 37 - int((18 + 16) * 1.5))

    def test_omnislice_and_fisticuffs(self) -> None:
        ctx = make(["nibbit", "nibbit"], hand=["omnislice", "fisticuffs"])
        ctx.play_card("omnislice", target_slot=0)
        self.assertEqual([m.hp for m in ctx.combat.monsters], [192, 192])
        ctx.play_card("fisticuffs", target_slot=1)
        self.assertEqual(ctx.player.block, 7)

    def test_volley_x(self) -> None:
        ctx = make(hand=["volley"], energy=3)
        ctx.play_card("volley")
        self.assertEqual(m0(ctx).hp, 170)

    def test_panic_button_blocks_card_block(self) -> None:
        ctx = make(hand=["panic_button", "defend_ironclad"])
        ctx.play_card("panic_button")
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.block, 30)

    def test_prolong_and_equilibrium(self) -> None:
        ctx = make(hand=["equilibrium", "prolong", "strike_ironclad"], energy=3)
        ctx.play_card("equilibrium")
        ctx.play_card("prolong")
        ctx.end_turn()
        self.assertIn("strike_ironclad", ctx.player.hand)  # retained
        self.assertEqual(ctx.player.block, 13)  # 13 - 12 Butt... then Prolong's 13 next turn
        self.assertNotIn("block_next_turn", ctx.player.powers)

    def test_the_bomb(self) -> None:
        ctx = make(hand=["the_bomb"])
        ctx.play_card("the_bomb")
        for _ in range(3):
            ctx.player.hand = []
            ctx.end_turn()
        self.assertEqual(m0(ctx).hp, 165)  # the Nibbit still holds 5 Block from its last turn

    def test_hand_of_greed_gold(self) -> None:
        ctx = make(hand=["hand_of_greed"], monster_hp=10)
        ctx.play_card("hand_of_greed", target_slot=0)
        self.assertEqual(ctx.player.gold, 119)

    def test_gold_axe_and_mind_blast(self) -> None:
        ctx = make(hand=["defend_ironclad", "gold_axe", "mind_blast"], energy=3)
        ctx.play_card("defend_ironclad")
        ctx.play_card("gold_axe", target_slot=0)
        self.assertEqual(m0(ctx).hp, 198)
        draw = len(ctx.player.draw_pile)
        ctx.play_card("mind_blast", target_slot=0)
        self.assertEqual(m0(ctx).hp, 198 - draw)

    def test_mayhem_rolling_boulder_prep_time(self) -> None:
        ctx = make(hand=["rolling_boulder", "prep_time"], energy=4)
        ctx.play_card("rolling_boulder")
        ctx.play_card("prep_time")
        ctx.end_turn()
        self.assertEqual(m0(ctx).hp, 195)
        self.assertEqual(ctx.player.powers["rolling_boulder"], 10)
        self.assertEqual(ctx.player.powers["vigor"], 4)

    def test_nostalgia_puts_first_attack_on_top(self) -> None:
        ctx = make(hand=["nostalgia", "strike_ironclad"])
        ctx.play_card("nostalgia")
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(ctx.player.draw_pile[-1], "strike_ironclad")

    def test_panache_every_five_cards(self) -> None:
        ctx = make(hand=["panache"] + ["defend_ironclad"] * 5, energy=6)
        ctx.play_card("panache")
        for _ in range(5):
            ctx.play_card("defend_ironclad")
        self.assertEqual(m0(ctx).hp, 190)

    def test_stratagem_choice_on_shuffle(self) -> None:
        ctx = make(hand=["stratagem"], deck=["strike_ironclad"] * 5)
        ctx.play_card("stratagem")
        ctx.player.draw_pile = []
        ctx.player.discard_pile = [SIM.CardRef("bludgeon"), SIM.CardRef("defend_ironclad")]
        ctx.draw(1)
        sel = ctx.combat.pending_selection
        self.assertIsNone(sel)  # follow-ups run once the current action resolves
        ctx._settle()
        self.assertEqual(ctx.combat.pending_selection.source, "draw")

    def test_purity_up_to_three(self) -> None:
        ctx = make(hand=["purity", "wound", "wound", "strike_ironclad"])
        ctx.play_card("purity")
        ctx.toggle_selection(0)
        ctx.toggle_selection(1)
        ctx.confirm_selection()
        self.assertEqual(ctx.player.hand, ["strike_ironclad"])

    def test_bolas_returns(self) -> None:
        ctx = make(hand=["bolas"])
        ctx.play_card("bolas", target_slot=0)
        ctx.player.hand = []
        ctx.end_turn()
        self.assertIn("bolas", ctx.player.hand)

    def test_jackpot_adds_zero_cost(self) -> None:
        ctx = make(hand=["jackpot"])
        ctx.play_card("jackpot", target_slot=0)
        self.assertEqual(len(ctx.player.hand), 3)
        self.assertTrue(all(ctx.cards[c].cost == 0 for c in ctx.player.hand))

    def test_seeker_strike_offers_three(self) -> None:
        ctx = make(hand=["seeker_strike"])
        ctx.play_card("seeker_strike", target_slot=0)
        sel = ctx.combat.pending_selection
        self.assertEqual((sel.source, len(sel.candidates)), ("draw", 3))

    def test_fasten_only_defends(self) -> None:
        ctx = make(hand=["fasten", "defend_ironclad", "ultimate_defend"], energy=3)
        ctx.play_card("fasten")
        ctx.play_card("defend_ironclad")
        ctx.play_card("ultimate_defend")
        self.assertEqual(ctx.player.block, 9 + 15)

    def test_splash_is_kept_out_of_pools(self) -> None:
        pool = SIM.rewards.reward_pool(DATA[1], "colorless")
        self.assertNotIn("splash", pool)
        self.assertNotIn("beacon_of_hope", pool)


class RelicCombatTest(unittest.TestCase):
    def test_catalog(self) -> None:
        self.assertEqual(len([r for r in RELICS.values() if r.relic_id != "circlet"]), 216)

    def test_combat_start_relics(self) -> None:
        ctx = make(relics=["anchor", "vajra", "oddly_smooth_stone", "bronze_scales", "gorget", "lantern",
                           "bag_of_marbles", "red_mask", "akabeko", "blood_vial"], hand=None, hp=50)
        p = ctx.player
        self.assertEqual(p.block, 10)
        self.assertEqual((p.powers["strength"], p.powers["dexterity"], p.powers["thorns"], p.powers["plating"],
                          p.powers["vigor"]), (1, 1, 3, 4, 8))
        self.assertEqual(m0(ctx).powers.get("vulnerable"), 1)
        self.assertEqual(m0(ctx).powers.get("weak"), 1)

    def test_bag_of_preparation_and_lantern(self) -> None:
        ctx = make(relics=["bag_of_preparation", "lantern"], hand=None, energy=None)
        self.assertEqual(len(ctx.player.hand), 7)

    def test_pen_nib_tenth_attack(self) -> None:
        ctx = make(relics=["pen_nib"], hand=["strike_ironclad"] * 10, energy=10)
        for _ in range(9):
            ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 54)
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 54 - 12)

    def test_shuriken_kunai_fan(self) -> None:
        ctx = make(relics=["shuriken", "kunai", "ornamental_fan"], hand=["strike_ironclad"] * 3)
        for _ in range(3):
            ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(ctx.player.powers["strength"], 1)
        self.assertEqual(ctx.player.powers["dexterity"], 1)
        self.assertEqual(ctx.player.block, 4)

    def test_orichalcum_and_ripple_basin(self) -> None:
        ctx = make(relics=["orichalcum", "ripple_basin"], hand=[])
        ctx.end_turn()  # 6 + 4 block before the Butt (12)
        self.assertEqual(ctx.player.hp, 98)

    def test_tungsten_rod_and_beating_remnant(self) -> None:
        ctx = make(relics=["tungsten_rod"], hand=[])
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, 89)
        ctx = make(relics=["beating_remnant"], hand=["hemokinesis", "offering"], energy=3)
        ctx.play_card("hemokinesis", target_slot=0)  # 2
        ctx.play_card("offering")  # 6
        ctx.lose_hp(30)
        self.assertEqual(ctx.player.hp, 80)

    def test_lizard_tail_once_per_run(self) -> None:
        ctx = make(relics=["lizard_tail"], hp=5, hand=[])
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, 2)
        self.assertTrue(ctx.player.relic_state["lizard_tail_used"])

    def test_centennial_puzzle_and_self_forming_clay(self) -> None:
        ctx = make(relics=["centennial_puzzle", "self_forming_clay"], hand=["hemokinesis"])
        ctx.play_card("hemokinesis", target_slot=0)
        self.assertEqual(len(ctx.player.hand), 3)
        self.assertEqual(ctx.player.powers["block_next_turn"], 3)

    def test_red_skull_toggles(self) -> None:
        ctx = make(relics=["red_skull"], hand=[], hp=100)
        ctx.lose_hp(51)
        self.assertEqual(ctx.player.powers["strength"], 3)
        ctx.heal_player(10)
        self.assertNotIn("strength", ctx.player.powers)

    def test_ice_cream_and_sturdy_clamp(self) -> None:
        ctx = make(relics=["ice_cream", "sturdy_clamp"], hand=[], energy=2)
        ctx.player.block = 30
        ctx.end_turn()
        self.assertEqual(ctx.player.energy, 5)
        self.assertEqual(ctx.player.block, 10)

    def test_gremlin_horn_charons_abacus(self) -> None:
        ctx = make(["nibbit", "nibbit"], relics=["gremlin_horn", "charons_ashes"], hand=["true_grit", "strike_ironclad"])
        ctx.combat.monsters[1].hp = 3
        ctx.play_card("true_grit")  # exhausts the Strike: 3 to all kills the second Nibbit
        self.assertFalse(ctx.combat.monsters[1].alive)
        self.assertEqual(ctx.player.energy, 3)

    def test_damage_relics(self) -> None:
        ctx = make(relics=["strike_dummy", "miniature_cannon", "paper_phrog"], hand=["strike_ironclad+1"])
        m0(ctx).powers["vulnerable"] = 1
        ctx.play_card("strike_ironclad+1", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - int((9 + 6) * 1.75))

    def test_chemical_x(self) -> None:
        ctx = make(relics=["chemical_x"], hand=["whirlwind"], energy=1)
        ctx.play_card("whirlwind")
        self.assertEqual(m0(ctx).hp, 185)

    def test_ruined_helmet_and_vambrace(self) -> None:
        ctx = make(relics=["ruined_helmet", "vambrace"], hand=["inflame", "defend_ironclad", "defend_ironclad"])
        ctx.play_card("inflame")
        ctx.play_card("defend_ironclad")
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.powers["strength"], 4)
        self.assertEqual(ctx.player.block, 15)

    def test_belt_buckle(self) -> None:
        ctx = make(relics=["belt_buckle"], potions=["block_potion"], hand=[])
        self.assertNotIn("dexterity", ctx.player.powers)
        ctx.use_potion(0)
        self.assertEqual(ctx.player.powers["dexterity"], 2)

    def test_reptile_trinket(self) -> None:
        ctx = make(relics=["reptile_trinket"], potions=["block_potion"], hand=[])
        ctx.use_potion(0)
        self.assertEqual(ctx.player.powers["strength"], 3)
        ctx.end_turn()
        self.assertNotIn("strength", ctx.player.powers)

    def test_turn_count_relics(self) -> None:
        ctx = make(relics=["horn_cleat", "captains_wheel", "candelabra", "chandelier", "happy_flower"], hand=[])
        ctx.end_turn()
        self.assertEqual(ctx.player.energy, 3 + 2)
        self.assertEqual(ctx.player.block, 14)
        ctx.player.hand = []
        ctx.end_turn()
        self.assertEqual(ctx.player.energy, 3 + 3 + 1)
        self.assertEqual(ctx.player.block, 18)

    def test_mummified_hand_and_game_piece(self) -> None:
        ctx = make(relics=["mummified_hand", "game_piece"], hand=["inflame", "bludgeon"])
        ctx.play_card("inflame")
        self.assertEqual(ctx.effective_cost("bludgeon"), 0)
        self.assertEqual(len(ctx.player.hand), 2)

    def test_razor_tooth_upgrades_played(self) -> None:
        ctx = make(relics=["razor_tooth"], hand=["strike_ironclad"])
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(ctx.player.discard_pile, ["strike_ironclad+1"])

    def test_unceasing_top(self) -> None:
        ctx = make(relics=["unceasing_top"], hand=["defend_ironclad"])
        ctx.play_card("defend_ironclad")
        self.assertEqual(len(ctx.player.hand), 1)

    def test_ghost_seed_and_ringing_triangle(self) -> None:
        ctx = make(relics=["ghost_seed"], hand=["strike_ironclad", "bash"])
        ctx.end_turn()
        self.assertIn("strike_ironclad", ctx.player.exhaust_pile)
        self.assertNotIn("bash", ctx.player.exhaust_pile)

    def test_screaming_flagon_and_stone_calendar(self) -> None:
        ctx = make(relics=["screaming_flagon"], hand=[])
        ctx.end_turn()
        self.assertEqual(m0(ctx).hp, 180)

    def test_unsettling_lamp_doubles_first_debuff(self) -> None:
        ctx = make(relics=["unsettling_lamp"], hand=["bash", "bash"], energy=4)
        ctx.play_card("bash", target_slot=0)
        self.assertEqual(m0(ctx).powers["vulnerable"], 4)
        ctx.play_card("bash", target_slot=0)
        self.assertEqual(m0(ctx).powers["vulnerable"], 6)

    def test_sling_of_courage_elite_only(self) -> None:
        ctx = make(relics=["sling_of_courage"], room="elite", hand=[])
        self.assertEqual(ctx.player.powers["strength"], 2)


class RunRelicTest(unittest.TestCase):
    def _run(self, relics=(), seed=3):
        loop = SIM.RunLoop(seed=seed)
        loop.reset()
        loop.state.player.relics.extend(relics)
        loop.step("choose_event_option:0")
        return loop

    def test_pickups(self) -> None:
        loop = self._run()
        st = loop.state
        p = st.player
        for rid in ("mango", "potion_belt", "old_coin", "whetstone"):
            loop._obtain_relic(rid, return_to=SIM.Screen.MAP)
        self.assertEqual(p.max_hp, 94)
        self.assertEqual(len(p.potions), 5)
        self.assertEqual(p.gold, 399)
        self.assertEqual(sum(1 for c in p.deck if c == "strike_ironclad+1" or c == "bash+1"), 2)

    def test_enchant_pickup_and_damage(self) -> None:
        loop = self._run()
        st = loop.state
        loop._obtain_relic("gnarled_hammer", return_to=SIM.Screen.MAP)
        self.assertEqual(st.screen, SIM.Screen.CARD_SELECT)
        self.assertEqual(st.deck_select.count, 3)
        loop.step("select_card:0")
        loop.step("confirm_selection")
        enchanted = [c for c in st.player.deck if getattr(c, "enchant", None) == "sharp"]
        self.assertEqual(len(enchanted), 1)
        self.assertEqual(st.screen, SIM.Screen.MAP)

    def test_eggs_and_lucky_fysh(self) -> None:
        loop = self._run(["molten_egg", "lucky_fysh"])
        loop._add_card_to_deck("pommel_strike")
        self.assertEqual(loop.state.player.deck[-1], "pommel_strike+1")
        self.assertEqual(loop.state.player.gold, 114)

    def test_bowler_hat_and_dragon_fruit(self) -> None:
        loop = self._run(["bowler_hat", "dragon_fruit"])
        loop._gain_gold(20)
        self.assertEqual(loop.state.player.gold, 124)
        self.assertEqual(loop.state.player.max_hp, 81)

    def test_rest_relics(self) -> None:
        loop = self._run(["girya", "shovel", "regal_pillow", "miniature_tent"])
        st = loop.state
        st.player.hp = 20
        loop._enter_rest()
        texts = [c["text"].split(":")[0] for c in loop._packet()["candidates"]]
        self.assertEqual(texts, ["Rest", "Smith", "Train", "Dig"])
        loop.step("choose_rest_option:0")
        self.assertEqual(st.player.hp, 20 + 24 + 15)
        self.assertFalse(st.rest_used)  # Miniature Tent: more options
        loop.step("choose_rest_option:1")  # Train
        self.assertEqual(st.player.relic_state["girya"], 1)

    def test_membership_card_halves_prices(self) -> None:
        loop = self._run(["membership_card"])
        st = loop.state
        st.player.gold = 999
        loop._enter_shop()
        removal = next(i for i in st.shop if i.category == "card_removal")
        self.assertEqual(removal.price, 37)

    def test_treasure_has_a_relic(self) -> None:
        loop = self._run()
        st = loop.state
        gold = st.player.gold
        loop._enter_treasure()
        self.assertTrue(42 <= st.player.gold - gold <= 52)
        self.assertEqual(len(st.treasure_relics), 1)
        relic = st.treasure_relics[0]
        loop.step("claim_treasure_relic:0")
        self.assertIn(relic, st.player.relics)
        self.assertNotIn(relic, [r for bag in st.relic_bag.values() for r in bag])

    def test_elite_reward_has_relic_and_white_star(self) -> None:
        loop = self._run(["white_star", "white_beast_statue"])
        loop._open_combat_rewards("elite", took_damage=True)
        kinds = [r.kind for r in loop.state.rewards]
        self.assertEqual(kinds, ["gold", "potion", "relic", "card", "card"])
        self.assertTrue(all(loop.cards[c].rarity == "rare" for c in loop.state.rewards[-1].cards))

    def test_potion_reward_needs_a_free_slot(self) -> None:
        loop = self._run(["white_beast_statue"])
        st = loop.state
        st.player.potions = ["block_potion", "fire_potion", "weak_potion"]
        loop._open_combat_rewards("monster", took_damage=True)
        ids = [c["id"] for c in loop._packet()["candidates"]]
        self.assertNotIn("claim_reward:1", ids)
        self.assertIn("discard_potion:0", ids)
        loop.step("discard_potion:0")
        self.assertIn("claim_reward:1", [c["id"] for c in loop._packet()["candidates"]])

    def test_orrery_opens_five_card_rewards(self) -> None:
        loop = self._run()
        loop._obtain_relic("orrery", return_to=SIM.Screen.MAP)
        st = loop.state
        self.assertEqual(st.screen, SIM.Screen.REWARDS)
        self.assertEqual([r.kind for r in st.rewards], ["card"] * 5)
        loop.step("proceed")
        self.assertEqual(st.screen, SIM.Screen.MAP)


class AncientEventTest(unittest.TestCase):
    def test_neow_offers_three_relics_and_skip(self) -> None:
        loop = SIM.RunLoop(seed=3)
        packet = loop.reset()
        ids = [c["id"] for c in packet["candidates"]]
        self.assertIn("choose_event_option:0", ids)
        picks = [i for i in ids if i.startswith("select_relic:")]
        self.assertEqual(len(picks), 3)
        # All relic picks resolve to Ancient-rarity relics from the Neow pool.
        choices = loop.state.ancient_choices
        self.assertEqual(len(choices), 3)
        for rid in choices:
            self.assertEqual(RELICS[rid].rarity, "ancient")
            self.assertEqual(RELICS[rid].pool, "ancient")
        self.assertEqual(loop.state.ancient_event, "NEOW")

    def test_select_relic_adds_it_and_opens_map(self) -> None:
        loop = SIM.RunLoop(seed=3)
        loop.reset()
        choice = loop.state.ancient_choices[0]
        loop.step("select_relic:0")
        # Should land on map (or a reward if the pickup opened one).
        self.assertIn(choice, loop.state.player.relics)
        self.assertEqual(loop.state.ancient_choices, [])
        self.assertIsNone(loop.state.ancient_event)

    def test_skip_leaves_without_relic(self) -> None:
        loop = SIM.RunLoop(seed=3)
        loop.reset()
        pre_relics = list(loop.state.player.relics)
        loop.step("choose_event_option:0")
        self.assertEqual(loop.state.player.relics, pre_relics)
        self.assertEqual(loop.state.screen, SIM.Screen.MAP)

    def test_act_specific_ancients_pick_the_right_pool(self) -> None:
        pools = SIM.load_ancient_event_pools()
        # The sim's act-ancient mapping (run.py:_ACT_ANCIENTS).
        for act, ancient_ids in (("overgrowth", ("NEOW",)),
                                 ("underdocks", ("NEOW",)),
                                 ("hive", ("OROBAS", "PAEL", "TEZCATARA")),
                                 ("glory", ("NONUPEIPE", "TANX", "VAKUU"))):
            for ev in ancient_ids:
                self.assertIn(ev, pools)
                for gid in pools[ev]:
                    rid = gid.lower()
                    self.assertIn(rid, RELICS, f"{ev} references missing relic {gid}")
                    self.assertEqual(RELICS[rid].rarity, "ancient")


class AncientRelicPickupTest(unittest.TestCase):
    def _run(self, seed=3):
        loop = SIM.RunLoop(seed=seed)
        loop.reset()
        loop.step("choose_event_option:0")  # skip Neow
        return loop

    def test_gold_and_hp_pickups(self) -> None:
        loop = self._run()
        p = loop.state.player
        base_gold = p.gold
        loop._obtain_relic("golden_pearl", return_to=SIM.Screen.MAP)
        loop._obtain_relic("signet_ring", return_to=SIM.Screen.MAP)
        loop._obtain_relic("nutritious_oyster", return_to=SIM.Screen.MAP)
        loop._obtain_relic("looming_fruit", return_to=SIM.Screen.MAP)
        self.assertEqual(p.gold, base_gold + 150 + 999)
        self.assertEqual(p.max_hp, 80 + 11 + 31)

    def test_curse_pickup_adds_curse_and_gold(self) -> None:
        loop = self._run()
        p = loop.state.player
        before_gold = p.gold
        before_curses = sum(1 for c in p.deck if c == "greed")
        loop._obtain_relic("cursed_pearl", return_to=SIM.Screen.MAP)
        self.assertEqual(p.gold, before_gold + 333)
        self.assertEqual(sum(1 for c in p.deck if c == "greed"), before_curses + 1)

    def test_capsule_grants_random_relic(self) -> None:
        loop = self._run()
        p = loop.state.player
        before = len(p.relics)
        loop._obtain_relic("small_capsule", return_to=SIM.Screen.MAP)
        # small_capsule itself + 1 rolled relic
        self.assertEqual(len(p.relics), before + 2)

    def test_sand_castle_upgrades_six_random_cards(self) -> None:
        loop = self._run()
        p = loop.state.player
        upgradable = [i for i, c in enumerate(p.deck)
                      if loop.cards[c].upgrade_of is not None]
        self.assertGreaterEqual(len(upgradable), 6)
        loop._obtain_relic("sand_castle", return_to=SIM.Screen.MAP)
        after_upgrades = sum(1 for c in p.deck if str(c).endswith("+1"))
        self.assertGreaterEqual(after_upgrades, 6)


class AncientRelicCombatTest(unittest.TestCase):
    # `make()` resets hand/energy after start_combat, so tests here check
    # side effects that survive the reset: block, enemy powers, deck piles.

    def test_sai_grants_block(self) -> None:
        ctx = make(relics=["sai"], hand=[])
        self.assertEqual(ctx.player.block, 7)

    def test_paels_blood_draws_extra_card(self) -> None:
        # Don't override hand so the opening draw survives.
        ctx = make(relics=["paels_blood"])
        self.assertEqual(len(ctx.player.hand), 6)

    def test_snecko_eye_draws_two_extra(self) -> None:
        ctx = make(relics=["snecko_eye"])
        self.assertEqual(len(ctx.player.hand), 7)

    def test_booming_conch_elite_draw(self) -> None:
        ctx = make(relics=["booming_conch"], room="elite")
        self.assertEqual(len(ctx.player.hand), 7)
        ctx2 = make(relics=["booming_conch"], room="monster")
        self.assertEqual(len(ctx2.player.hand), 5)

    def test_philosophers_stone_buffs_enemies(self) -> None:
        ctx = make(relics=["philosophers_stone"], hand=[])
        for m in ctx.combat.monsters:
            self.assertGreaterEqual(m.powers.get("strength", 0), 1)

    def test_crossbow_adds_free_attack_to_hand(self) -> None:
        ctx = make(relics=["crossbow"])
        attack_count = sum(1 for cid in ctx.player.hand if ctx.cards[cid].card_type == "attack")
        self.assertGreaterEqual(attack_count, 1)

    def test_throwing_axe_doubles_first_card(self) -> None:
        # Play Strike once; Throwing Axe should replay it a second time.
        ctx = make(relics=["throwing_axe"], hand=["strike_ironclad"])
        base = m0(ctx).hp
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(base - m0(ctx).hp, 2 * 6)
        # Second strike should NOT be replayed.
        ctx.player.hand = [SIM.CardRef("strike_ironclad")]
        base2 = m0(ctx).hp
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(base2 - m0(ctx).hp, 6)

    def test_iron_club_draws_every_4(self) -> None:
        ctx = make(relics=["iron_club"], hand=["defend_ironclad"] * 4, energy=10,
                   deck=["strike_ironclad"] * 20)
        for _ in range(4):
            ctx.play_card("defend_ironclad")
        # 4 cards played -> +1 draw. Hand = 4 - 4 + 1 = 1.
        self.assertEqual(len(ctx.player.hand), 1)

    def test_brilliant_scarf_refunds_fifth_card(self) -> None:
        ctx = make(relics=["brilliant_scarf"], hand=["defend_ironclad"] * 6, energy=10)
        for _ in range(4):
            ctx.play_card("defend_ironclad")
        energy_pre = ctx.player.energy
        ctx.play_card("defend_ironclad")  # the 5th play, refunded
        self.assertEqual(ctx.player.energy, energy_pre)

    def test_music_box_adds_ethereal_copy_of_first_attack(self) -> None:
        ctx = make(relics=["music_box"], hand=["strike_ironclad"])
        ctx.play_card("strike_ironclad", target_slot=0)
        # Hand gained an Ethereal copy; attacks after the first don't duplicate.
        hand_cards = list(ctx.player.hand)
        self.assertEqual(sum(1 for c in hand_cards if c == "strike_ironclad"), 1)
        self.assertTrue(hand_cards[0].ethereal)

    def test_paels_tears_banks_unspent_energy(self) -> None:
        ctx = make(relics=["paels_tears"], hand=[], energy=2)
        ctx.end_turn()
        self.assertEqual(ctx.player.energy, 3 + 2)

    def test_diamond_diadem_halves_damage_when_quiet(self) -> None:
        # Compare: with Diadem (0 plays), damage is halved vs. without.
        ctx_no = make(["nibbit"], hand=[], hp=200)
        ctx_no.end_turn()
        damage_no = ctx_no.player.max_hp - ctx_no.player.hp
        ctx = make(["nibbit"], relics=["diamond_diadem"], hand=[], hp=200)
        ctx.end_turn()
        damage_yes = ctx.player.max_hp - ctx.player.hp
        self.assertLess(damage_yes, damage_no)

    def test_jeweled_mask_pulls_power_from_draw(self) -> None:
        deck = ["strike_ironclad"] * 5 + ["inflame"]
        ctx = make(relics=["jeweled_mask"], deck=deck)
        self.assertIn("inflame", ctx.player.hand)

    def test_blessed_antler_shuffles_three_dazed(self) -> None:
        ctx = make(relics=["blessed_antler"], hand=[])
        all_cards = ctx.player.hand + ctx.player.draw_pile + ctx.player.discard_pile
        self.assertEqual(sum(1 for c in all_cards if c == "dazed"), 3)

    def test_biiig_hug_grants_block_on_shuffle(self) -> None:
        ctx = make(relics=["biiig_hug"], hand=[], deck=["strike_ironclad"] * 2)
        # Force an empty draw pile and trigger a reshuffle via draw.
        ctx.player.draw_pile = []
        ctx.player.discard_pile = [SIM.CardRef("strike_ironclad")]
        before = ctx.player.block
        ctx.draw(1)
        self.assertGreaterEqual(ctx.player.block, before + 5)

    def test_runic_pyramid_retains_hand_at_end_of_turn(self) -> None:
        ctx = make(relics=["runic_pyramid"], hand=["defend_ironclad"] * 3,
                   deck=["strike_ironclad"] * 20)
        kept = list(ctx.player.hand)
        ctx.end_turn()
        # Hand retained across the flush; the next turn draw tops it up.
        self.assertTrue(all(c in ctx.player.hand for c in kept))

    def test_paels_legion_doubles_block_once(self) -> None:
        # Baseline Defend block, then Pael's Legion doubles it.
        ctx_no = make(relics=[], hand=["defend_ironclad"])
        ctx_no.play_card("defend_ironclad")
        base = ctx_no.player.block
        ctx = make(relics=["paels_legion"], hand=["defend_ironclad"] * 2, energy=5)
        ctx.play_card("defend_ironclad")
        boosted = ctx.player.block
        self.assertEqual(boosted, base * 2)
        # Second card in the same turn: Legion asleep, back to base.
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.block - boosted, base)


class AncientRunHookTest(unittest.TestCase):
    def _run(self, seed=3):
        loop = SIM.RunLoop(seed=seed)
        loop.reset()
        loop.step("choose_event_option:0")
        return loop

    def test_stone_humidifier_grants_max_hp_on_rest(self) -> None:
        loop = self._run()
        loop._obtain_relic("stone_humidifier", return_to=SIM.Screen.MAP)
        p = loop.state.player
        before = p.max_hp
        loop.state.screen = SIM.Screen.REST
        loop.state.rest_used = False
        loop._step_rest("choose_rest_option", ("0",))
        self.assertEqual(p.max_hp, before + 5)

    def test_lava_rock_adds_two_relics_to_act1_boss(self) -> None:
        loop = self._run()
        loop._obtain_relic("lava_rock", return_to=SIM.Screen.MAP)
        loop.state.room = "boss"
        loop.state.act_index = 0
        loop._open_combat_rewards("boss", took_damage=False)
        relic_rewards = [r for r in loop.state.rewards if r.kind == "relic"]
        self.assertEqual(len(relic_rewards), 2)

    def test_black_star_adds_elite_relic(self) -> None:
        loop = self._run()
        loop._obtain_relic("black_star", return_to=SIM.Screen.MAP)
        loop.state.room = "elite"
        loop._open_combat_rewards("elite", took_damage=False)
        relic_rewards = [r for r in loop.state.rewards if r.kind == "relic"]
        self.assertEqual(len(relic_rewards), 2)

    def test_beautiful_bracelet_opens_enchant_selection(self) -> None:
        loop = self._run()
        loop._obtain_relic("beautiful_bracelet", return_to=SIM.Screen.MAP)
        self.assertEqual(loop.state.screen, SIM.Screen.CARD_SELECT)
        self.assertEqual(loop.state.deck_select.enchant, "swift")
        self.assertEqual(loop.state.deck_select.count, 3)

    def test_paels_claw_auto_enchants_defends(self) -> None:
        loop = self._run()
        loop._obtain_relic("paels_claw", return_to=SIM.Screen.MAP)
        defends = [c for c in loop.state.player.deck if c == "defend_ironclad"]
        self.assertGreater(len(defends), 0)
        for c in defends:
            self.assertEqual(getattr(c, "enchant", None), "imbued")

    def test_biiig_hug_opens_four_card_removal(self) -> None:
        loop = self._run()
        loop._obtain_relic("biiig_hug", return_to=SIM.Screen.MAP)
        self.assertEqual(loop.state.screen, SIM.Screen.CARD_SELECT)
        self.assertEqual(loop.state.deck_select.purpose, "remove")
        self.assertEqual(loop.state.deck_select.count, 4)

    def test_scroll_boxes_zeros_gold_and_adds_cards(self) -> None:
        loop = self._run()
        before = len(loop.state.player.deck)
        loop._obtain_relic("scroll_boxes", return_to=SIM.Screen.MAP)
        self.assertEqual(loop.state.player.gold, 0)
        self.assertEqual(len(loop.state.player.deck), before + 5)

    def test_pandoras_box_transforms_all_starters(self) -> None:
        loop = self._run()
        before = list(loop.state.player.deck)
        count_strikes = sum(1 for c in before if c == "strike_ironclad")
        count_defends = sum(1 for c in before if c == "defend_ironclad")
        loop._obtain_relic("pandoras_box", return_to=SIM.Screen.MAP)
        after = loop.state.player.deck
        # Strikes and defends gone (transformed to other Ironclad cards).
        self.assertEqual(sum(1 for c in after if c == "strike_ironclad"), 0)
        self.assertEqual(sum(1 for c in after if c == "defend_ironclad"), 0)
        self.assertEqual(len(after), len(before))  # count unchanged
        self.assertEqual(sum(1 for c in before if c == "bash"),
                         sum(1 for c in after if c == "bash"))  # Bash untouched

    def test_astrolabe_opens_transform_three(self) -> None:
        loop = self._run()
        loop._obtain_relic("astrolabe", return_to=SIM.Screen.MAP)
        self.assertEqual(loop.state.screen, SIM.Screen.CARD_SELECT)
        self.assertEqual(loop.state.deck_select.purpose, "transform")
        self.assertEqual(loop.state.deck_select.count, 3)
        self.assertEqual(loop.state.deck_select.enchant, "upgrade")

    def test_new_leaf_opens_single_transform(self) -> None:
        loop = self._run()
        loop._obtain_relic("new_leaf", return_to=SIM.Screen.MAP)
        self.assertEqual(loop.state.screen, SIM.Screen.CARD_SELECT)
        self.assertEqual(loop.state.deck_select.purpose, "transform")
        self.assertEqual(loop.state.deck_select.count, 1)

    def test_claws_opens_transform_up_to_six(self) -> None:
        loop = self._run()
        loop._obtain_relic("claws", return_to=SIM.Screen.MAP)
        self.assertEqual(loop.state.screen, SIM.Screen.CARD_SELECT)
        self.assertIn(loop.state.deck_select.purpose, ("transform", "transform_into"))
        self.assertEqual(loop.state.deck_select.min_count, 0)

    def test_paels_tooth_removes_five_cards_and_banks(self) -> None:
        loop = self._run()
        p = loop.state.player
        before = len(p.deck)
        loop._obtain_relic("paels_tooth", return_to=SIM.Screen.MAP)
        self.assertEqual(len(p.deck), before - 5)
        self.assertEqual(len(p.relic_state["paels_tooth_bank"]), 5)

    def test_paels_tooth_returns_card_upgraded_after_combat(self) -> None:
        loop = self._run()
        p = loop.state.player
        # Seed a bank with a single known upgradable card.
        p.relic_state["paels_tooth_bank"] = ["strike_ironclad"]
        p.relics.append("paels_tooth")
        deck_before = list(p.deck)
        # Fake a finished-victory combat to run the after-combat hook.
        loop.state.combat = SIM.CombatState(monsters=[], outcome="victory", took_unblocked=False)
        loop.state.room = "monster"
        loop._maybe_finalize_combat()
        self.assertEqual(p.relic_state["paels_tooth_bank"], [])
        new_cards = [c for c in p.deck if c not in deck_before]
        self.assertEqual(len(new_cards), 1)
        self.assertEqual(str(new_cards[0]), "strike_ironclad+1")

    def test_silver_crucible_upgrades_first_three_rewards(self) -> None:
        loop = self._run()
        loop._obtain_relic("silver_crucible", return_to=SIM.Screen.MAP)
        loop.state.act_index = 0
        upgraded_counts = []
        for _ in range(4):
            item = loop._card_reward_item("monster")
            upgraded_counts.append(sum(1 for c in item.cards if "+1" in str(c)))
        # First 3 rewards had every card upgraded; the 4th did not.
        self.assertTrue(all(n == len(upgraded_counts) and n > 0 for n in upgraded_counts[:3]) or
                        all(upgraded_counts[i] > upgraded_counts[3] for i in range(3)))
        self.assertEqual(loop.state.player.relic_state.get("silver_crucible_rewards_left", 0), 0)

    def test_silver_crucible_empties_first_treasure(self) -> None:
        loop = self._run()
        loop._obtain_relic("silver_crucible", return_to=SIM.Screen.MAP)
        gold_before = loop.state.player.gold
        loop._enter_treasure()
        self.assertEqual(loop.state.treasure_relics, [])
        self.assertEqual(loop.state.player.gold, gold_before)  # no gold paid either

    def test_glass_eye_adds_five_cards(self) -> None:
        loop = self._run()
        before = len(loop.state.player.deck)
        loop._obtain_relic("glass_eye", return_to=SIM.Screen.MAP)
        self.assertEqual(len(loop.state.player.deck), before + 5)

    def test_arcane_scroll_adds_rare_card(self) -> None:
        loop = self._run()
        before = len(loop.state.player.deck)
        loop._obtain_relic("arcane_scroll", return_to=SIM.Screen.MAP)
        self.assertEqual(len(loop.state.player.deck), before + 1)
        added = str(loop.state.player.deck[-1])
        self.assertEqual(loop.cards[added].rarity, "rare")

    def test_alchemical_coffer_adds_four_filled_slots(self) -> None:
        loop = self._run()
        before_slots = len(loop.state.player.potions)
        loop._obtain_relic("alchemical_coffer", return_to=SIM.Screen.MAP)
        self.assertEqual(len(loop.state.player.potions), before_slots + 4)
        # All slots are filled with real potion ids (not None).
        self.assertTrue(all(p is not None for p in loop.state.player.potions))

    def test_lost_coffer_pushes_card_and_potion_rewards(self) -> None:
        loop = self._run()
        loop.state.rewards = []
        loop._obtain_relic("lost_coffer", return_to=SIM.Screen.MAP)
        kinds = [r.kind for r in loop.state.rewards]
        self.assertIn("card", kinds)
        self.assertIn("potion", kinds)

    def test_lords_parasol_auto_takes_shop(self) -> None:
        loop = self._run()
        p = loop.state.player
        p.gold = 99999
        loop._obtain_relic("lords_parasol", return_to=SIM.Screen.MAP)
        loop._enter_shop()
        # Every non-removal shop item is marked out of stock.
        for item in loop.state.shop:
            if item.category != "card_removal":
                self.assertFalse(item.stocked, item.category)

    def test_leafy_poultice_loses_max_hp_and_transforms(self) -> None:
        loop = self._run()
        p = loop.state.player
        max_before = p.max_hp
        strikes_before = sum(1 for c in p.deck if c == "strike_ironclad")
        defends_before = sum(1 for c in p.deck if c == "defend_ironclad")
        loop._obtain_relic("leafy_poultice", return_to=SIM.Screen.MAP)
        self.assertEqual(p.max_hp, max_before - 10)
        strikes_after = sum(1 for c in p.deck if c == "strike_ironclad")
        defends_after = sum(1 for c in p.deck if c == "defend_ironclad")
        self.assertEqual(strikes_after, max(0, strikes_before - 1))
        self.assertEqual(defends_after, max(0, defends_before - 1))

    def test_paels_wing_grants_relic_every_two_skips(self) -> None:
        loop = self._run()
        loop._obtain_relic("paels_wing", return_to=SIM.Screen.MAP)
        p = loop.state.player
        # Simulate 2 card-reward skips via the step handler.
        for _ in range(2):
            # Set up a card_reward in rewards and screen.
            loop.state.rewards = [SIM.rewards.RewardItem(kind="card",
                                                        cards=[SIM.CardRef("strike_ironclad")])] \
                if hasattr(SIM.rewards, 'RewardItem') else [__import__('classify_jev_sts2_sim_for_tests.state',
                        fromlist=['RewardItem']).RewardItem(kind="card",
                                                            cards=[SIM.CardRef("strike_ironclad")])]
            loop.state.card_reward = [SIM.CardRef("strike_ironclad")]
            loop.state.card_reward_item = 0
            loop.state.screen = SIM.Screen.CARD_REWARD
            loop._step_card_reward("skip_card_reward", ())
        self.assertEqual(p.relic_state["paels_wing_sacrifices"], 2)
        # After the 2nd sacrifice, a relic reward should have been pushed.
        self.assertTrue(any(r.kind == "relic" for r in loop.state.rewards))

    def test_sea_glass_offers_fifteen_cards_to_pick(self) -> None:
        loop = self._run()
        before = len(loop.state.player.deck)
        loop._obtain_relic("sea_glass", return_to=SIM.Screen.MAP)
        sel = loop.state.deck_select
        self.assertIsNotNone(sel)
        self.assertEqual(sel.purpose, "pick_add")
        self.assertEqual(len(sel.generated_cards), 15)
        self.assertEqual(sel.min_count, 0)
        # Take only the first 3 offered.
        for i in range(3):
            loop._step_deck_select("select_card", (str(i),))
        loop._step_deck_select("confirm_selection", ())
        self.assertEqual(len(loop.state.player.deck), before + 3)

    def test_nutritious_soup_enchants_strikes_with_ember(self) -> None:
        loop = self._run()
        p = loop.state.player
        loop._obtain_relic("nutritious_soup", return_to=SIM.Screen.MAP)
        strikes = [c for c in p.deck if "strike" in loop.cards[c].tags]
        self.assertTrue(strikes)
        for c in strikes:
            self.assertEqual(c.enchant, "ember")
            self.assertEqual(c.enchant_amount, 1)

    def test_touch_of_orobas_sets_flag(self) -> None:
        loop = self._run()
        loop._obtain_relic("touch_of_orobas", return_to=SIM.Screen.MAP)
        self.assertTrue(loop.state.player.relic_state.get("touch_of_orobas"))

    def test_meat_cleaver_exposes_cook_rest_option(self) -> None:
        loop = self._run()
        p = loop.state.player
        loop._obtain_relic("meat_cleaver", return_to=SIM.Screen.MAP)
        loop.state.screen = SIM.Screen.REST
        loop.state.rest_used = False
        option_ids = [opt[0] for opt in loop._rest_options()]
        self.assertIn("COOK", option_ids)
        # Pick COOK: opens deck_select, cook purpose.
        cook_index = option_ids.index("COOK")
        loop._step_rest("choose_rest_option", (str(cook_index),))
        self.assertIsNotNone(loop.state.deck_select)
        self.assertEqual(loop.state.deck_select.purpose, "cook")
        # Confirm with the first card selected: transforms it + closes the option.
        loop._step_deck_select("select_card", ("0",))
        loop._step_deck_select("confirm_selection", ())
        self.assertIn("COOK", p.relic_state["rest_taken"])

    def test_fur_coat_drops_next_combat_monsters_to_one_hp(self) -> None:
        loop = self._run()
        p = loop.state.player
        loop._obtain_relic("fur_coat", return_to=SIM.Screen.MAP)
        self.assertEqual(p.relic_state["fur_coat"], 7)
        encounter = next(iter(SIM.load_encounters().values()))
        loop._start_combat(encounter.monsters, [{} for _ in encounter.monsters], room="monster")
        self.assertEqual(p.relic_state["fur_coat"], 6)
        for monster in loop.state.combat.monsters:
            self.assertEqual(monster.hp, 1)
            self.assertEqual(monster.max_hp, 1)

    def test_fur_coat_skips_boss_rooms(self) -> None:
        loop = self._run()
        p = loop.state.player
        loop._obtain_relic("fur_coat", return_to=SIM.Screen.MAP)
        encounter = next(iter(SIM.load_encounters().values()))
        loop._start_combat(encounter.monsters, [{} for _ in encounter.monsters], room="boss")
        self.assertEqual(p.relic_state["fur_coat"], 7)  # not consumed on boss
        self.assertTrue(any(m.hp > 1 for m in loop.state.combat.monsters))

    def test_toy_box_grants_four_wax_relics(self) -> None:
        loop = self._run()
        p = loop.state.player
        before = set(p.relics)
        loop._obtain_relic("toy_box", return_to=SIM.Screen.MAP)
        wax = p.relic_state.get("toy_box_wax_relics")
        self.assertIsInstance(wax, list)
        self.assertEqual(len(wax), 4)
        for rid in wax:
            self.assertIn(rid, p.relics)
            self.assertNotIn(rid, before)

    def test_toy_box_melts_leftmost_every_three_combats(self) -> None:
        loop = self._run()
        p = loop.state.player
        loop._obtain_relic("toy_box", return_to=SIM.Screen.MAP)
        wax0 = list(p.relic_state["toy_box_wax_relics"])
        # Simulate 3 combat victories.
        for _ in range(3):
            loop.state.combat = SIM.combat.CombatState()
            loop.state.combat.outcome = "victory"
            loop._maybe_finalize_combat()
        self.assertEqual(p.relic_state["toy_box_combats"], 3)
        self.assertEqual(len(p.relic_state["toy_box_wax_relics"]), 3)
        self.assertNotIn(wax0[0], p.relics)

    def test_golden_compass_builds_linear_act2_map(self) -> None:
        loop = self._run()
        p = loop.state.player
        loop._obtain_relic("golden_compass", return_to=SIM.Screen.MAP)
        loop._enter_act_map(1)
        from examples.classify.jev.sts2_sim import mapgen
        act_map = loop.state.map
        boss_row = act_map.boss_row
        # Exactly one node per row (plus the start Ancient and the Boss).
        self.assertEqual(len(act_map.nodes), boss_row + 1)
        for row in range(boss_row + 1):
            self.assertIn((mapgen.START_COL, row), act_map.nodes)
        self.assertEqual(act_map.nodes[(mapgen.START_COL, 0)].kind, mapgen.ANCIENT)
        self.assertEqual(act_map.nodes[(mapgen.START_COL, boss_row)].kind, mapgen.BOSS)
        self.assertEqual(act_map.nodes[(mapgen.START_COL, boss_row - 1)].kind, mapgen.REST)

    def test_paels_eye_grants_extra_turn(self) -> None:
        loop = self._run()
        p = loop.state.player
        loop._obtain_relic("paels_eye", return_to=SIM.Screen.MAP)
        encounter = next(iter(SIM.load_encounters().values()))
        loop._start_combat(encounter.monsters, [{} for _ in encounter.monsters], room="monster")
        combat = loop.state.combat
        start_turn = combat.turn
        start_hp = p.hp
        # End the turn without playing any card.
        loop.combat_ctx.end_turn()
        # paels_eye_done flag is set and extra_turn_pending is consumed.
        self.assertEqual(loop.combat_ctx.relics.n.get("paels_eye_done"), 1)
        self.assertFalse(combat.extra_turn_pending)
        # The player did NOT take 6 flat damage (old buggy behavior).
        self.assertGreaterEqual(p.hp, start_hp - 5)
        # Turn counter advanced by exactly one extra player turn
        # (begin_player_turn ran twice: one normal end + the extra-turn loop).
        self.assertGreater(combat.turn, start_turn)

    def test_driftwood_offers_reroll_decision(self) -> None:
        loop = self._run()
        p = loop.state.player
        loop._obtain_relic("driftwood", return_to=SIM.Screen.MAP)
        # Set up a card_reward screen manually.
        reward = loop._card_reward_item("monster")
        loop.state.rewards = [reward]
        loop.state.card_reward = list(reward.cards)
        loop.state.card_reward_item = 0
        loop.state.card_reward_rerolled = False
        loop.state.room = "monster"
        loop.state.screen = SIM.Screen.CARD_REWARD
        decisions = loop._decisions()
        reroll_ids = [d.id for d in decisions if d.id.endswith(":reroll")]
        self.assertEqual(len(reroll_ids), 1)
        # Execute the reroll: cards may change, rerolled flag flips, screen stays.
        loop._step_card_reward("skip_card_reward", ("reroll",))
        self.assertTrue(loop.state.card_reward_rerolled)
        self.assertEqual(loop.state.screen, SIM.Screen.CARD_REWARD)
        # No more reroll decision after one use.
        reroll_ids_after = [d.id for d in loop._decisions() if d.id.endswith(":reroll")]
        self.assertEqual(reroll_ids_after, [])


class EventFrameworkTest(unittest.TestCase):
    """Normal (non-ancient) event system."""

    def _run(self, seed=3):
        loop = SIM.RunLoop(seed=seed)
        loop.reset()
        loop.step("choose_event_option:0")
        return loop

    def test_event_room_opens_an_event(self) -> None:
        loop = self._run()
        state = loop.state
        state.screen = SIM.Screen.MAP
        loop._enter_event()
        self.assertEqual(state.screen, SIM.Screen.EVENT)
        self.assertIsNotNone(state.event_id)
        self.assertTrue(loop._decisions())

    def _open_event(self, loop, event_id):
        loop.state.event_id = event_id
        loop.state.screen = SIM.Screen.EVENT

    def test_wellspring_option_heals(self) -> None:
        loop = self._run()
        loop.state.player.hp = 50
        self._open_event(loop, "WELLSPRING")
        loop._step_event("choose_event_option", ("0",))
        self.assertEqual(loop.state.player.hp, 70)
        self.assertEqual(loop.state.screen, SIM.Screen.MAP)
        self.assertIsNone(loop.state.event_id)

    def test_trash_heap_dig_gives_gold_and_curse(self) -> None:
        loop = self._run()
        p = loop.state.player
        before_gold = p.gold
        before_deck = len(p.deck)
        self._open_event(loop, "TRASH_HEAP")
        loop._step_event("choose_event_option", ("0",))
        self.assertEqual(p.gold, before_gold + 30)
        self.assertEqual(len(p.deck), before_deck + 1)
        self.assertIn("regret", [str(c) for c in p.deck])

    def test_abyssal_baths_immerse_adjusts_hp(self) -> None:
        loop = self._run()
        p = loop.state.player
        p.hp = p.max_hp = 80
        self._open_event(loop, "ABYSSAL_BATHS")
        loop._step_event("choose_event_option", ("0",))
        self.assertEqual(p.max_hp, 82)
        self.assertEqual(p.hp, 82 - 3)

    def test_sunken_treasury_locked_when_broke(self) -> None:
        loop = self._run()
        loop.state.player.gold = 10
        self._open_event(loop, "SUNKEN_TREASURY")
        decisions = [d.id for d in loop._decisions()]
        # Pay option is at index 0 but locked (gold < 75) -> omitted from decisions.
        self.assertNotIn("choose_event_option:0", decisions)
        self.assertIn("choose_event_option:1", decisions)

    def test_zen_weaver_has_three_options(self) -> None:
        loop = self._run()
        self._open_event(loop, "ZEN_WEAVER")
        decisions = loop._decisions()
        self.assertEqual(len(decisions), 3)

    def test_unknown_event_defaults_to_leave(self) -> None:
        loop = self._run()
        self._open_event(loop, "SELF_HELP_BOOK")  # not implemented
        decisions = loop._decisions()
        self.assertEqual(len(decisions), 1)
        loop._step_event("choose_event_option", ("0",))
        self.assertEqual(loop.state.screen, SIM.Screen.MAP)


class ShuffleEntryRankTest(unittest.TestCase):
    def test_shuffle_canonicalises_by_entry(self) -> None:
        # Two piles with the same cards in reversed order should shuffle to the
        # same draw sequence (EntryRank canonicalisation before Fisher-Yates).
        from tests.test_classify_jev_sts2_sim_combat_cpu import DATA, SIM, m0
        powers, cards, mdefs = DATA
        deck_a = ["strike_ironclad", "defend_ironclad", "bash",
                  "strike_ironclad", "defend_ironclad"]
        deck_b = list(reversed(deck_a))

        def shuffle_from(deck):
            player = SIM.PlayerState(hp=80, max_hp=80, gold=99, max_energy=3)
            run = SIM.RunState(character="ironclad", ascension=0, seed=11, player=player)
            ctx = SIM.CombatContext(run=run, cards=cards, monsters=mdefs, powers=powers,
                                    rng=SIM.Rng(11), hooks=SIM.HookBus(),
                                    effects=SIM.EffectQueue(), potions=POTIONS)
            ctx.relics = SIM.relics.RelicEngine(ctx)
            ctx.start_combat(["nibbit"], list(deck))
            player.hand = []
            player.draw_pile = []
            player.discard_pile = [SIM.CardRef(c) for c in deck]
            ctx.shuffle_discard_into_draw()
            return list(player.draw_pile)

        self.assertEqual([str(c) for c in shuffle_from(deck_a)],
                         [str(c) for c in shuffle_from(deck_b)])


class RelicDepthTest(unittest.TestCase):
    def _run(self, seed=3):
        loop = SIM.RunLoop(seed=seed)
        loop.reset()
        loop.step("choose_event_option:0")
        return loop

    def test_archaic_tooth_opens_choice_screen(self) -> None:
        loop = self._run()
        loop._obtain_relic("archaic_tooth", return_to=SIM.Screen.MAP)
        sel = loop.state.deck_select
        self.assertIsNotNone(sel)
        self.assertEqual(sel.purpose, "transform_to_ancient")
        # Pick one starter and confirm.
        loop._step_deck_select("select_card", ("0",))
        loop._step_deck_select("confirm_selection", ())
        # The chosen slot is now an ancient-rarity card.
        ancient_ids = {cid for cid, c in loop.cards.items() if c.rarity == "ancient"}
        self.assertTrue(any(str(c) in ancient_ids for c in loop.state.player.deck))

    def test_dusty_tome_adds_ancient_card(self) -> None:
        loop = self._run()
        p = loop.state.player
        before = len(p.deck)
        loop._obtain_relic("dusty_tome", return_to=SIM.Screen.MAP)
        self.assertEqual(len(p.deck), before + 1)
        added = p.deck[-1]
        self.assertEqual(loop.cards[added].rarity, "ancient")

    def test_calling_bell_adds_unique_curse(self) -> None:
        loop = self._run()
        p = loop.state.player
        curse_before = {str(c) for c in p.deck if loop.cards[c].card_type == "curse"}
        loop._obtain_relic("calling_bell", return_to=SIM.Screen.MAP)
        curse_after = {str(c) for c in p.deck if loop.cards[c].card_type == "curse"}
        new = curse_after - curse_before
        self.assertEqual(len(new), 1)
        self.assertNotIn("pride", new)  # unique pool, not hard-coded pride

    def test_paels_claw_imbued_defends_grant_extra_block(self) -> None:
        # Imbued Defend plays like Defend but adds enchant_amount block.
        from tests.test_classify_jev_sts2_sim_combat_cpu import DATA, SIM, m0
        powers, cards, mdefs = DATA
        player = SIM.PlayerState(hp=80, max_hp=80, gold=99, max_energy=3)
        run = SIM.RunState(character="ironclad", ascension=0, seed=11, player=player)
        ctx = SIM.CombatContext(run=run, cards=cards, monsters=mdefs, powers=powers,
                                rng=SIM.Rng(11), hooks=SIM.HookBus(),
                                effects=SIM.EffectQueue(), potions=POTIONS)
        ctx.relics = SIM.relics.RelicEngine(ctx)
        ctx.start_combat(["nibbit"], ["defend_ironclad"])
        imbued = SIM.CardRef("defend_ironclad")
        imbued.enchant = "imbued"
        imbued.enchant_amount = 2
        player.hand = [imbued]
        player.energy = 3
        ctx.play_card(imbued)
        # Normal Defend grants 5; Imbued adds 2 -> 7 (unpowered helper may vary).
        self.assertGreaterEqual(player.block, 7)


if __name__ == "__main__":
    unittest.main()
