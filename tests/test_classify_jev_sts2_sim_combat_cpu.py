"""sts2_sim combat engine CPU tests (STS2 v0.107.1 rules).

Builds combats directly (start_combat + a planted hand) and checks damage
math, card effects, card selections, powers, statuses, monster mechanics,
turn flow and the error surface. Expected numbers follow the game's rules
as ported from r33hab/sts2 (`BuffSystem.IncomingDamage`, `CardEffects`,
`EnemyAI`) and the v0.107.1 card / monster data.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _import_sim():
    mod_name = "classify_jev_sts2_sim_for_tests"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    root = Path(__file__).resolve().parents[1] / "examples" / "classify" / "jev" / "sts2_sim"
    init = root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        mod_name, init, submodule_search_locations=[str(root)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


SIM = _import_sim()
DATA = SIM.load_all()


def make(monsters, *, deck=None, hand=None, energy=3, seed=7, hp=100, monster_hp=200, ascension=0,
         flags=None):
    powers, cards, mdefs = DATA
    player = SIM.PlayerState(hp=80, max_hp=80, gold=99, max_energy=3)
    run = SIM.RunState(character="ironclad", ascension=ascension, seed=seed, player=player)
    ctx = SIM.CombatContext(run=run, cards=cards, monsters=mdefs, powers=powers, rng=SIM.Rng(seed),
                            hooks=SIM.HookBus(), effects=SIM.EffectQueue())
    ctx.start_combat(list(monsters), list(deck or ["strike_ironclad"] * 10), monster_flags=flags)
    player.hp = player.max_hp = hp
    if monster_hp is not None:
        for m in ctx.combat.monsters:
            m.hp = m.max_hp = monster_hp
    if hand is not None:
        player.draw_pile.extend(player.hand)
        player.hand = list(hand)
    player.energy = energy
    return ctx


def m0(ctx):
    return ctx.combat.monsters[0]


class DamageMathTest(unittest.TestCase):
    def test_strike_and_upgrade(self) -> None:
        ctx = make(["nibbit"], hand=["strike_ironclad", "strike_ironclad+1"])
        ctx.play_card("strike_ironclad", target_slot=0)
        ctx.play_card("strike_ironclad+1", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 6 - 9)

    def test_strength_weak_vulnerable_truncate_once(self) -> None:
        # (6 + 1) * 0.75 * 1.5 = 7.875 -> 7 (one truncation at the end).
        ctx = make(["nibbit"], hand=["strike_ironclad"])
        ctx.player.powers.update(strength=1, weak=1)
        m0(ctx).powers["vulnerable"] = 1
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).hp, 193)

    def test_cruelty_raises_vulnerable_multiplier(self) -> None:
        ctx = make(["nibbit"], hand=["strike_ironclad"])
        ctx.player.powers["cruelty"] = 25
        m0(ctx).powers["vulnerable"] = 2
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - int(6 * 1.75))

    def test_dexterity_and_frail_on_block(self) -> None:
        ctx = make(["nibbit"], hand=["defend_ironclad"])
        ctx.player.powers.update(dexterity=2, frail=1)
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.block, int((5 + 2) * 0.75))

    def test_block_absorbs_before_hp(self) -> None:
        ctx = make(["nibbit"], hand=["strike_ironclad"])
        m0(ctx).block = 4
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual((m0(ctx).block, m0(ctx).hp), (0, 198))


class AttackCardTest(unittest.TestCase):
    def test_bash_applies_vulnerable_and_vicious_draws(self) -> None:
        ctx = make(["nibbit"], hand=["bash"])
        ctx.player.powers["vicious"] = 1
        ctx.play_card("bash", target_slot=0)
        self.assertEqual((m0(ctx).hp, m0(ctx).powers["vulnerable"]), (192, 2))
        self.assertEqual(len(ctx.player.hand), 1)

    def test_artifact_blocks_debuff(self) -> None:
        ctx = make(["punch_construct"], hand=["bash"])
        ctx.play_card("bash", target_slot=0)
        self.assertNotIn("vulnerable", m0(ctx).powers)
        self.assertNotIn("artifact", m0(ctx).powers)

    def test_anger_copies_itself(self) -> None:
        ctx = make(["nibbit"], hand=["anger+1"])
        ctx.play_card("anger+1", target_slot=0)
        self.assertEqual(m0(ctx).hp, 192)
        self.assertEqual(ctx.player.discard_pile.count("anger+1"), 2)

    def test_twin_strike_and_dismantle(self) -> None:
        ctx = make(["nibbit"], hand=["twin_strike", "dismantle"])
        ctx.play_card("twin_strike", target_slot=0)
        self.assertEqual(m0(ctx).hp, 190)
        m0(ctx).powers["vulnerable"] = 1
        ctx.play_card("dismantle", target_slot=0)
        self.assertEqual(m0(ctx).hp, 190 - 2 * 12)

    def test_body_slam_uses_block_plus_strength(self) -> None:
        ctx = make(["nibbit"], hand=["body_slam"])
        ctx.player.block = 9
        ctx.player.powers["strength"] = 2
        ctx.play_card("body_slam", target_slot=0)
        self.assertEqual(m0(ctx).hp, 189)

    def test_perfected_strike_counts_every_strike(self) -> None:
        ctx = make(["nibbit"], deck=["defend_ironclad"] * 10, hand=["perfected_strike", "twin_strike"])
        ctx.player.discard_pile = ["strike_ironclad", "pommel_strike"]
        ctx.play_card("perfected_strike", target_slot=0)
        # itself + twin_strike + strike + pommel_strike = 4 Strikes.
        self.assertEqual(m0(ctx).hp, 200 - (6 + 2 * 4))

    def test_ashen_strike_and_pacts_end_read_exhaust_pile(self) -> None:
        ctx = make(["nibbit", "nibbit"], hand=["ashen_strike", "pacts_end"])
        ctx.player.exhaust_pile = ["wound", "wound"]
        ctx.play_card("pacts_end")
        self.assertEqual(m0(ctx).hp, 200, "below 3 exhausted: no damage")
        ctx.play_card("ashen_strike", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - (6 + 3 * 2))

    def test_hemokinesis_breakthrough_pay_hp(self) -> None:
        ctx = make(["nibbit", "nibbit"], hand=["hemokinesis", "breakthrough"])
        ctx.play_card("hemokinesis", target_slot=1)
        ctx.play_card("breakthrough")
        self.assertEqual(ctx.player.hp, 100 - 2 - 1)
        self.assertEqual([m.hp for m in ctx.combat.monsters], [200 - 9, 200 - 15 - 9])

    def test_whirlwind_spends_all_energy(self) -> None:
        ctx = make(["nibbit", "nibbit"], hand=["whirlwind"], energy=3)
        ctx.play_card("whirlwind")
        self.assertEqual(ctx.player.energy, 0)
        self.assertEqual([m.hp for m in ctx.combat.monsters], [185, 185])

    def test_rampage_grows(self) -> None:
        ctx = make(["nibbit"], hand=["rampage"])
        ctx.play_card("rampage", target_slot=0)
        ctx.player.hand.append(ctx.player.discard_pile.pop())
        ctx.player.energy = 1
        ctx.play_card("rampage", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 9 - 14)

    def test_rampage_growth_is_per_copy(self) -> None:
        ctx = make(["nibbit"], hand=["rampage", "rampage"])
        ctx.player.energy = 2
        ctx.play_card("rampage", target_slot=0)
        ctx.play_card("rampage", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 9 - 9)
        self.assertEqual([c.bonus for c in ctx.player.discard_pile], [5, 5])

    def test_spite_hits_twice_after_losing_hp(self) -> None:
        ctx = make(["nibbit"], hand=["spite", "bloodletting", "spite+1"])
        ctx.play_card("spite", target_slot=0)
        ctx.play_card("bloodletting")
        ctx.play_card("spite+1", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 5 - 3 * 5)

    def test_feed_raises_max_hp_on_kill_only(self) -> None:
        ctx = make(["nibbit", "nibbit"], hand=["feed"], monster_hp=None)
        m0(ctx).hp = 5
        ctx.play_card("feed", target_slot=0)
        self.assertEqual(ctx.player.max_hp, 103)
        self.assertIn("feed", ctx.player.exhaust_pile)

    def test_fiend_fire_exhausts_hand(self) -> None:
        ctx = make(["nibbit"], hand=["fiend_fire", "wound", "defend_ironclad", "strike_ironclad"])
        ctx.play_card("fiend_fire", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 3 * 7)
        self.assertEqual(sorted(ctx.player.exhaust_pile),
                         ["defend_ironclad", "fiend_fire", "strike_ironclad", "wound"])

    def test_sword_boomerang_and_conflagration(self) -> None:
        ctx = make(["nibbit", "nibbit"], hand=["sword_boomerang", "conflagration"])
        ctx.play_card("sword_boomerang")
        self.assertEqual(sum(200 - m.hp for m in ctx.combat.monsters), 9)
        ctx.play_card("conflagration")
        self.assertEqual(sum(200 - m.hp for m in ctx.combat.monsters), 9 + 2 * 4 * 2)

    def test_stomp_cost_drops_per_attack(self) -> None:
        ctx = make(["nibbit"], hand=["stomp", "strike_ironclad", "strike_ironclad"])
        self.assertEqual(ctx.effective_cost("stomp"), 3)
        ctx.play_card("strike_ironclad", target_slot=0)
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(ctx.effective_cost("stomp"), 1)

    def test_unrelenting_makes_next_attack_free(self) -> None:
        ctx = make(["nibbit"], hand=["unrelenting", "bludgeon"], energy=5)
        ctx.play_card("unrelenting", target_slot=0)
        self.assertEqual(ctx.effective_cost("bludgeon"), 0)
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(ctx.player.energy, 3)
        self.assertEqual(m0(ctx).hp, 200 - 14 - 32)

    def test_mangle_strength_loss_lasts_one_enemy_turn(self) -> None:
        ctx = make(["nibbit"], hand=["mangle"])
        ctx.play_card("mangle", target_slot=0)
        self.assertEqual(m0(ctx).powers["strength"], -10)
        ctx.end_turn()
        self.assertNotIn("strength", m0(ctx).powers)

    def test_molten_fist_doubles_vulnerable(self) -> None:
        ctx = make(["nibbit"], hand=["molten_fist"])
        m0(ctx).powers["vulnerable"] = 2
        ctx.play_card("molten_fist", target_slot=0)
        self.assertEqual(m0(ctx).powers["vulnerable"], 4)

    def test_thrash_eats_an_attack(self) -> None:
        ctx = make(["nibbit"], hand=["thrash", "bludgeon"])
        ctx.play_card("thrash", target_slot=0)
        self.assertEqual(m0(ctx).hp, 192)
        thrash = next(c for c in ctx.player.discard_pile if c == "thrash")
        self.assertEqual(thrash.bonus, 32)
        self.assertIn("bludgeon", ctx.player.exhaust_pile)

    def test_setup_strike_strength_is_temporary(self) -> None:
        ctx = make(["nibbit"], hand=["setup_strike"])
        ctx.play_card("setup_strike", target_slot=0)
        self.assertEqual(ctx.player.powers["strength"], 2)
        ctx.end_turn()
        self.assertNotIn("strength", ctx.player.powers)


class SkillCardTest(unittest.TestCase):
    def test_bloodletting_offering(self) -> None:
        ctx = make(["nibbit"], hand=["bloodletting", "offering"], energy=0)
        ctx.play_card("bloodletting")
        ctx.play_card("offering")
        self.assertEqual(ctx.player.hp, 100 - 3 - 6)
        self.assertEqual(ctx.player.energy, 4)
        self.assertEqual(len(ctx.player.hand), 3)

    def test_battle_trance_blocks_further_draws(self) -> None:
        ctx = make(["nibbit"], hand=["battle_trance", "pommel_strike"])
        ctx.play_card("battle_trance")
        self.assertEqual(len(ctx.player.hand), 4)
        ctx.play_card("pommel_strike", target_slot=0)
        self.assertEqual(len(ctx.player.hand), 3)

    def test_second_wind_and_evil_eye(self) -> None:
        ctx = make(["nibbit"], hand=["second_wind", "defend_ironclad", "wound", "strike_ironclad", "evil_eye"])
        ctx.play_card("second_wind")
        # Exhausts Defend, Wound and Evil Eye (non-Attacks): 3 x 5 block.
        self.assertEqual(ctx.player.block, 15)
        self.assertEqual(ctx.player.hand, ["strike_ironclad"])

    def test_evil_eye_doubles_after_exhaust(self) -> None:
        ctx = make(["nibbit"], hand=["true_grit", "wound"])
        ctx.play_card("true_grit")
        ctx.player.hand.append("evil_eye")
        ctx.play_card("evil_eye")
        self.assertEqual(ctx.player.block, 7 + 16)

    def test_expect_a_fight_counts_attacks_and_blocks_more_energy(self) -> None:
        ctx = make(["nibbit"], hand=["expect_a_fight", "strike_ironclad", "bash", "bloodletting"], energy=3)
        ctx.play_card("expect_a_fight")
        self.assertEqual(ctx.player.energy, 1 + 2)
        ctx.play_card("bloodletting")
        self.assertEqual(ctx.player.energy, 3)

    def test_flame_barrier_hits_back_per_hit(self) -> None:
        ctx = make(["inklet"], hand=["flame_barrier"], energy=2, flags=[{}])
        ctx.combat.monsters[0].queued_move = "WHIRLWIND"
        ctx.play_card("flame_barrier")
        ctx.end_turn()
        # 4 back per hit; Slippery 1 caps only the first HP loss at 1.
        self.assertEqual(m0(ctx).hp, 200 - 1 - 4 - 4)

    def test_havoc_plays_top_card_and_exhausts_it(self) -> None:
        ctx = make(["nibbit"], hand=["havoc"])
        ctx.player.draw_pile.append("bash")
        ctx.play_card("havoc")
        self.assertEqual(m0(ctx).hp, 192)
        self.assertIn("bash", ctx.player.exhaust_pile)
        self.assertEqual(ctx.player.energy, 2, "the auto-played card is free")

    def test_cascade_plays_x_cards(self) -> None:
        ctx = make(["nibbit"], hand=["cascade+1"], energy=2)
        ctx.player.draw_pile = ["defend_ironclad", "strike_ironclad", "strike_ironclad"]
        ctx.play_card("cascade+1")
        self.assertEqual(m0(ctx).hp, 188)
        self.assertEqual(ctx.player.block, 5)
        self.assertEqual(ctx.player.energy, 0)

    def test_infernal_blade_adds_free_attack(self) -> None:
        ctx = make(["nibbit"], hand=["infernal_blade"])
        ctx.play_card("infernal_blade")
        added = ctx.player.hand[-1]
        self.assertEqual(ctx.cards[added].card_type, "attack")
        self.assertEqual(ctx.effective_cost(added), 0)

    def test_primal_force_turns_attacks_into_giant_rock(self) -> None:
        ctx = make(["nibbit"], hand=["primal_force+1", "strike_ironclad", "defend_ironclad"])
        ctx.play_card("primal_force+1")
        self.assertEqual(ctx.player.hand, ["giant_rock+1", "defend_ironclad"])

    def test_dominate(self) -> None:
        ctx = make(["nibbit"], hand=["dominate"])
        m0(ctx).powers["vulnerable"] = 2
        ctx.play_card("dominate", target_slot=0)
        self.assertEqual(ctx.player.powers["strength"], 3)


class SelectionTest(unittest.TestCase):
    def test_burning_pact_prompts_then_draws(self) -> None:
        ctx = make(["nibbit"], hand=["burning_pact", "wound", "strike_ironclad"])
        ctx.play_card("burning_pact")
        sel = ctx.combat.pending_selection
        self.assertIsNotNone(sel)
        self.assertEqual((sel.source, sel.purpose, sel.candidates), ("hand", "exhaust", [0, 1]))
        with self.assertRaises(SIM.CombatError):
            ctx.end_turn()
        ctx.toggle_selection(0)
        ctx.confirm_selection()
        self.assertIn("wound", ctx.player.exhaust_pile)
        self.assertEqual(len(ctx.player.hand), 1 + 2)
        self.assertIn("burning_pact", ctx.player.discard_pile)

    def test_true_grit_random_vs_chosen(self) -> None:
        ctx = make(["nibbit"], hand=["true_grit", "wound", "defend_ironclad"])
        ctx.play_card("true_grit")
        self.assertIsNone(ctx.combat.pending_selection)
        self.assertEqual(len(ctx.player.exhaust_pile), 1)
        ctx = make(["nibbit"], hand=["true_grit+1", "wound", "defend_ironclad"])
        ctx.play_card("true_grit+1")
        self.assertIsNotNone(ctx.combat.pending_selection)

    def test_armaments_upgrades_chosen_or_all(self) -> None:
        ctx = make(["nibbit"], hand=["armaments", "strike_ironclad", "wound", "bash+1"])
        ctx.play_card("armaments")
        sel = ctx.combat.pending_selection
        self.assertEqual(sel.candidates, [0], "only the un-upgraded Strike is upgradable")
        ctx.toggle_selection(0)
        ctx.confirm_selection()
        self.assertEqual(ctx.player.hand, ["strike_ironclad+1", "wound", "bash+1"])
        ctx = make(["nibbit"], hand=["armaments+1", "strike_ironclad", "defend_ironclad"])
        ctx.play_card("armaments+1")
        self.assertEqual(ctx.player.hand, ["strike_ironclad+1", "defend_ironclad+1"])

    def test_headbutt_picks_from_discard(self) -> None:
        ctx = make(["nibbit"], hand=["headbutt"])
        ctx.player.discard_pile = ["bash", "defend_ironclad"]
        ctx.play_card("headbutt", target_slot=0)
        sel = ctx.combat.pending_selection
        self.assertEqual(sel.source, "discard")
        ctx.toggle_selection(0)
        ctx.confirm_selection()
        self.assertEqual(ctx.player.draw_pile[-1], "bash")

    def test_brand_exhausts_then_strength(self) -> None:
        ctx = make(["nibbit"], hand=["brand", "wound"])
        ctx.play_card("brand")
        ctx.toggle_selection(0)
        ctx.confirm_selection()
        self.assertEqual((ctx.player.hp, ctx.player.powers["strength"]), (99, 1))


class PowerTest(unittest.TestCase):
    def test_rupture_pays_after_the_card(self) -> None:
        ctx = make(["nibbit"], hand=["rupture", "hemokinesis"])
        ctx.play_card("rupture")
        ctx.play_card("hemokinesis", target_slot=0)
        # Hemokinesis' own hit does not get the Strength: it is paid after the card.
        self.assertEqual(m0(ctx).hp, 185)
        self.assertEqual(ctx.player.powers["strength"], 1)

    def test_inferno_burns_on_self_damage(self) -> None:
        ctx = make(["nibbit", "nibbit"], hand=["inferno", "bloodletting"])
        ctx.play_card("inferno")
        ctx.play_card("bloodletting")
        self.assertEqual([m.hp for m in ctx.combat.monsters], [194, 194])

    def test_feel_no_pain_and_dark_embrace(self) -> None:
        ctx = make(["nibbit"], hand=["feel_no_pain", "dark_embrace", "true_grit", "wound"], energy=4)
        ctx.play_card("feel_no_pain")
        ctx.play_card("dark_embrace")
        ctx.play_card("true_grit")
        self.assertEqual(ctx.player.block, 7 + 3)
        self.assertEqual(len(ctx.player.hand), 1)

    def test_juggernaut_hits_on_block(self) -> None:
        ctx = make(["nibbit"], hand=["juggernaut", "defend_ironclad"])
        ctx.play_card("juggernaut")
        ctx.play_card("defend_ironclad")
        self.assertEqual(m0(ctx).hp, 194)

    def test_barricade_and_demon_form_carry_over(self) -> None:
        ctx = make(["nibbit"], hand=["barricade"], energy=3)
        ctx.player.powers["demon_form"] = 2
        ctx.play_card("barricade")
        ctx.player.block = 30
        m0(ctx).queued_move = "HISS"
        ctx.end_turn()
        self.assertEqual(ctx.player.block, 30)
        self.assertEqual(ctx.player.powers["strength"], 2)

    def test_crimson_mantle_and_stone_armor(self) -> None:
        ctx = make(["nibbit"], hand=["crimson_mantle", "stone_armor"])
        ctx.play_card("crimson_mantle")
        ctx.play_card("stone_armor")
        m0(ctx).queued_move = "HISS"
        ctx.end_turn()
        # Plating 4 -> 4 block at end of turn (block clears), then Crimson Mantle.
        self.assertEqual(ctx.player.hp, 99)
        self.assertEqual(ctx.player.block, 8)
        self.assertEqual(ctx.player.powers["plating"], 3)

    def test_corruption_free_skills_exhaust(self) -> None:
        ctx = make(["nibbit"], hand=["corruption", "shrug_it_off"])
        ctx.play_card("corruption")
        self.assertEqual(ctx.effective_cost("shrug_it_off"), 0)
        ctx.play_card("shrug_it_off")
        self.assertIn("shrug_it_off", ctx.player.exhaust_pile)

    def test_rage_one_two_punch_juggling(self) -> None:
        ctx = make(["nibbit"], hand=["rage", "one_two_punch", "strike_ironclad", "juggling"], energy=4)
        ctx.play_card("rage")
        ctx.play_card("juggling")
        ctx.play_card("one_two_punch")
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).hp, 188, "One-Two Punch replays the Strike")
        self.assertEqual(ctx.player.block, 3)

    def test_unmovable_doubles_first_card_block(self) -> None:
        ctx = make(["nibbit"], hand=["unmovable", "defend_ironclad", "defend_ironclad"], energy=4)
        ctx.play_card("unmovable")
        ctx.play_card("defend_ironclad")
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.block, 10 + 5)

    def test_hellraiser_autoplays_drawn_strikes(self) -> None:
        ctx = make(["nibbit"], hand=["hellraiser", "pommel_strike"], energy=3)
        ctx.play_card("hellraiser")
        ctx.player.draw_pile.append("twin_strike")
        ctx.play_card("pommel_strike", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 9 - 10)


class StatusTest(unittest.TestCase):
    def test_burn_and_infection_hurt_at_end_of_turn(self) -> None:
        ctx = make(["nibbit"], hand=["burn", "infection"])
        m0(ctx).queued_move = "HISS"
        ctx.player.block = 2
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, 100 - (2 + 3 - 2))

    def test_dazed_exhausts_and_slimed_draws(self) -> None:
        ctx = make(["nibbit"], hand=["dazed", "slimed"])
        ctx.play_card("slimed")
        self.assertIn("slimed", ctx.player.exhaust_pile)
        self.assertEqual(len(ctx.player.hand), 2)
        m0(ctx).queued_move = "HISS"
        ctx.end_turn()
        self.assertIn("dazed", ctx.player.exhaust_pile)

    def test_beckon_costs_hp_unblockable(self) -> None:
        ctx = make(["nibbit"], hand=["beckon"])
        ctx.player.block = 50
        m0(ctx).queued_move = "HISS"
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, 94)


class MonsterTest(unittest.TestCase):
    def test_nibbit_pair_openers(self) -> None:
        ctx = make(["nibbit", "nibbit"])
        self.assertEqual([m.queued_move for m in ctx.combat.monsters], ["SLICE", "HISS"])
        ctx = make(["nibbit"])
        self.assertEqual(m0(ctx).queued_move, "BUTT")

    def test_slice_blocks_and_attack_damage_scales(self) -> None:
        ctx = make(["nibbit"], hand=[])
        m0(ctx).queued_move = "SLICE"
        ctx.end_turn()
        self.assertEqual((ctx.player.hp, m0(ctx).block), (94, 5))
        ctx = make(["nibbit"], hand=[], ascension=9)
        m0(ctx).queued_move = "SLICE"
        ctx.end_turn()
        self.assertEqual((ctx.player.hp, m0(ctx).block), (93, 6))

    def test_debuff_from_enemy_skips_first_tick(self) -> None:
        ctx = make(["mawler"], hand=[])
        m0(ctx).queued_move = "ROAR"
        ctx.end_turn()
        self.assertEqual(ctx.player.powers["vulnerable"], 3)
        ctx.end_turn()
        self.assertEqual(ctx.player.powers["vulnerable"], 2)

    def test_cultist_ritual_starts_after_incantation_turn(self) -> None:
        ctx = make(["calcified_cultist"], hand=[])
        ctx.end_turn()
        self.assertEqual(m0(ctx).powers.get("strength", 0), 0)
        ctx.end_turn()
        self.assertEqual(m0(ctx).powers["strength"], 2)
        self.assertEqual(ctx.player.hp, 100 - 9)

    def test_slippery_caps_each_hp_loss(self) -> None:
        ctx = make(["vantom"], hand=["bludgeon", "strike_ironclad"], energy=4)
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual((m0(ctx).hp, m0(ctx).powers["slippery"]), (199, 7))

    def test_hardened_shell_caps_per_turn(self) -> None:
        ctx = make(["skulking_colony"], hand=["bludgeon"])
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(m0(ctx).hp, 180)

    def test_plow_stuns_and_switches_loop(self) -> None:
        ctx = make(["ceremonial_beast"], hand=["bludgeon"], monster_hp=None)
        beast = m0(ctx)
        beast.powers["plow"] = 150
        beast.powers["strength"] = 4
        beast.hp = 170
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(beast.queued_move, "__stunned__")
        self.assertNotIn("strength", beast.powers)
        ctx.end_turn()
        self.assertEqual(beast.queued_move, "BEAST_CRY")

    def test_shriek_then_terrorize(self) -> None:
        ctx = make(["terror_eel"], hand=["bludgeon"], monster_hp=None)
        eel = m0(ctx)
        eel.hp = 90
        ctx.play_card("bludgeon", target_slot=0)
        ctx.end_turn()
        self.assertEqual(eel.queued_move, "TERROR")
        ctx.end_turn()
        self.assertEqual(ctx.player.powers["vulnerable"], 99)

    def test_illusion_revives_and_minion_leaves_with_leader(self) -> None:
        ctx = make(["fogmog"], hand=[])
        ctx.end_turn()  # Illusory Spores summons the eye
        names = [m.monster_id for m in ctx.combat.monsters]
        self.assertEqual(names, ["eye_with_teeth", "fogmog"])
        eye, fog = ctx.combat.monsters
        ctx.player.hand = ["bludgeon"]
        ctx.player.energy = 3
        ctx.play_card("bludgeon", target_slot=0)
        self.assertFalse(eye.alive)
        self.assertIsNone(ctx.combat.outcome)
        ctx.end_turn()
        self.assertEqual(eye.hp, eye.max_hp)
        fog.hp = 1
        ctx.player.hand = ["strike_ironclad"]
        ctx.player.energy = 1
        ctx.play_card("strike_ironclad", target_slot=1)
        self.assertEqual(ctx.combat.outcome, "victory")

    def test_phrog_parasite_spawns_wrigglers(self) -> None:
        ctx = make(["phrog_parasite"], hand=["strike_ironclad"], monster_hp=None)
        m0(ctx).hp = 3
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertIsNone(ctx.combat.outcome)
        wrigglers = [m for m in ctx.combat.monsters if m.monster_id == "wriggler"]
        self.assertEqual(len(wrigglers), 4)
        self.assertTrue(all(w.queued_move == "__stunned__" for w in wrigglers))
        ctx.end_turn()
        self.assertEqual([w.queued_move for w in wrigglers], ["NASTY_BITE", "WRIGGLE", "NASTY_BITE", "WRIGGLE"])

    def test_gremlin_merc_steals_and_surprise(self) -> None:
        ctx = make(["gremlin_merc"], hand=[])
        ctx.end_turn()
        self.assertEqual((ctx.player.gold, m0(ctx).stolen_gold), (79, 20))
        m0(ctx).hp = 1
        ctx.player.hand = ["strike_ironclad"]
        ctx.player.energy = 1
        ctx.play_card("strike_ironclad", target_slot=0)
        ids = [m.monster_id for m in ctx.combat.monsters if m.alive]
        self.assertEqual(ids, ["sneaky_gremlin", "fat_gremlin"])

    def test_ravenous_slug_eats_ally(self) -> None:
        ctx = make(["corpse_slug", "corpse_slug"], hand=["strike_ironclad"], monster_hp=None,
                   flags=[{"starter": 0}, {"starter": 1}])
        a, b = ctx.combat.monsters
        a.hp = 1
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual((b.powers["strength"], b.queued_move), (4, "__stunned__"))

    def test_suck_gains_strength_on_unblocked_hit(self) -> None:
        ctx = make(["fossil_stalker"], hand=[])
        ctx.end_turn()
        self.assertEqual(m0(ctx).powers["strength"], 3)

    def test_toadpole_thorns_hurt_attacker(self) -> None:
        ctx = make(["toadpole"], hand=["strike_ironclad"])
        m0(ctx).powers["thorns"] = 2
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(ctx.player.hp, 98)

    def test_skittish_blocks_after_first_hit(self) -> None:
        ctx = make(["phantasmal_gardener"], hand=["strike_ironclad", "strike_ironclad"])
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).block, 6)
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual((m0(ctx).hp, m0(ctx).block), (194, 0))

    def test_lagavulin_wakes_when_struck(self) -> None:
        ctx = make(["lagavulin_matriarch"], hand=["bludgeon"])
        lag = m0(ctx)
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(lag.hp, 200 - 20)
        self.assertNotIn("asleep", lag.powers)
        self.assertNotIn("plating", lag.powers)
        ctx.end_turn()
        self.assertEqual(lag.queued_move, "SLASH")

    def test_waterfall_giant_explodes_after_death(self) -> None:
        ctx = make(["waterfall_giant"], hand=["strike_ironclad"], monster_hp=None)
        giant = m0(ctx)
        giant.powers["steam_eruption"] = 21
        giant.hp = 1
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertIsNone(ctx.combat.outcome)
        self.assertEqual(giant.queued_move, "ABOUT_TO_BLOW")
        ctx.end_turn()
        self.assertEqual(giant.queued_move, "EXPLODE")
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, 100 - 21)
        self.assertEqual(ctx.combat.outcome, "victory")

    def test_shrink_and_constrict_end_with_their_source(self) -> None:
        ctx = make(["slithering_strangler", "shrinker_beetle"], hand=[])
        ctx.end_turn()
        self.assertEqual((ctx.player.powers["constrict"], ctx.player.powers["shrink"]), (3, 1))
        for m in ctx.combat.monsters:
            m.hp = 1
        ctx.player.hand = ["strike_ironclad", "strike_ironclad"]
        ctx.player.energy = 2
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertNotIn("constrict", ctx.player.powers)

    def test_soul_siphon_takes_strength_and_dexterity(self) -> None:
        ctx = make(["lagavulin_matriarch"], hand=[])
        lag = m0(ctx)
        lag.powers.pop("asleep")
        lag.queued_move = "SOUL_SIPHON"
        ctx.end_turn()
        self.assertEqual((ctx.player.powers["strength"], ctx.player.powers["dexterity"]), (-2, -2))
        self.assertEqual(lag.powers["strength"], 2)

    def test_smoggy_ringing_tangled(self) -> None:
        ctx = make(["nibbit"], hand=["defend_ironclad", "shrug_it_off", "strike_ironclad"], energy=5)
        ctx.player.powers.update(smoggy=1, tangled=1)
        self.assertEqual(ctx.effective_cost("strike_ironclad"), 2)
        ctx.play_card("defend_ironclad")
        self.assertFalse(ctx.can_play("shrug_it_off"))
        ctx.player.powers["ringing"] = 1
        self.assertFalse(ctx.can_play("strike_ironclad"))


class TurnFlowTest(unittest.TestCase):
    def test_block_resets_and_energy_refills(self) -> None:
        ctx = make(["nibbit"], hand=["defend_ironclad"])
        ctx.play_card("defend_ironclad")
        m0(ctx).queued_move = "HISS"
        ctx.end_turn()
        self.assertEqual((ctx.player.block, ctx.player.energy, ctx.combat.turn), (0, 3, 2))
        self.assertEqual(len(ctx.player.hand), 5)

    def test_victory_and_defeat(self) -> None:
        ctx = make(["nibbit"], hand=["strike_ironclad"], monster_hp=None)
        m0(ctx).hp = 6
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(ctx.combat.outcome, "victory")
        ctx = make(["nibbit"], hand=[], hp=5)
        m0(ctx).queued_move = "BUTT"
        ctx.end_turn()
        self.assertEqual(ctx.combat.outcome, "defeat")


class ErrorSurfaceTest(unittest.TestCase):
    def test_errors(self) -> None:
        ctx = make(["nibbit"], hand=["bash", "strike_ironclad", "wound"], energy=1)
        with self.assertRaisesRegex(SIM.CombatError, "insufficient energy"):
            ctx.play_card("bash", target_slot=0)
        with self.assertRaisesRegex(SIM.CombatError, "not in hand"):
            ctx.play_card("anger", target_slot=0)
        with self.assertRaisesRegex(SIM.CombatError, "requires target_slot"):
            ctx.play_card("strike_ironclad")
        with self.assertRaisesRegex(SIM.CombatError, "unplayable"):
            ctx.play_card("wound")
        with self.assertRaisesRegex(SIM.CombatError, "unknown monster_id"):
            make(["jaw_worm"])


if __name__ == "__main__":
    unittest.main()
