"""sts2_sim Act 2 (Hive) / Act 3 (Glory) monster mechanics, CPU only.

Each test builds a combat against the monsters directly and checks one
hidden rule ported from r33hab/sts2 (`EnemyAI.cs`, `CombatEngine.cs`,
`CardEffects.cs`) against STS2 v0.107.1 numbers.
"""

from __future__ import annotations

import unittest

from tests.test_classify_jev_sts2_sim_combat_cpu import SIM, m0, make


def end_turn(ctx) -> None:
    ctx.end_turn()


class HiveMechanicsTest(unittest.TestCase):
    def test_hard_to_kill_caps_each_hit(self) -> None:
        ctx = make(["exoskeleton"], hand=["bludgeon"], monster_hp=50)
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(m0(ctx).hp, 50 - 9)

    def test_burrow_break_stuns_and_returns_to_bite(self) -> None:
        ctx = make(["tunneler"], hand=["strike_ironclad"], monster_hp=200, energy=3)
        m = m0(ctx)
        m.queued_move = "BELOW"
        m.ai_state = "BELOW"
        m.powers["burrowed"] = 1
        m.block = 6
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertNotIn("burrowed", m.powers)
        self.assertTrue(m.stunned)
        hp = ctx.player.hp
        end_turn(ctx)
        self.assertEqual(ctx.player.hp, hp)  # the stun ate its turn
        self.assertEqual(m.queued_move, "BITE")

    def test_burrowed_block_survives_the_enemy_turn(self) -> None:
        ctx = make(["tunneler"], hand=[])
        m = m0(ctx)
        end_turn(ctx)  # BITE
        end_turn(ctx)  # BURROW: +32 block, Burrowed
        self.assertEqual(m.powers.get("burrowed"), 1)
        self.assertEqual(m.block, 32)
        end_turn(ctx)  # BELOW: block kept through its own turn start
        self.assertEqual(m.block, 32)

    def test_bowlbug_rock_goes_dizzy_when_fully_blocked(self) -> None:
        ctx = make(["bowlbug_rock"], hand=[])
        ctx.player.block = 99
        end_turn(ctx)
        m = m0(ctx)
        self.assertEqual(m.queued_move, "DIZZY")
        hp = ctx.player.hp
        end_turn(ctx)
        self.assertEqual(ctx.player.hp, hp)
        self.assertEqual(m.queued_move, "HEADBUTT")

    def test_slumbering_beetle_wakes_on_third_hp_loss(self) -> None:
        ctx = make(["slumbering_beetle"], hand=["strike_ironclad"] * 3, monster_hp=200)
        m = m0(ctx)
        m.block = 0
        for _ in range(3):
            ctx.play_card("strike_ironclad", target_slot=0)
        self.assertNotIn("slumber", m.powers)
        self.assertTrue(m.stunned)
        end_turn(ctx)
        self.assertEqual(m.queued_move, "ROLL_OUT")
        self.assertNotIn("plating", m.powers)

    def test_thieving_hopper_steals_and_returns_on_death(self) -> None:
        deck = ["strike_ironclad"] * 8 + ["uppercut", "bash"]
        ctx = make(["thieving_hopper"], deck=deck, hand=[], monster_hp=20)
        end_turn(ctx)  # THIEVERY takes the Uncommon
        stolen = ctx.combat.stolen
        self.assertEqual([c for _, c in stolen], ["uppercut"])
        piles = ctx.player.draw_pile + ctx.player.discard_pile + ctx.player.hand
        self.assertNotIn("uppercut", piles)
        ctx.player.hand = ["bludgeon"]
        ctx.player.energy = 3
        ctx.play_card("bludgeon", target_slot=0)  # 32 halved by Flutter? not yet applied
        self.assertEqual(ctx.combat.stolen, [])

    def test_flutter_halves_and_stuns_after_five_hits(self) -> None:
        ctx = make(["thieving_hopper"], hand=["strike_ironclad"] * 5, energy=5)
        m = m0(ctx)
        m.powers["flutter"] = 5
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m.hp, 200 - 3)
        for _ in range(4):
            ctx.play_card("strike_ironclad", target_slot=0)
        self.assertNotIn("flutter", m.powers)
        self.assertTrue(m.stunned)

    def test_decimillipede_segment_reattaches_at_25(self) -> None:
        ctx = make(["decimillipede_segment_front", "decimillipede_segment_middle", "decimillipede_segment_back"],
                   hand=["bludgeon"], flags=[{"starter": 0}, {"starter": 1}, {"starter": 2}], monster_hp=30)
        seg = m0(ctx)
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(seg.hp, 0)
        self.assertEqual(seg.queued_move, "DEAD")
        self.assertIsNone(ctx.combat.outcome)
        end_turn(ctx)
        self.assertEqual(seg.queued_move, "REATTACH")
        self.assertEqual(seg.hp, 0)
        end_turn(ctx)
        self.assertEqual(seg.hp, 25)
        self.assertIn(seg.queued_move, ("WRITHE", "BULK", "CONSTRICT"))

    def test_last_segment_ends_the_fight(self) -> None:
        ctx = make(["decimillipede_segment_front"], hand=["bludgeon"], flags=[{"starter": 0}], monster_hp=30)
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(ctx.combat.outcome, "victory")

    def test_crusher_hits_from_behind_until_player_turns(self) -> None:
        ctx = make(["crusher", "rocket"], hand=["strike_ironclad"])
        self.assertEqual(ctx.player.powers["surrounded"], 1)
        crusher, rocket = ctx.combat.monsters
        self.assertTrue(ctx.attacks_from_behind(crusher))
        self.assertFalse(ctx.attacks_from_behind(rocket))
        ctx.play_card("strike_ironclad", target_slot=0)  # face the Crusher
        self.assertEqual(ctx.player.powers["surrounded"], 2)
        self.assertTrue(ctx.attacks_from_behind(rocket))
        hp = ctx.player.hp
        end_turn(ctx)
        # Crusher Thrash 12 from the front; Rocket Targeting Reticle 3 * 1.5 = 4 from behind.
        self.assertEqual(hp - ctx.player.hp, 12 + 4)

    def test_crab_rage_on_partner_death(self) -> None:
        ctx = make(["crusher", "rocket"], hand=["bludgeon"], monster_hp=20)
        ctx.play_card("bludgeon", target_slot=1)
        crusher = m0(ctx)
        self.assertEqual(crusher.powers.get("strength"), 6)
        self.assertEqual(crusher.block, 99)
        self.assertNotIn("crab_rage", crusher.powers)

    def test_curse_of_knowledge_is_a_choice(self) -> None:
        ctx = make(["knowledge_demon"], hand=[])
        end_turn(ctx)
        sel = ctx.combat.pending_selection
        self.assertIsNotNone(sel)
        self.assertEqual(sel.options, ["disintegration", "mind_rot"])
        ctx.toggle_selection(0)
        ctx.confirm_selection()
        self.assertEqual(ctx.player.powers["disintegration"], 6)
        self.assertEqual(ctx.combat.turn, 2)
        hp = ctx.player.hp
        ctx.player.block = 0
        ctx.player.hand = []
        end_turn(ctx)  # Slap 17 + Disintegration 6
        self.assertEqual(hp - ctx.player.hp, 17 + 6)

    def test_mind_rot_draws_fewer(self) -> None:
        ctx = make(["knowledge_demon"], hand=[])
        end_turn(ctx)
        ctx.toggle_selection(1)
        ctx.confirm_selection()
        self.assertEqual(ctx.player.powers["mind_rot"], 1)
        self.assertEqual(len(ctx.player.hand), 4)

    def test_sandpit_kills_and_frantic_escape_buys_time(self) -> None:
        ctx = make(["the_insatiable"], hand=[], hp=999)
        end_turn(ctx)  # LIQUIFY_GROUND
        m = m0(ctx)
        self.assertEqual(m.powers["sandpit"], 4)
        frantic = [c for c in ctx.player.draw_pile + ctx.player.discard_pile + ctx.player.hand
                   if c == "frantic_escape"]
        self.assertEqual(len(frantic), 6)
        ctx.player.hand = [SIM.CardRef("frantic_escape"), SIM.CardRef("frantic_escape")]
        ctx.player.energy = 3
        ctx.play_card("frantic_escape")
        self.assertEqual(m.powers["sandpit"], 5)
        played = ctx.player.discard_pile[-1]
        self.assertEqual(played.cost_bump, 1)
        self.assertEqual(ctx.effective_cost(played), 2)
        self.assertEqual(ctx.effective_cost("frantic_escape"), 1)  # the other copy
        for _ in range(5):
            if ctx.combat.outcome:
                break
            ctx.player.hand = []
            end_turn(ctx)
        self.assertEqual(ctx.combat.outcome, "defeat")

    def test_tender_takes_and_returns_stats(self) -> None:
        ctx = make(["hunter_killer"], hand=["defend_ironclad", "defend_ironclad"])
        ctx.player.powers["tender"] = 1
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.powers.get("strength"), -1)
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.block, 5 + 4)
        ctx.player.hand = []
        end_turn(ctx)
        self.assertNotIn("strength", ctx.player.powers)
        self.assertNotIn("dexterity", ctx.player.powers)

    def test_vital_spark_taints_skills(self) -> None:
        ctx = make(["infested_prism"], hand=["defend_ironclad"])
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.powers["tainted"], 2)
        ctx.player.block = 0
        hp = ctx.player.hp
        ctx.player.hand = []
        end_turn(ctx)  # Jab 15 + 2
        self.assertEqual(hp - ctx.player.hp, 17)
        self.assertNotIn("tainted", ctx.player.powers)

    def test_personal_hive_adds_dazed_per_hit(self) -> None:
        ctx = make(["entomancer"], hand=["twin_strike"])
        before = sum(1 for c in ctx.player.draw_pile if c == "dazed")
        ctx.play_card("twin_strike", target_slot=0)
        self.assertEqual(sum(1 for c in ctx.player.draw_pile if c == "dazed") - before, 2)

    def test_ovicopter_lays_eggs_that_hatch(self) -> None:
        ctx = make(["ovicopter"], hand=[])
        end_turn(ctx)
        eggs = [m for m in ctx.combat.monsters if m.monster_id == "tough_egg"]
        self.assertEqual(len(eggs), 3)
        self.assertEqual(ctx.combat.monsters[-1].monster_id, "ovicopter")
        self.assertTrue(all(m.powers.get("minion") for m in eggs))
        end_turn(ctx)
        for egg in eggs:
            self.assertTrue(19 <= egg.max_hp <= 21)
            self.assertNotIn("hatch", egg.powers)
            self.assertEqual(egg.queued_move, "NIBBLE")

    def test_obscura_parafright_is_an_illusion(self) -> None:
        ctx = make(["the_obscura"], hand=[])
        end_turn(ctx)
        para = m0(ctx)
        self.assertEqual(para.monster_id, "parafright")
        self.assertTrue(para.powers.get("illusion"))


class GloryMechanicsTest(unittest.TestCase):
    def test_axebot_respawns_from_stock(self) -> None:
        ctx = make(["axebot"], hand=["bludgeon"], monster_hp=20)
        ctx.play_card("bludgeon", target_slot=0)
        m = m0(ctx)
        self.assertTrue(m.alive)
        self.assertEqual(m.powers.get("stock"), 1)
        self.assertEqual(m.queued_move, "BOOT_UP")
        self.assertIsNone(ctx.combat.outcome)
        ctx.player.hand = []
        end_turn(ctx)  # BOOT_UP: block 10, +3 Strength (one helping at Stock 1)
        self.assertEqual(m.powers.get("strength"), 3)

    def test_test_subject_revives_twice(self) -> None:
        ctx = make(["test_subject"], hand=["bludgeon"], monster_hp=20)
        ctx.play_card("bludgeon", target_slot=0)
        m = m0(ctx)
        self.assertEqual(m.queued_move, "RESPAWN")
        self.assertIsNone(ctx.combat.outcome)
        end_turn(ctx)
        self.assertEqual((m.hp, m.max_hp), (200, 200))
        self.assertEqual(m.powers.get("painful_stabs"), 1)
        self.assertEqual(m.queued_move, "MULTI_CLAW")
        m.hp = 10
        ctx.player.hand = ["bludgeon"]
        ctx.player.energy = 3
        ctx.play_card("bludgeon", target_slot=0)
        end_turn(ctx)
        self.assertEqual(m.max_hp, 300)
        self.assertNotIn("adaptable", m.powers)
        self.assertEqual(m.powers.get("nemesis"), 1)
        self.assertEqual(m.queued_move, "PHASE3_LACERATE")

    def test_multi_claw_grows(self) -> None:
        ctx = make(["test_subject"], hand=[])
        m = m0(ctx)
        m.powers["painful_stabs"] = 1
        m.ai_state = m.queued_move = "MULTI_CLAW"
        hp = ctx.player.hp
        end_turn(ctx)  # 10 x 3, three Wounds
        self.assertEqual(hp - ctx.player.hp, 30)
        self.assertEqual(sum(1 for c in ctx.player.discard_pile if c == "wound"), 3)
        hp = ctx.player.hp
        ctx.player.hand = []
        end_turn(ctx)  # 10 x 4
        self.assertEqual(hp - ctx.player.hp, 40)

    def test_enrage_on_skills(self) -> None:
        ctx = make(["test_subject"], hand=["defend_ironclad"])
        ctx.play_card("defend_ironclad")
        self.assertEqual(m0(ctx).powers["strength"], 2)

    def test_galvanic_punishes_powers(self) -> None:
        ctx = make(["globe_head"], hand=["inflame"])
        ctx.play_card("inflame")
        self.assertEqual(ctx.player.hp, 100 - 6)

    def test_soar_halves_attack_damage(self) -> None:
        ctx = make(["owl_magistrate"], hand=["strike_ironclad"])
        m0(ctx).powers["soar"] = 1
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(m0(ctx).hp, 200 - 3)

    def test_possessed_strength_returns_on_death(self) -> None:
        ctx = make(["the_lost", "the_forgotten"], hand=[])
        end_turn(ctx)
        self.assertEqual(ctx.player.powers.get("strength"), -2)
        self.assertEqual(ctx.player.powers.get("dexterity"), -2)
        lost = m0(ctx)
        lost.hp = 1
        ctx.player.hand = ["strike_ironclad"]
        ctx.player.energy = 1
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertNotIn("strength", ctx.player.powers)
        self.assertEqual(ctx.player.powers.get("dexterity"), -2)

    def test_forgotten_dread_adds_dexterity(self) -> None:
        ctx = make(["the_forgotten"], hand=[])
        end_turn(ctx)  # MIASMA
        hp = ctx.player.hp
        end_turn(ctx)  # DREAD 13 + 2
        self.assertEqual(hp - ctx.player.hp, 15)

    def test_dampen_downgrades_until_magi_dies(self) -> None:
        ctx = make(["magi_knight"], hand=["strike_ironclad+1", "bash+1"], monster_hp=30)
        ctx.apply_power(ctx.player, "dampen", 1)
        self.assertEqual(ctx.player.hand, ["strike_ironclad", "bash"])
        m0(ctx).hp = 5
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertIn("bash+1", ctx.player.hand)

    def test_hex_makes_the_hand_ethereal(self) -> None:
        ctx = make(["spectral_knight"], hand=["strike_ironclad", "defend_ironclad"])
        ctx.player.powers["hex"] = 2
        end_turn(ctx)
        self.assertEqual(sorted(ctx.player.exhaust_pile), ["defend_ironclad", "strike_ironclad"])

    def test_chains_of_binding_bind_first_draws(self) -> None:
        ctx = make(["nibbit"], hand=[])
        ctx.player.powers["chains_of_binding"] = 3
        ctx.player.hand = []
        end_turn(ctx)
        bound = [c for c in ctx.player.hand if c.bound]
        self.assertEqual(len(bound), 3)
        ctx.player.energy = 5
        ctx.play_card(bound[0], target_slot=0)
        self.assertFalse(ctx.can_play(bound[1]))

    def test_queen_enrages_when_amalgam_dies_mid_burn(self) -> None:
        ctx = make(["torch_head_amalgam", "queen"], hand=["bludgeon"], monster_hp=20)
        queen = ctx.combat.monsters[1]
        queen.ai_state = queen.queued_move = "BURN_BRIGHT_FOR_ME"
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(queen.queued_move, "ENRAGE")
        self.assertIsNone(ctx.combat.outcome)

    def test_fabricator_builds_bots_into_slots(self) -> None:
        ctx = make(["fabricator"], hand=[])
        fab = m0(ctx)
        fab.ai_state = fab.queued_move = "FABRICATE"
        end_turn(ctx)
        ids = [m.monster_id for m in ctx.combat.monsters]
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[2], "fabricator")
        self.assertIn(ids[0], ("guardbot", "noisebot"))
        self.assertIn(ids[1], ("zapbot", "stabbot"))
        self.assertTrue(all(m.powers.get("minion") for m in ctx.combat.monsters[:2]))

    def test_rampart_shields_the_turret(self) -> None:
        ctx = make(["living_shield", "turret_operator"], hand=[])
        self.assertEqual(ctx.combat.monsters[1].block, 25)

    def test_paper_cuts_lower_max_hp(self) -> None:
        ctx = make(["scroll_of_biting"], hand=[], flags=[{"starter": 0}])
        end_turn(ctx)  # Chomp, unblocked
        self.assertEqual(ctx.player.max_hp, 98)

    def test_withering_presence_every_six_cards(self) -> None:
        ctx = make(["aeonglass"], hand=["defend_ironclad"] * 6, energy=6)
        for _ in range(6):
            ctx.play_card("defend_ironclad")
        self.assertIn("wither", ctx.player.hand)
        self.assertEqual(m0(ctx).powers["withering_presence"], 6)

    def test_nemesis_alternates_intangible(self) -> None:
        ctx = make(["test_subject"], hand=[])
        m = m0(ctx)
        m.powers.pop("adaptable")
        m.powers["nemesis"] = 1
        m.ai_state = m.queued_move = "BIG_POUNCE"
        end_turn(ctx)
        self.assertEqual(m.powers.get("intangible"), 1)
        ctx.player.hand = []
        end_turn(ctx)
        self.assertNotIn("intangible", m.powers)


if __name__ == "__main__":
    unittest.main()
