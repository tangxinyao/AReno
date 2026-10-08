"""sts2_sim full-run tests: map generation, act rooms, rewards, rest sites,
shops, treasure and act transitions (CPU only).

Rules follow r33hab/sts2 `RunMapGenerator`, `RunRewardGenerator` and
`RunEngine` for STS2 v0.107.1.
"""

from __future__ import annotations

import collections
import random
import unittest

from tests.test_classify_jev_sts2_sim_combat_cpu import SIM

mapgen = SIM.mapgen
rewards = SIM.rewards


def new_run(seed: int = 1, ascension: int = 0):
    loop = SIM.RunLoop(seed=seed, ascension=ascension)
    loop.reset()
    return loop


def win_combat(loop) -> dict:
    """Kill every monster outright and let the run finalize the fight."""

    combat = loop.state.combat
    for m in combat.monsters:
        m.hp = 1
        m.block = 0
        m.powers = {}
    loop.state.player.hand[:] = ["whirlwind"]
    loop.state.player.energy = 3
    return loop.step("play_card:whirlwind")


class MapGenerationTest(unittest.TestCase):
    def test_shape_rules(self) -> None:
        for act in ("overgrowth", "underdocks", "hive", "glory"):
            for seed in range(12):
                m = mapgen.generate_act_map(act, 0, random.Random(seed))
                boss_row = mapgen.boss_row_for(act)
                self.assertEqual(m.boss_row, boss_row)
                kinds = collections.Counter(n.kind for n in m.nodes.values())
                self.assertEqual(kinds[mapgen.SHOP], 3)
                self.assertEqual(kinds[mapgen.ELITE], 5)
                self.assertEqual(kinds[mapgen.BOSS], 1)
                self.assertEqual(kinds[mapgen.ANCIENT], 1)
                for n in m.nodes.values():
                    if n.row == 1:
                        self.assertEqual(n.kind, mapgen.MONSTER)
                    if n.row == boss_row - 7:
                        self.assertEqual(n.kind, mapgen.TREASURE)
                    if n.row == boss_row - 1:
                        self.assertEqual(n.kind, mapgen.REST)
                    if n.kind in (mapgen.REST, mapgen.ELITE) and n.row != boss_row - 1:
                        self.assertGreaterEqual(n.row, 6)
                    for c in n.children:
                        self.assertEqual(c[1], n.row + 1)
                        if n.row > 0 and m.nodes[c].kind != mapgen.BOSS:
                            self.assertLessEqual(abs(c[0] - n.col), 1)
                        child = m.nodes[c]
                        if n.kind in (mapgen.ELITE, mapgen.REST, mapgen.SHOP, mapgen.TREASURE) and n.row > 0:
                            if not (n.kind == mapgen.REST and n.row == boss_row - 1):
                                self.assertNotEqual(child.kind, n.kind, (act, seed, n.coord, c))
                # Every point is reachable from the Ancient and leads to the boss.
                seen, todo = set(), [m.start]
                while todo:
                    x = todo.pop()
                    if x not in seen:
                        seen.add(x)
                        todo.extend(m.nodes[x].children)
                self.assertEqual(seen, set(m.nodes))

    def test_swarming_elites(self) -> None:
        m = mapgen.generate_act_map("overgrowth", 1, random.Random(3))
        self.assertEqual(sum(1 for n in m.nodes.values() if n.kind == mapgen.ELITE), 8)

    def test_double_boss_point(self) -> None:
        m = mapgen.generate_act_map("glory", 10, random.Random(3), second_boss=True)
        bosses = sorted(n.row for n in m.nodes.values() if n.kind == mapgen.BOSS)
        self.assertEqual(bosses, [m.boss_row, m.boss_row + 1])


class ActRoomsTest(unittest.TestCase):
    def test_weak_fights_come_first_and_tags_never_repeat(self) -> None:
        for seed in range(10):
            loop = new_run(seed, ascension=10)
            acts = loop.state.acts
            self.assertIn(acts[0].act, ("overgrowth", "underdocks"))
            self.assertEqual([a.act for a in acts[1:]], ["hive", "glory"])
            for rooms in acts:
                weak, total = mapgen.ACT_ROOMS[rooms.act]
                self.assertEqual(len(rooms.normal), total)
                self.assertTrue(all(loop.encounters[e].pool == "weak" for e in rooms.normal[:weak]))
                self.assertTrue(all(loop.encounters[e].pool == "normal" for e in rooms.normal[weak:]))
                self.assertEqual(len(rooms.elite), 15)
                for seq in (rooms.normal, rooms.elite):
                    for a, b in zip(seq, seq[1:]):
                        self.assertNotEqual(a, b)
                        self.assertFalse(set(loop.encounters[a].tags) & set(loop.encounters[b].tags))
            self.assertIsNotNone(acts[2].second_boss)
            self.assertNotEqual(acts[2].second_boss, acts[2].boss)
            self.assertIsNone(acts[0].second_boss)


class StartTest(unittest.TestCase):
    def test_ascension_start(self) -> None:
        loop = new_run(ascension=0)
        p = loop.state.player
        self.assertEqual((p.hp, p.max_hp, p.gold, len(p.potions)), (80, 80, 99, 3))
        self.assertEqual(p.relics, ["burning_blood"])
        loop = new_run(ascension=5)
        p = loop.state.player
        self.assertEqual(p.hp, 64)  # Weary Traveler: the Ancient heals 80% of the missing HP
        self.assertEqual(len(p.potions), 2)  # Tight Belt
        self.assertIn("ascenders_bane", p.deck)


class RewardRollTest(unittest.TestCase):
    def test_gold_ranges(self) -> None:
        rng = random.Random(1)
        normal = {rewards.combat_gold("monster", 0, rng) for _ in range(500)}
        self.assertEqual((min(normal), max(normal)), (10, 20))
        elite = {rewards.combat_gold("elite", 0, rng) for _ in range(500)}
        self.assertEqual((min(elite), max(elite)), (35, 45))
        self.assertEqual(rewards.combat_gold("boss", 0, rng), 100)
        self.assertEqual(rewards.combat_gold("boss", 3, rng), 75)  # Poverty

    def test_rarity_offset_pity(self) -> None:
        rng = random.Random(0)
        offset = rewards.CARD_RARITY_BASE_OFFSET
        rarity, offset = rewards.roll_rarity(offset, (0.0, 0.0), rng, ascension=0, mutate=True)
        self.assertEqual(rarity, "common")
        self.assertAlmostEqual(offset, -0.04)
        rarity, offset = rewards.roll_rarity(0.4, (1.0, 0.0), rng, ascension=0, mutate=True)
        self.assertEqual(rarity, "rare")
        self.assertAlmostEqual(offset, rewards.CARD_RARITY_BASE_OFFSET)

    def test_boss_card_reward_is_all_rare(self) -> None:
        loop = new_run()
        cards, _ = rewards.card_reward(loop.cards, loop._reward_pool, "boss", ascension=0, act_index=0,
                                       offset=0.0, rng=random.Random(4))
        self.assertEqual(len(set(cards)), 3)
        self.assertTrue(all(loop.cards[c].rarity == "rare" for c in cards))

    def test_upgrade_odds_by_act(self) -> None:
        loop = new_run()
        card = loop.cards["pommel_strike"]
        rng = random.Random(2)
        self.assertFalse(any(rewards.roll_upgrade(card, 0, 0, rng) for _ in range(200)))
        hits = sum(rewards.roll_upgrade(card, 2, 0, rng) for _ in range(4000))
        self.assertAlmostEqual(hits / 4000, 0.5, delta=0.04)
        rare = loop.cards["demon_form"]
        self.assertEqual(rare.rarity, "rare")
        self.assertFalse(any(rewards.roll_upgrade(rare, 2, 0, rng) for _ in range(200)))


class RunFlowTest(unittest.TestCase):
    def test_first_fight_rewards_and_card_pick(self) -> None:
        loop = new_run(seed=2)
        loop.step("choose_event_option:0")
        loop.step("choose_map_node:0")
        self.assertEqual(loop.state.room, "monster")
        self.assertEqual(loop.encounters[loop.state.encounter_id].pool, "weak")
        hp_before = loop.state.player.hp
        packet = win_combat(loop)
        self.assertEqual(packet["decision_point"], "rewards")
        self.assertEqual(loop.state.player.hp, min(80, hp_before + 6))  # Burning Blood
        kinds = [r.kind for r in loop.state.rewards]
        self.assertEqual(kinds[0], "gold")
        self.assertEqual(kinds[-1], "card")
        gold = loop.state.player.gold
        amount = loop.state.rewards[0].gold
        loop.step("claim_reward:0")
        self.assertEqual(loop.state.player.gold, gold + amount)
        card_index = len(loop.state.rewards) - 1
        packet = loop.step(f"claim_reward:{card_index}")
        self.assertEqual(packet["decision_point"], "card_reward")
        self.assertEqual(len(packet["candidates"]), 4)
        offered = list(loop.state.card_reward)
        deck = len(loop.state.player.deck)
        loop.step("select_card_reward:1")
        self.assertEqual(loop.state.player.deck[-1], offered[1])
        self.assertEqual(len(loop.state.player.deck), deck + 1)
        packet = loop.step("proceed")
        self.assertEqual(packet["decision_point"], "map_select")
        self.assertEqual(loop.state.floor, 2)

    def test_rest_site_heal_and_smith(self) -> None:
        loop = new_run(seed=3)
        loop.step("choose_event_option:0")
        st = loop.state
        st.room = "rest"
        st.rest_used = False
        st.screen = SIM.Screen.REST
        st.player.hp = 40
        packet = loop._packet()
        self.assertEqual([c["text"] for c in packet["candidates"]],
                         ["Rest: Heal for 30% of your Max HP (24).", "Smith: Upgrade a card in your Deck."])
        loop.step("choose_rest_option:0")
        self.assertEqual(st.player.hp, 64)
        self.assertEqual([c["id"] for c in loop._packet()["candidates"]], ["proceed"])
        st.rest_used = False
        packet = loop.step("choose_rest_option:1")
        self.assertEqual(packet["decision_point"], "card_select")
        bash = next(i for i, c in enumerate(packet["candidates"]) if "Bash" in c["text"])
        loop.step(f"select_card:{bash}")
        loop.step("confirm_selection")
        self.assertIn("bash+1", st.player.deck)
        self.assertNotIn("bash", st.player.deck)

    def test_shop_stock_and_removal(self) -> None:
        loop = new_run(seed=4)
        loop.step("choose_event_option:0")
        st = loop.state
        st.player.gold = 999
        loop._enter_shop()
        cards = [i for i in st.shop if i.category == "card"]
        self.assertEqual([loop.cards[i.item].card_type for i in cards],
                         ["attack", "attack", "skill", "skill", "power"])
        self.assertEqual(sum(i.on_sale for i in cards), 1)
        removal = next(i for i in st.shop if i.category == "card_removal")
        self.assertEqual(removal.price, 75)
        for i in cards:
            base = {"rare": 150, "uncommon": 75, "common": 50}[loop.cards[i.item].rarity]
            lo, hi = round(base * 0.95), round(base * 1.05)
            if i.on_sale:
                lo, hi = lo // 2, hi // 2
            self.assertTrue(lo <= i.price <= hi, (i, lo, hi))
        packet = loop._packet()
        idx = next(j for j, c in enumerate(packet["candidates"]) if "card_removal" in c["text"])
        loop.step(f"shop_purchase:{idx}")
        strike = next(j for j, c in enumerate(loop._packet()["candidates"]) if "Strike" in c["text"])
        deck = len(st.player.deck)
        loop.step(f"select_card:{strike}")
        loop.step("confirm_selection")
        self.assertEqual(len(st.player.deck), deck - 1)
        self.assertEqual(st.player.gold, 999 - 75)
        self.assertEqual(st.removals_used, 1)
        self.assertFalse(any("card_removal" in c["text"] for c in loop._packet()["candidates"]))

    def test_unknown_room_odds_reset_and_grow(self) -> None:
        loop = new_run(seed=5)
        st = loop.state
        st.unknown_odds = {"monster": 0.9, "elite": -1.0, "treasure": 0.02, "shop": 0.03}
        st.map_coord = st.map.start
        st.last_room = None
        self.assertEqual(loop._roll_unknown(), "monster")
        # Elite stays out of reach: its -1 grows by its own (negative) base, as in the game.
        self.assertEqual(st.unknown_odds, {"monster": 0.1, "elite": -2.0, "treasure": 0.04, "shop": 0.06})

    def test_boss_win_moves_to_the_next_act(self) -> None:
        loop = new_run(seed=6, ascension=2)
        loop.step("choose_event_option:0")
        st = loop.state
        boss = next(n for n in st.map.nodes.values() if n.kind == mapgen.BOSS)
        st.map_coord = boss.parents[0]
        loop.step("choose_map_node:0")
        self.assertEqual(st.room, "boss")
        self.assertEqual(st.encounter_id, st.acts[0].boss)
        win_combat(loop)
        self.assertEqual(st.rewards[0].gold, 100)
        cards = st.rewards[-1].cards
        self.assertTrue(all(loop.cards[c].rarity == "rare" for c in cards))
        packet = loop.step("proceed")
        self.assertEqual(st.act, 2)
        self.assertEqual(st.map.act, "hive")
        self.assertEqual(packet["decision_point"], "map_select")
        self.assertEqual([c["text"].split(" (")[0] for c in packet["candidates"]], ["go to Ancient"])
        st.player.hp = 30
        loop.step("choose_map_node:0")
        self.assertEqual(st.player.hp, 30 + int(50 * 0.8))
        loop.step("choose_event_option:0")
        packet = loop._packet()
        self.assertTrue(all(c["text"].startswith("go to Monster (row 1") for c in packet["candidates"]))
        loop.step("choose_map_node:0")
        self.assertEqual(loop.encounters[st.encounter_id].act, "hive")
        self.assertEqual(loop.encounters[st.encounter_id].pool, "weak")

    def test_final_boss_ends_the_run(self) -> None:
        loop = new_run(seed=7, ascension=10)
        loop.step("choose_event_option:0")
        st = loop.state
        st.acts = st.acts  # rolled up front
        loop._enter_act_map(2)
        boss = min((n for n in st.map.nodes.values() if n.kind == mapgen.BOSS), key=lambda n: n.row)
        st.map_coord = boss.parents[0]
        loop.step("choose_map_node:0")
        self.assertEqual(st.encounter_id, st.acts[2].boss)
        packet = win_combat(loop)
        self.assertFalse(packet["done"])  # A10: the second boss is next
        self.assertEqual(packet["decision_point"], "map_select")
        loop.step("choose_map_node:0")
        self.assertEqual(st.encounter_id, st.acts[2].second_boss)
        packet = win_combat(loop)
        self.assertTrue(packet["done"])
        self.assertEqual(packet["outcome"], SIM.Outcome.VICTORY)

    def test_thieving_hopper_escape_loses_the_card(self) -> None:
        loop = new_run(seed=8)
        loop.step("choose_event_option:0")
        st = loop.state
        st.player.deck.append("uppercut")
        loop._start_encounter("thieving_hopper_weak", "monster")
        hopper = st.combat.monsters[0]
        st.combat.stolen.append((hopper, "uppercut"))
        hopper.escaped = True
        st.player.hand[:] = []
        loop.combat_ctx._check_end()
        loop._maybe_finalize_combat()
        self.assertNotIn("uppercut", st.player.deck)


if __name__ == "__main__":
    unittest.main()
