"""Guide knowledge base: sim consistency, candidate scoring, and labeled records."""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JEV = ROOT / "examples" / "classify" / "jev"
STRATEGY = JEV / "strategy"
for path in (str(ROOT), str(JEV), str(STRATEGY)):
    if path not in sys.path:
        sys.path.insert(0, path)

import knowledge  # noqa: E402
import label_from_kb  # noqa: E402
from dataset_loader import load_questions, record_to_questions  # noqa: E402


def _card(cid: str, text: str) -> dict:
    return {"id": cid, "text": text}


class KnowledgeBaseTest(unittest.TestCase):
    def test_every_name_resolves_against_sim_data(self) -> None:
        self.assertEqual(knowledge.validate(), [])

    def test_card_name_parsing_drops_upgrade_and_enchant(self) -> None:
        self.assertEqual(knowledge.card_name_from_text("take Bash+ (attack, cost 2): Deal 8 damage."), "Bash")
        self.assertEqual(knowledge.card_name_from_text("take Anger (attack, cost 0): x"), "Anger")
        self.assertEqual(
            knowledge.card_name_from_text("take Anger+ [Sharp 3] (attack, cost 0): x"), "Anger"
        )
        self.assertIsNone(knowledge.card_name_from_text("skip card reward"))

    def test_relic_name_parsing(self) -> None:
        self.assertEqual(knowledge.relic_name_from_text("take Pael's Legion: Every two turns ..."), "Pael's Legion")
        self.assertIsNone(knowledge.relic_name_from_text("leave ancient"))


class CandidateScoresTest(unittest.TestCase):
    def test_card_reward_uses_tiers_skip_and_neutral_unknown(self) -> None:
        candidates = [
            _card("select_card_reward:0", "take Anger (attack, cost 0): x"),
            _card("select_card_reward:1", "take Cinder (attack, cost 2): x"),
            _card("skip_card_reward", "skip card reward"),
        ]
        scores, covered, eligible = knowledge.candidate_scores("card_reward", candidates)
        self.assertEqual(scores, [3.0, knowledge.UNKNOWN_SCORE, knowledge.SKIP_CARD_SCORE])
        self.assertEqual((covered, eligible), (1, 2))

    def test_unknown_is_not_tier_zero(self) -> None:
        # Tier 0 means "do not take early"; an uncovered card must not score like it.
        self.assertNotEqual(knowledge.UNKNOWN_SCORE, 0.0)

    def test_ancient_choice_uses_relic_tiers(self) -> None:
        candidates = [
            _card("select_relic:0", "take Golden Compass: x"),
            _card("select_relic:1", "take Lava Rock: x"),
            _card("skip_relic_selection", "leave ancient"),
        ]
        scores, covered, eligible = knowledge.candidate_scores("event_choice", candidates)
        self.assertEqual(scores[0], 3.0)
        self.assertEqual(scores[1], -1.0)
        self.assertEqual(scores[2], knowledge.LEAVE_RELIC_SCORE)
        self.assertEqual((covered, eligible), (2, 2))

    def test_other_decisions_are_not_labeled(self) -> None:
        candidates = [_card("play_card:0", "Strike"), _card("end_turn", "end turn")]
        self.assertIsNone(knowledge.candidate_scores("combat_play", candidates))


class RecordTest(unittest.TestCase):
    def test_soft_targets_sum_to_one_and_follow_scores(self) -> None:
        probs = label_from_kb.soft_targets([3.0, 1.0, 0.0], temperature=1.0)
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

    def test_record_round_trips_through_dataset_loader(self) -> None:
        candidates = [
            _card("select_card_reward:0", "take Anger (attack, cost 0): x"),
            _card("skip_card_reward", "skip card reward"),
        ]
        scores, covered, _ = knowledge.candidate_scores("card_reward", candidates)
        record = label_from_kb.make_record(
            "t:0", "train", "state text", "card_reward", candidates, scores, covered, 1.0
        )
        rows = record_to_questions(record)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["type"], "choice")
        self.assertEqual(len(row["candidates"]), 2)
        self.assertAlmostEqual(sum(row["target"]), 1.0)
        self.assertGreater(row["target"][0], row["target"][1])


class LabelerEndToEndTest(unittest.TestCase):
    def test_short_run_writes_loadable_records_and_strict_is_a_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mixed"
            strict_out = Path(tmp) / "strict"
            for target, extra in ((out, []), (strict_out, ["--strict"])):
                argv = [
                    "label_from_kb.py", "--episodes", "3", "--dev-every", "0",
                    "--rng-seed", "5", "--out-dir", str(target), *extra,
                ]
                old = sys.argv
                sys.argv = argv
                try:
                    label_from_kb.main()
                finally:
                    sys.argv = old

            mixed = load_questions(out, "train")
            strict = load_questions(strict_out, "train")
            self.assertGreater(len(mixed), 0)
            self.assertLessEqual(len(strict), len(mixed))
            for row in mixed + strict:
                self.assertAlmostEqual(sum(row["target"]), 1.0, places=6)
                self.assertTrue(all(math.isfinite(t) for t in row["target"]))

            for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self.assertEqual(record["split"], "train")
                self.assertEqual(set(record["targets"]["q"]), set(record["request"]["questions"]["q"]["criteria"]))


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
