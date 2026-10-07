"""CPU test for the toy MDP that feeds `examples/classify/jev/train_rl.py`."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "classify" / "jev"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    module_name = f"classify_jev_{name}_for_tests"
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # @dataclass resolves string annotations via sys.modules; register before exec.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))


class ToyRLEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _load_module("toy_rl_env")

    def test_rollout_produces_three_steps_and_valid_terminal_reward(self):
        import random

        steps, terminal = self.env.rollout_episode(random.Random(0))
        self.assertEqual(len(steps), self.env.STEPS_PER_EPISODE)
        self.assertEqual(steps[0].sum_so_far, 0)
        self.assertEqual(steps[0].remaining, self.env.STEPS_PER_EPISODE)
        # Reward is non-positive, exactly `-|final_sum - target|`.
        final_sum = steps[-1].sum_so_far + self.env.ACTION_VALUES[steps[-1].chosen]
        self.assertEqual(terminal, -abs(final_sum - steps[0].target))

    def test_generate_decisions_rows_match_rl_trainer_schema(self):
        rows, stats = self.env.generate_decisions(32, seed=0, gamma=1.0)
        self.assertEqual(stats["decisions"], len(rows))
        self.assertEqual(stats["decisions"], 32 * self.env.STEPS_PER_EPISODE)
        required = {"prompt", "candidates", "chosen", "old_logp", "advantage"}
        for row in rows:
            self.assertTrue(required <= set(row))
            self.assertEqual(len(row["candidates"]), len(self.env.CANDIDATES))
            self.assertIn(row["chosen"], range(len(self.env.CANDIDATES)))
            self.assertAlmostEqual(row["old_logp"], -1.0986122886681098, places=6)  # log(1/3)

    def test_generate_decisions_advantage_centered_and_scaled(self):
        rows, _ = self.env.generate_decisions(64, seed=0, gamma=1.0)
        advantages = [row["advantage"] for row in rows]
        mean = sum(advantages) / len(advantages)
        self.assertAlmostEqual(mean, 0.0, places=6)
        var = sum(value * value for value in advantages) / len(advantages)
        self.assertAlmostEqual(var, 1.0, places=6)

    def test_write_then_load_jsonl_roundtrips(self):
        rows, _ = self.env.generate_decisions(8, seed=1, gamma=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            self.env.write_jsonl(rows, path)
            text = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(text), len(rows))
            reloaded = self.env.load_jsonl(path)
            self.assertEqual(reloaded, rows)
            # Each line parses back to the exact same dict (JSON preserves order via dumps/loads).
            self.assertEqual(json.loads(text[0]), rows[0])

    def test_rows_are_consumable_by_the_rl_trainer_encoder(self):
        from areno.experimental.classify.rl_trainer import encode_decision

        rows, _ = self.env.generate_decisions(4, seed=0, gamma=1.0)

        class _Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(char) % 128 for char in text]

        tok = _Tokenizer()
        for row in rows:
            decision = encode_decision(row, tok)
            self.assertEqual(len(decision.leaves), len(self.env.CANDIDATES))
            self.assertEqual(decision.chosen, row["chosen"])
            self.assertIsNone(decision.sft_target)


if __name__ == "__main__":
    unittest.main()
