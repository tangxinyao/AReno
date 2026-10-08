"""The server policy must score the same question that goes into the training prompt.

Regression for a bug where `_ServerPolicy.pick` built its question with an
empty decision_point, so serve_decisions scored "Pick an action at ." while
the saved prompt said "Pick an action in combat ...", making old_logp
inconsistent with training-time scoring.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path


def _import_rollout():
    mod_name = "classify_jev_rollout_sts2_sim_for_tests"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = Path(__file__).resolve().parents[1] / "examples" / "classify" / "jev" / "rollout_sts2_sim.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


class ServerPolicyQuestionTest(unittest.TestCase):
    def test_scored_question_matches_training_prompt(self) -> None:
        rollout = _import_rollout()
        seen: list[tuple[str, dict]] = []

        class _FakeClient:
            def logits(self, state_text, question):
                seen.append((state_text, question))
                return [0.0] * len(question["criteria"])

        policy = rollout._ServerPolicy("http://unused", temperature=1.0, model_name="t", timeout=1.0)
        policy._client = _FakeClient()
        records, _ = rollout.run_episode(
            rollout.Sts2SimBackend(), "SERVER-Q-TEST",
            policy="server", server_policy=policy, max_steps=50, rng=random.Random(0),
        )
        self.assertTrue(records)
        combat_questions = [q for _, q in seen if len(q["criteria"]) >= 2]
        self.assertTrue(combat_questions)
        for state_text, question in seen:
            self.assertNotIn("Pick an action at", question["instructions"])
        # Every multi-candidate record's prompt was rendered from a question the server saw.
        for record, (state_text, question) in zip(records, [s for s in seen if len(s[1]["criteria"]) >= 2]):
            expected_prompt, _ = rollout.render(state_text, question)
            self.assertEqual(record.prompt, expected_prompt)


if __name__ == "__main__":
    unittest.main()
