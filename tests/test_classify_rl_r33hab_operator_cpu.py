"""CPU test for the r33hab/sts2 operator shim.

We don't require the NativeAOT library to be built; instead we inject a
fake `sts2_gym` module via sys.modules before importing the shim, so the
shim's shape and dispatch logic can be exercised without the real emulator.
Build-time integration against the real r33hab backend is verified by
running `examples/classify/jev/operators/demo_r33hab.py`.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


class _FakeRunEnv:
    """Mimics the parts of Sts2RunEnv the shim calls."""

    def __init__(self, *, seed, ascension, max_episode_steps):
        self._seed = seed
        self._ascension = ascension
        self._max_episode_steps = max_episode_steps
        self._script = []
        self._cursor = 0

    def set_script(self, packets):
        """packets: list of (obs, info, reward, terminated, truncated, mask)."""

        self._script = list(packets)
        self._cursor = 0

    def reset(self):
        self._cursor = 0
        self._current = self._script[0]
        return self._current[0], self._current[1]

    def step(self, action, target=-1):
        del target
        self._cursor += 1
        self._current = self._script[self._cursor]
        obs, info, reward, terminated, truncated, _ = self._current
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        return self._current[5]

    def close(self):
        pass


def _install_fake_sts2_gym():
    module = types.ModuleType("sts2_gym")

    def _factory(**kwargs):
        return _FakeRunEnv(**kwargs)

    module.Sts2RunEnv = _factory
    constants = types.ModuleType("sts2_gym.run_constants")
    constants.PHASE_COMBAT = 0
    constants.PHASE_CARD_REWARD = 1
    constants.PHASE_MAP = 2
    constants.PHASE_REST = 3
    constants.PHASE_SHOP = 4
    constants.PHASE_RELIC_REWARD = 5
    constants.PHASE_COMPLETE = 6
    constants.PHASE_EVENT = 7
    constants.PHASE_ANCIENT = 8
    constants.PHASE_TRANSFORM_SELECT = 9
    constants.PHASE_TREASURE = 10
    constants.PHASE_CRYSTAL_SPHERE = 11
    constants.PHASE_BUNDLE_SELECT = 12
    module.run_constants = constants
    sys.modules["sts2_gym"] = module
    sys.modules["sts2_gym.run_constants"] = constants
    return module


def _import_shim():
    """Fresh import so the fake sts2_gym is picked up."""

    sys.modules.pop("examples.classify.jev.operators.r33hab_sts2", None)
    sys.modules.pop("classify_jev_r33hab_sts2_for_tests", None)
    path = Path(__file__).resolve().parents[1] / "examples" / "classify" / "jev" / "operators" / "r33hab_sts2.py"
    module_name = "classify_jev_r33hab_sts2_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class R33habOperatorTest(unittest.TestCase):
    def setUp(self):
        _install_fake_sts2_gym()
        # Also reset the shim module-level cache so each test starts clean.
        import examples.classify.jev.operators.r33hab_sts2 as shim_module  # noqa: F401
        self.shim_module = _import_shim()
        # Reset the shim's _PHASE_* lazy tables so re-initialization uses our constants.
        self.shim_module._PHASE_TO_DECISION_POINT = None
        self.shim_module._PHASE_NAMES = None

    def _mask(self, legal_actions, size=16):
        return [i in legal_actions for i in range(size)]

    def _make_backend(self):
        return self.shim_module.R33habSts2Backend(max_episode_steps=50)

    def test_capabilities_declares_ironclad_and_bit_exact(self):
        caps = self._make_backend().capabilities()
        self.assertEqual(caps["backend"], "r33hab-sts2")
        self.assertEqual(caps["characters"], ["ironclad"])
        self.assertTrue(caps["paired_seed_bit_exact"])
        self.assertIn("full_run", caps["episode_scopes"])

    def test_reset_rejects_non_ironclad_or_battle_scope(self):
        backend = self._make_backend()
        with self.assertRaisesRegex(ValueError, "ironclad"):
            backend.reset(
                {"character": "silent", "ascension": 0, "episode_scope": "full_run"}
            )
        with self.assertRaisesRegex(ValueError, "full-run"):
            backend.reset(
                {"character": "ironclad", "ascension": 0, "episode_scope": "battle"}
            )
        with self.assertRaisesRegex(ValueError, "ascension"):
            backend.reset(
                {"character": "ironclad", "ascension": 99, "episode_scope": "full_run"}
            )

    def test_full_reset_step_loop_matches_schema(self):
        """Patch Sts2RunEnv to be our _FakeRunEnv so we can drive reset -> step*3."""

        script = [
            # (obs, info, reward, terminated, truncated, mask)
            (
                None,
                {
                    "phase": 8,  # PHASE_ANCIENT
                    "player_hp": 80,
                    "player_max_hp": 80,
                    "floor": 1,
                    "act": "overgrowth",
                    "gold": 99,
                    "deck_size": 10,
                    "neow_options": (206, 167, 240),
                },
                0.0,
                False,
                False,
                self._mask({0, 1, 2}),
            ),
            (
                None,
                {
                    "phase": 2,  # PHASE_MAP
                    "player_hp": 80,
                    "player_max_hp": 80,
                    "floor": 1,
                    "act": "overgrowth",
                    "gold": 99,
                    "deck_size": 10,
                    "map_choices": ({"node_type": 1, "x": 0, "y": 0}, {"node_type": 1, "x": 1, "y": 0}),
                },
                0.0,
                False,
                False,
                self._mask({0, 1}),
            ),
            (
                None,
                {
                    "phase": 0,  # PHASE_COMBAT
                    "player_hp": 80,
                    "player_max_hp": 80,
                    "floor": 2,
                    "act": "overgrowth",
                    "gold": 99,
                    "deck_size": 10,
                    "encounter": "cultist",
                },
                1.0,
                False,
                False,
                self._mask({3, 7}),
            ),
            (
                None,
                {
                    "phase": 6,  # PHASE_COMPLETE
                    "player_hp": 0,
                    "player_max_hp": 80,
                    "floor": 2,
                    "act": "overgrowth",
                    "gold": 99,
                    "deck_size": 10,
                    "player_won": False,
                },
                -5.0,
                True,
                False,
                self._mask(set()),
            ),
        ]

        # Patch Sts2RunEnv so the shim constructs our fake and we can seed the script.
        envs: list[_FakeRunEnv] = []

        def _factory(**kwargs):
            env = _FakeRunEnv(**kwargs)
            env.set_script(script)
            envs.append(env)
            return env

        sys.modules["sts2_gym"].Sts2RunEnv = _factory
        self.shim_module._load_sts2_gym = lambda: (_factory, sys.modules["sts2_gym.run_constants"])

        backend = self._make_backend()
        reset = backend.reset(
            {"character": "ironclad", "ascension": 0, "episode_scope": "full_run", "seed": "DEMO01"}
        )
        self.assertEqual(reset["decision_point"], "neow_bonus")
        self.assertEqual(reset["step"], 0)
        self.assertFalse(reset["done"])
        self.assertEqual([c["id"] for c in reset["candidates"]], ["0", "1", "2"])
        self.assertEqual(reset["info"]["backend"], "r33hab-sts2")
        self.assertEqual(reset["info"]["seed"], "DEMO01")
        self.assertEqual(reset["info"]["character"], "ironclad")
        self.assertIn("neow", reset["state_text"])

        episode_id = reset["episode_id"]

        step1 = backend.step({"episode_id": episode_id, "action_id": "0"})
        self.assertEqual(step1["decision_point"], "map_select")
        self.assertEqual(step1["step"], 1)
        self.assertEqual([c["id"] for c in step1["candidates"]], ["0", "1"])

        step2 = backend.step({"episode_id": episode_id, "action_id": "1"})
        self.assertEqual(step2["decision_point"], "combat_play")
        self.assertAlmostEqual(step2["reward"], 1.0)
        # combat preserves non-contiguous legal action ids.
        self.assertEqual([c["id"] for c in step2["candidates"]], ["3", "7"])

        step3 = backend.step({"episode_id": episode_id, "action_id": "3"})
        self.assertTrue(step3["done"])
        self.assertEqual(step3["decision_point"], "game_over")
        self.assertEqual([c["id"] for c in step3["candidates"]], ["terminal"])
        self.assertEqual(step3["info"]["outcome"], "loss")

        backend.close(episode_id)

    def test_step_rejects_bad_action_or_unknown_episode(self):
        backend = self._make_backend()
        with self.assertRaisesRegex(ValueError, "unknown episode_id"):
            backend.step({"episode_id": "nope", "action_id": "0"})

        # Need a real reset to get an episode id.
        script = [
            (None, {"phase": 8, "player_hp": 80, "player_max_hp": 80, "floor": 1, "act": "overgrowth",
                    "gold": 0, "deck_size": 10, "neow_options": (1, 2, 3)},
             0.0, False, False, self._mask({0, 1, 2})),
        ]

        def _factory(**kwargs):
            env = _FakeRunEnv(**kwargs)
            env.set_script(script)
            return env

        sys.modules["sts2_gym"].Sts2RunEnv = _factory
        self.shim_module._load_sts2_gym = lambda: (_factory, sys.modules["sts2_gym.run_constants"])
        backend = self._make_backend()
        reset = backend.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run"})

        with self.assertRaisesRegex(ValueError, "integer"):
            backend.step({"episode_id": reset["episode_id"], "action_id": "not-an-int"})


if __name__ == "__main__":
    unittest.main()
