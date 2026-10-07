"""Phase 0 CPU smoke tests for the sts2 sim skeleton.

Covers only what Phase 0 built: RNG stream determinism and independence,
HookBus priority + registration order, EffectQueue FIFO/LIFO, and the
minimal reset -> step("skip") -> game_over run loop. Combat / map / cards
land in Phase 1+ with their own tests.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _import_sim():
    """Load `examples/classify/jev/sts2_sim/` as a stable module name.

    `examples` is not a package in this repo, so importlib.spec_from_file_location
    is the cleanest way to resolve a dotted name without touching sys.path. We
    load the package once per process; later tests reuse it.
    """

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


class RngTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_same_master_seed_reproduces_same_stream(self) -> None:
        a = self.sim.Rng(1234).stream("combat").random()
        b = self.sim.Rng(1234).stream("combat").random()
        self.assertEqual(a, b)

    def test_streams_are_independent(self) -> None:
        """combat shuffle must not drift when a new relic peeks `relics` stream."""

        rng = self.sim.Rng(42)
        combat_before = [rng.stream("combat").random() for _ in range(10)]

        rng2 = self.sim.Rng(42)
        for _ in range(5):
            rng2.stream("relics").random()  # unrelated draws
        combat_after = [rng2.stream("combat").random() for _ in range(10)]

        self.assertEqual(combat_before, combat_after)

    def test_different_masters_diverge(self) -> None:
        a = [self.sim.Rng(1).stream("x").random() for _ in range(5)]
        b = [self.sim.Rng(2).stream("x").random() for _ in range(5)]
        self.assertNotEqual(a, b)

    def test_fork_decouples_from_source(self) -> None:
        rng = self.sim.Rng(7)
        _ = [rng.stream("combat").random() for _ in range(3)]
        fork = rng.fork()
        next_in_source = rng.stream("combat").random()
        next_in_fork = fork.stream("combat").random()
        self.assertEqual(next_in_source, next_in_fork)
        # Advancing one side must not change the other.
        _ = rng.stream("combat").random()
        same_in_fork = fork.stream("combat").random()
        again_in_fork = fork.stream("combat").random()
        self.assertNotEqual(same_in_fork, again_in_fork)


class HookBusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_handlers_fire_in_priority_then_registration_order(self) -> None:
        bus = self.sim.HookBus()
        order: list[str] = []
        bus.register("evt", lambda _p: order.append("mid-a"), priority=100)
        bus.register("evt", lambda _p: order.append("mid-b"), priority=100)
        bus.register("evt", lambda _p: order.append("early"), priority=50)
        bus.register("evt", lambda _p: order.append("late"), priority=200)
        bus.dispatch("evt")
        self.assertEqual(order, ["early", "mid-a", "mid-b", "late"])

    def test_dispatch_passes_payload(self) -> None:
        bus = self.sim.HookBus()
        seen: list[dict] = []
        bus.register("evt", lambda payload: seen.append(payload))
        bus.dispatch("evt", {"dmg": 7})
        self.assertEqual(seen, [{"dmg": 7}])

    def test_unregister_removes_handler(self) -> None:
        bus = self.sim.HookBus()
        hits: list[int] = []
        off = bus.register("evt", lambda _p: hits.append(1))
        bus.dispatch("evt")
        off()
        bus.dispatch("evt")
        self.assertEqual(hits, [1])
        self.assertEqual(bus.handler_count("evt"), 0)

    def test_unknown_event_is_noop(self) -> None:
        bus = self.sim.HookBus()
        bus.dispatch("never-registered")  # must not raise
        self.assertEqual(bus.handler_count("never-registered"), 0)


class EffectQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_fifo_drain(self) -> None:
        q = self.sim.EffectQueue()
        q.enqueue(self.sim.Effect(name="a"))
        q.enqueue(self.sim.Effect(name="b"))
        q.enqueue(self.sim.Effect(name="c"))
        drained = []
        while len(q):
            drained.append(q.drain_one().name)
        self.assertEqual(drained, ["a", "b", "c"])
        self.assertIsNone(q.drain_one())

    def test_enqueue_front_jumps_queue(self) -> None:
        q = self.sim.EffectQueue()
        q.enqueue(self.sim.Effect(name="later"))
        q.enqueue_front(self.sim.Effect(name="first"))
        self.assertEqual(q.peek().name, "first")

    def test_clear_drops_everything(self) -> None:
        q = self.sim.EffectQueue()
        q.enqueue(self.sim.Effect(name="x"))
        q.enqueue(self.sim.Effect(name="y"))
        q.clear()
        self.assertEqual(len(q), 0)
        self.assertIsNone(q.peek())


class RunLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()

    def _loop(self, **kw):
        kw.setdefault("seed", 123)
        return self.sim.RunLoop(**kw)

    def test_reject_non_ironclad(self) -> None:
        with self.assertRaisesRegex(self.sim.RunLoopError, "character"):
            self._loop(character="silent")

    def test_reject_bad_ascension(self) -> None:
        with self.assertRaisesRegex(self.sim.RunLoopError, "ascension"):
            self._loop(ascension=99)

    def test_requires_reset_before_step_or_properties(self) -> None:
        loop = self._loop()
        with self.assertRaises(self.sim.RunLoopError):
            loop.step("skip")
        with self.assertRaises(self.sim.RunLoopError):
            _ = loop.state

    def test_reset_yields_neow_with_skip_candidate(self) -> None:
        loop = self._loop(ascension=0)
        packet = loop.reset()
        self.assertEqual(packet["screen"], self.sim.Screen.NEOW)
        self.assertEqual(packet["decision_point"], self.sim.DecisionPoint.NEOW_BONUS)
        self.assertFalse(packet["done"])
        self.assertEqual(packet["step"], 0)
        self.assertEqual([c["id"] for c in packet["candidates"]], ["skip"])

    def test_skip_transitions_to_combat(self) -> None:
        """Phase 1 behavior: skip Neow now enters a Jaw Worm combat."""

        loop = self._loop()
        loop.reset()
        packet = loop.step("skip")
        self.assertFalse(packet["done"])
        self.assertEqual(packet["screen"], self.sim.Screen.COMBAT)
        self.assertEqual(packet["decision_point"], self.sim.DecisionPoint.COMBAT_PLAY)
        # Candidates must include at least one play:* and end_turn.
        ids = {c["id"] for c in packet["candidates"]}
        self.assertTrue(any(i.startswith("play:") for i in ids))
        self.assertIn("end_turn", ids)

    def test_illegal_action_rejected(self) -> None:
        loop = self._loop()
        loop.reset()
        with self.assertRaisesRegex(self.sim.RunLoopError, "illegal action"):
            loop.step("not-a-real-move")

    def test_close_resets_run(self) -> None:
        loop = self._loop()
        loop.reset()
        loop.close()
        with self.assertRaises(self.sim.RunLoopError):
            _ = loop.state

    def test_reset_wires_subsystems(self) -> None:
        loop = self._loop()
        loop.reset()
        self.assertIsNotNone(loop.rng)
        self.assertIsNotNone(loop.effects)
        self.assertIsNotNone(loop.hooks)
        self.assertEqual(loop.state.player.hp, 80)
        self.assertEqual(loop.state.player.gold, 99)


if __name__ == "__main__":
    unittest.main()
