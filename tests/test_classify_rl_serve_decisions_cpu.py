"""CPU tests for serve_decisions ServedModel + /admin/reload + logits exposure.

These bypass the real DecisionModel: FastAPI's TestClient drives the route
with a stub factory, which lets us check the lock/reload semantics and the
response schema without loading any checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "classify" / "jev"


def _load_serve_decisions():
    """Load serve_decisions.py without executing main()."""

    module_name = "classify_jev_serve_decisions_for_tests"
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_DIR / "serve_decisions.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))


class _StubModel:
    """Mimics DecisionModel.decide without any torch/model dependency."""

    def __init__(self, checkpoint: str, opts: dict, logits_by_qid: dict | None = None):
        self.checkpoint = checkpoint
        self.temperature = float(opts["temperature"])
        self.calls = 0
        self._logits_by_qid = logits_by_qid or {}

    def decide(self, state: str, questions: dict):
        import math

        self.calls += 1
        answers = {}
        total_tokens = 0
        for qid, question in questions.items():
            kind = question["type"]
            if kind == "choice":
                ids = list(question["criteria"])
            elif kind == "score":
                ids = [str(i) for i in range(len(question["criteria"]))]
            else:
                ids = ["false", "true"]
            raw = self._logits_by_qid.get(qid, [0.0] * len(ids))
            scaled = [v / self.temperature for v in raw]
            m = max(scaled) if scaled else 0.0
            exps = [math.exp(v - m) for v in scaled]
            z = sum(exps) or 1.0
            probs = [v / z for v in exps]
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": probs[1], "logits": {"false": raw[0], "true": raw[1]}}
            elif kind == "choice":
                best = max(range(len(ids)), key=probs.__getitem__)
                answers[qid] = {
                    "type": "choice",
                    "choice": ids[best],
                    "probabilities": dict(zip(ids, probs, strict=True)),
                    "confidence": max(0.0, (max(probs) - 1 / len(ids)) * len(ids) / (len(ids) - 1)),
                    "logits": dict(zip(ids, raw, strict=True)),
                }
            else:
                answers[qid] = {
                    "type": "score",
                    "score": sum(i * v for i, v in enumerate(probs)),
                    "legend": {str(i): text for i, text in enumerate(question["criteria"])},
                    "probabilities": {str(i): v for i, v in enumerate(probs)},
                    "confidence": max(0.0, (max(probs) - 1 / len(ids)) * len(ids) / (len(ids) - 1)),
                    "logits": {str(i): raw[i] for i in range(len(raw))},
                }
            total_tokens += len(ids) * 8  # dummy
        return answers, total_tokens


class ServedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_serve_decisions()

    def _make(self, *, logits_by_qid=None, temperature=1.0):
        instances = []

        def factory(checkpoint, opts):
            instance = _StubModel(checkpoint, opts, logits_by_qid=logits_by_qid)
            instances.append(instance)
            return instance

        served = self.module.ServedModel(
            factory=factory,
            checkpoint="/fake/step_000001",
            model_name="stub-v1",
            model_opts={"max_length": 512, "temperature": temperature, "attn_backend": "native", "max_tokens": 1024},
        )
        return served, instances

    def test_decide_delegates_and_bumps_calls(self):
        served, instances = self._make(logits_by_qid={"q": [2.0, 0.0]})
        answers, _ = served.decide(
            "state",
            {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}},
        )
        self.assertEqual(answers["q"]["choice"], "a")
        self.assertEqual(answers["q"]["logits"], {"a": 2.0, "b": 0.0})
        self.assertEqual(instances[0].calls, 1)
        self.assertEqual(served.model_info()["model"], "stub-v1")

    def test_reload_atomically_swaps_and_counts(self):
        served, instances = self._make()
        info = served.reload(
            "/fake/step_000400",
            overrides={"temperature": 1.5},
            model_name="stub-v2",
        )
        self.assertEqual(info["model"], "stub-v2")
        self.assertEqual(info["temperature"], 1.5)
        self.assertEqual(info["reloads"], 1)
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0].checkpoint, "/fake/step_000001")
        self.assertEqual(instances[1].checkpoint, "/fake/step_000400")
        self.assertEqual(instances[1].temperature, 1.5)

    def test_reload_lock_serializes_concurrent_decide(self):
        """decide() blocks until a concurrent reload() finishes."""

        gate = threading.Event()

        class _SlowStub(_StubModel):
            def __init__(self, checkpoint, opts, logits_by_qid=None):
                super().__init__(checkpoint, opts, logits_by_qid=logits_by_qid)

            def decide(self, state, questions):
                gate.set()
                time.sleep(0.05)
                return super().decide(state, questions)

        def factory(checkpoint, opts):
            return _SlowStub(checkpoint, opts, logits_by_qid={"q": [2.0, 0.0]})

        served = self.module.ServedModel(
            factory=factory,
            checkpoint="/fake/step_000001",
            model_name="slow-v1",
            model_opts={"max_length": 512, "temperature": 1.0, "attn_backend": "native", "max_tokens": 1024},
        )
        order: list[str] = []

        def do_decide():
            served.decide("s", {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}})
            order.append("decide_done")

        thread = threading.Thread(target=do_decide)
        thread.start()
        gate.wait()
        served.reload("/fake/step_000002", model_name="slow-v2")
        order.append("reload_done")
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        # The reload waited for the in-flight decide (lock contention), so decide completes first.
        self.assertEqual(order[0], "decide_done")
        self.assertEqual(order[1], "reload_done")
        self.assertEqual(served.model_info()["model"], "slow-v2")


class DecisionsRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_serve_decisions()
        from fastapi.testclient import TestClient

        cls.TestClient = TestClient

    def _served(self, *, logits_by_qid, temperature=1.0):
        def factory(checkpoint, opts):
            return _StubModel(checkpoint, opts, logits_by_qid=logits_by_qid)

        return self.module.ServedModel(
            factory=factory,
            checkpoint="/fake/step_000001",
            model_name="stub",
            model_opts={"max_length": 512, "temperature": temperature, "attn_backend": "native", "max_tokens": 1024},
        )

    def test_choice_response_carries_logits(self):
        served = self._served(logits_by_qid={"q": [1.5, 0.1]})
        app = self.module.build_app(served, api_key=None)
        client = self.TestClient(app)
        body = {
            "state": "ctx",
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}},
        }
        response = client.post("/api/alpha/decisions", json=body)
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answers"]["q"]
        self.assertEqual(answer["logits"], {"a": 1.5, "b": 0.1})
        # probabilities and logits are consistent (prob[a] > prob[b]).
        self.assertGreater(answer["probabilities"]["a"], answer["probabilities"]["b"])

    def test_admin_reload_requires_admin_key(self):
        served = self._served(logits_by_qid={})
        app = self.module.build_app(served, api_key=None, admin_key="s3cret")
        client = self.TestClient(app)

        unauth = client.post("/admin/reload", json={"checkpoint": "/fake/step_000002"})
        self.assertEqual(unauth.status_code, 401)

        ok = client.post(
            "/admin/reload",
            json={"checkpoint": "/fake/step_000002", "temperature": 1.3, "model_name": "stub-v2"},
            headers={"Authorization": "Bearer s3cret"},
        )
        self.assertEqual(ok.status_code, 200, ok.json())
        payload = ok.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model"], "stub-v2")
        self.assertEqual(payload["temperature"], 1.3)
        self.assertEqual(payload["reloads"], 1)

    def test_admin_reload_rejects_bad_body(self):
        served = self._served(logits_by_qid={})
        app = self.module.build_app(served, api_key=None, admin_key="s3cret")
        client = self.TestClient(app)
        headers = {"Authorization": "Bearer s3cret"}

        self.assertEqual(client.post("/admin/reload", json={}, headers=headers).status_code, 422)
        self.assertEqual(
            client.post("/admin/reload", json={"checkpoint": ""}, headers=headers).status_code, 422
        )
        self.assertEqual(
            client.post(
                "/admin/reload", json={"checkpoint": "/fake/x", "temperature": -1.0}, headers=headers
            ).status_code,
            422,
        )

    def test_admin_reload_route_absent_without_admin_key(self):
        served = self._served(logits_by_qid={})
        app = self.module.build_app(served, api_key=None, admin_key=None)
        client = self.TestClient(app)
        response = client.post("/admin/reload", json={"checkpoint": "/x"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
