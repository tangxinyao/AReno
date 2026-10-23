"""CPU tests for the live ReAct player: observation layout and the model call.

The model call is exercised against a local stub HTTP server; no game, GPU or
network is involved.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "sts2"))

import play_live  # noqa: E402
import env  # noqa: E402
from sts2mcp_live import Sts2McpLiveBackend  # noqa: E402


def _combat_state() -> dict:
    return {
        "state_type": "monster",
        "run": {"act": 1, "floor": 2},
        "player": {
            "hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3,
            "hand": [
                {"index": 0, "id": "STRIKE_IRONCLAD", "name": "Strike", "type": "Attack", "cost": "1",
                 "description": "Deal 6 damage.", "target_type": "AnyEnemy", "can_play": True, "is_upgraded": False},
                {"index": 1, "id": "DEFEND_IRONCLAD", "name": "Defend", "type": "Skill", "cost": "1",
                 "description": "Gain 5 Block.", "target_type": "Self", "can_play": True, "is_upgraded": False},
            ],
            "potions": [], "status": [], "relics": [],
            "draw_pile_count": 5, "discard_pile_count": 0, "exhaust_pile_count": 0,
        },
        "battle": {
            "round": 1, "turn": "player", "is_play_phase": True,
            "enemies": [
                {"entity_id": "TWIG_0", "name": "Twig Slime (S)", "hp": 7, "max_hp": 7, "block": 0,
                 "intents": [{"type": "attack", "label": "TACKLE"}], "status": []},
                {"entity_id": "LEAF_1", "name": "Leaf Slime (S)", "hp": 0, "max_hp": 11, "block": 0,
                 "intents": [], "status": []},
            ],
        },
    }


class _FakeClient:
    def __init__(self, state: dict) -> None:
        self._state = state

    def get_state(self) -> dict:
        return self._state

    def post(self, body: dict) -> dict:
        return {"status": "ok"}


class RenderObservationTest(unittest.TestCase):
    def test_combat_layout_matches_sim_observation(self) -> None:
        backend = Sts2McpLiveBackend(_FakeClient(_combat_state()), settle_timeout=0.0, sleep=lambda _s: None)
        packet = backend.reset()
        text = play_live.render_observation(packet)
        self.assertTrue(text.startswith("Turn 1 | HP 70/80 | Block 0 | Energy 3/3"))
        self.assertIn("- [0] Twig Slime (S) HP 7/7 Block 0 Intent: TACKLE", text)
        enemies_section = text.split("Legal actions:")[0]
        self.assertNotIn("Leaf Slime", enemies_section)  # dead enemies are hidden from the enemy list
        self.assertIn("Legal actions:", text)
        self.assertIn("- end_turn ::", text)

    def test_sim_action_ids_are_produced_for_live_cards(self) -> None:
        backend = Sts2McpLiveBackend(_FakeClient(_combat_state()), settle_timeout=0.0, sleep=lambda _s: None)
        ids = [c["id"] for c in backend.reset()["candidates"]]
        self.assertIn("play_card:strike_ironclad:0", ids)
        self.assertIn("end_turn", ids)


class _StubModel(BaseHTTPRequestHandler):
    seen: dict = {}

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _StubModel.seen = {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        reply = {"choices": [{"message": {"role": "assistant", "content": "Thought: x\nAction: end_turn"}}]}
        data = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args) -> None:
        return


class ChatCompletionTest(unittest.TestCase):
    def test_round_trip_to_openai_compatible_endpoint(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _StubModel)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/v1"
            reply = play_live.chat_completion(url, "token", [{"role": "user", "content": "hi"}], timeout=5)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(env.parse_action(reply), "end_turn")
        self.assertEqual(_StubModel.seen["path"], "/v1/chat/completions")
        self.assertEqual(_StubModel.seen["auth"], "Bearer token")
        self.assertEqual(_StubModel.seen["body"]["messages"], [{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
