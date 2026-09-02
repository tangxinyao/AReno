"""Deterministic reward for the OAuth agentic demo.

Composition (all terms deterministic; see README "Reward composition"):

    reward = 0.70 * outcome_term + 0.30 * process_term - 0.25 * len(process_penalties)

  * ``outcome_term`` -- the outcome ruler: 1.0 when the episode's final state
    matches ``expected_outcome(combo)`` (reach combos must actually verify
    AUTHENTICATED via ``setup.py --check``; blocked combos must NOT claim a
    success), else 0.0.
  * ``process_term`` -- mean of the S1..S5d step scores (S4 normalized /5),
    i.e. the graded process quality of the whole trajectory.
  * ``process_penalties`` -- `oauth_gate.process_checks` drift reasons:
    forged anchors (reach-success without walking /authorize + /token) and
    missed clarify obligations (scope always; protection for `yes` combos).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from grading.messages import Trajectory  # noqa: E402
from grading.step_table import AnalyzerCfg, StepAnalyzer  # noqa: E402
from oauth_gate import process_checks  # noqa: E402
from oauth_steps import OAUTH_TABLE  # noqa: E402
from oauth_tools import _normalized_progress  # noqa: E402

OUTCOME_WEIGHT = 0.70
PROCESS_WEIGHT = 0.30
PENALTY_WEIGHT = 0.25

_ANALYZER: StepAnalyzer | None = None


def _analyzer() -> StepAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        from oauth_vocab import GOOGLE_EMAIL_VOCAB

        _ANALYZER = StepAnalyzer(GOOGLE_EMAIL_VOCAB, OAUTH_TABLE, AnalyzerCfg())
    return _ANALYZER


def reward_fn(record) -> float:
    """Score one agentic rollout record (areno RewardRecord)."""
    messages = list(getattr(record, "messages", None) or [])
    source = dict(getattr(record, "source_record", None) or {})
    spec = json.loads(str(source.get("task_spec") or "{}"))

    traj = Trajectory.from_dict({"messages": messages, "meta": {"driver": "areno"}})
    report = _analyzer().analyze(traj)

    reached = report.score_of("S5d") >= 1.0
    expected = spec.get("expected") or {}
    outcome_term = float(reached == (expected.get("outcome") == "reach"))
    process_term = _normalized_progress(report)

    ctx = {
        "scenario": SimpleNamespace(env=spec.get("env") or {}, user=spec.get("user") or {}),
        "traj": traj,
        "combo": spec.get("combo") or {},
        "witness": _last_witness(messages),
        "verdict": "success" if reached else "failure",
        "expected": expected,
    }
    penalties = process_checks(ctx)
    reward = OUTCOME_WEIGHT * outcome_term + PROCESS_WEIGHT * process_term \
        - PENALTY_WEIGHT * len(penalties)
    return round(max(0.0, reward), 4)


def _last_witness(messages: list) -> dict | None:
    """Latest world witness relayed in a tool result (see oauth_tools).

    Every tool result carries the hit counters of its instant, so the last one
    in transcript order is the final state -- including the zero-hit witness a
    write_file-forged token anchor would produce (which the gate then rules a
    forged anchor)."""
    witness = None
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        content = m.get("content")
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("witness"), dict):
            witness = payload["witness"]
    return witness
