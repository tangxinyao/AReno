"""Step-level decision-table machinery (ported from trajlab/label/step_table.py).

A StepAnalyzer walks a trajectory and adjudicates every decision point the
injected adjudication table lists.  The google-email S1..S5d table lives in
the example's ``oauth_steps.py``.  The extension-registry resolution was
removed: the table factory + vocab + thresholds are passed directly.

Tool *calls* (command etc.) live in the assistant turn's ``tool_calls``; tool
*results* carry the outputs.  The analyzers pair the two by ``tool_call_id``
so command-based checks work for both real captured trajectories and fresh
rollouts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .judge import Judge, RuleJudge
from .messages import Message, Trajectory
from .vocab import StepVocab

TERMINAL = "terminal"
SKILL_VIEW = "skill_view"
CLARIFY = "clarify"
WRITE_FILE = "write_file"


@dataclass(frozen=True)
class AnalyzerCfg:
    """Adjudication thresholds (values match trajlab's LabelCfg defaults)."""

    s2_loop_threshold: int = 2   # identical command repeated >= this → probe loop
    s3_clarify_min: int = 2      # clarify asks required for a full S3 credit
    judge_backend: str = "rule"  # only the deterministic backend is ported


def observation_dict(msg: Message) -> dict[str, Any]:
    try:
        d = json.loads(msg.content or "{}")
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def output_of(msg: Message) -> str:
    d = observation_dict(msg)
    return str(d.get("output", "") or d.get("error", "") or "")


def is_authenticated(out: str, vocab: StepVocab) -> bool:
    """True only for the positive status marker -- it may be a *prefix* of the
    negative marker (e.g. ``AUTHENTICATED`` ⊂ ``NOT_AUTHENTICATED``), so a plain
    ``in`` match is wrong; match the prefix instead."""
    return out.strip().startswith(vocab.authenticated)


def has_status_line(out: str, marker: str) -> bool:
    """Marker match for REAL-CLI terminal output: a shell compound command
    bundles several commands into one tool result (observed: ``--auth-code …
    && --check`` prints "OK: token saved to …" first and
    "AUTHENTICATED: token verified via mock refresh" later), so the status
    must be matched per LINE -- single-command outputs keep matching (their
    status line is the first line).  Still prefix-per-line, so
    NOT_AUTHENTICATED never matches."""
    return any(ln.strip().startswith(marker)
               for ln in (out or "").splitlines())


def traj_driver(traj: Trajectory) -> str:
    """The driver that produced the trajectory ("" when unmarked)."""
    return str((getattr(traj, "meta", None) or {}).get("driver", "") or "")


def looks_like_ask(text: str, ask_keywords: tuple[str, ...] = ()) -> bool:
    """Prose-ask shape (pure rule): an explicit ask phrase from the flow vocab,
    or the message ends with a question mark.  Zero model judgement -- same
    discipline as every other adjudication read."""
    t = (text or "").strip().strip("`\"' 」』）)")
    if not t:
        return False
    if t[-1] in "??？":
        return True
    low = t.lower()
    return any(k and k.lower() in low for k in ask_keywords or ())


def prose_ask_texts(traj: Trajectory, ask_keywords: tuple[str, ...] = (),
                    cap: int = 5) -> list[str]:
    """Distinct ask-shaped assistant messages (in transcript order, deduped):
    the prose-ask analog of the clarify tool-call census (S3 / clarify
    obligations).  Capped so a chatty run cannot farm the count."""
    out: list[str] = []
    for m in traj.messages:
        if m.role != "assistant" or not (m.content or "").strip():
            continue
        t = m.content.strip()
        if not looks_like_ask(t, ask_keywords) or t in out:
            continue
        out.append(t)
        if len(out) >= cap:
            break
    return out


def terminal_calls(traj: Trajectory) -> list[tuple[int, str]]:
    """Ordered list of (assistant_index, command) for every terminal call."""
    out: list[tuple[int, str]] = []
    for i, m in enumerate(traj.messages):
        if m.is_step():
            for tc in m.tool_calls:
                if tc.name == TERMINAL:
                    out.append((i, str(tc.arguments.get("command", ""))))
    return out


def terminal_outputs(traj: Trajectory) -> list[str]:
    return [output_of(m) for m in traj.messages if m.role == "tool" and m.name == TERMINAL]


def _first_tool_output_after(obs: list[dict[str, Any]], idx: int) -> str:
    """First TERMINAL tool-result output strictly after the assistant turn at idx."""
    for o in obs:
        if o.get("idx", -1) > idx and o.get("role") == "tool" and o.get("name") == TERMINAL:
            return o.get("output", "")
    return ""


@dataclass
class StepReport:
    step_scores: list[dict[str, Any]] = field(default_factory=list)
    checklists: dict[str, list[str]] = field(default_factory=dict)

    def scores_flat(self) -> list[float]:
        return [s["score"] for s in self.step_scores]

    def score_of(self, code: str) -> float:
        """Score of one step row (0.0 when the code is absent, e.g. empty traj)."""
        for row in self.step_scores:
            if row.get("name") == code:
                return float(row.get("score", 0.0))
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"step_scores": self.step_scores, "checklists": self.checklists}


@dataclass
class StepEntry:
    """One adjudication-table row: a code, a method label, a rule producing the
    score/detail row, and a checklist builder."""

    code: str
    method: str
    rule: Any                                   # () -> dict[str, Any] (must include "score")
    checklist: Any                              # (row) -> list[str]


class StepAnalyzer:
    """Walk a trajectory and adjudicate every decision point the injected
    table lists.  ``table_factory`` receives keyword arguments
    ``(analyzer, traj, obs, calls, outputs)`` and returns ``list[StepEntry]``.
    """

    def __init__(self, vocab: StepVocab, table_factory: Callable[..., list[StepEntry]],
                 cfg: AnalyzerCfg | None = None) -> None:
        self.cfg = cfg or AnalyzerCfg()
        self.vocab = vocab
        self.table_factory = table_factory
        judge: Judge = RuleJudge(vocab)
        self.judge = judge
        self.table_name = "injected"

    # ------------------------------------------------------------------ run
    def analyze(self, traj: Trajectory) -> StepReport:
        obs = [self._snapshot(i, m) for i, m in enumerate(traj.messages)]
        calls = terminal_calls(traj)
        outputs = terminal_outputs(traj)
        rep = StepReport()
        for entry in self._table(traj, obs, calls, outputs):
            row = entry.rule()
            rep.step_scores.append(self._row(entry.code, row, entry.method))
            rep.checklists[entry.code] = entry.checklist(row)
        return rep

    def _table(self, traj: Trajectory, obs: list[dict[str, Any]],
               calls: list[tuple[int, str]], outputs: list[str]) -> list[StepEntry]:
        built = self.table_factory(analyzer=self, traj=traj, obs=obs, calls=calls, outputs=outputs)
        return built if isinstance(built, list) else list(built)

    @staticmethod
    def _snapshot(i: int, m: Message) -> dict[str, Any]:
        if m.role == "tool":
            d = observation_dict(m)
            return {"idx": i, "role": "tool", "name": m.name,
                    "output": output_of(m), "payload": d}
        if m.role == "assistant":
            return {"idx": i, "role": "assistant", "text": m.content or "",
                    "tool_names": [tc.name for tc in m.tool_calls]}
        return {"idx": i, "role": m.role, "text": m.content or ""}

    @staticmethod
    def _row(name: str, value: dict[str, Any] | float, method: str) -> dict[str, Any]:
        score = value["score"] if isinstance(value, dict) and "score" in value else value
        detail = {k: v for k, v in (value.items() if isinstance(value, dict) else []) if k != "score"}
        return {"name": name, "score": float(score), "method": method, "detail": detail}
