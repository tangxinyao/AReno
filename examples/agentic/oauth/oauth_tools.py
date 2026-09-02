"""Tools for the OAuth agentic example.

Five tools mirror the Hermes surface the original captures were collected
with: ``terminal`` / ``read_file`` / ``write_file`` / ``skill_view`` plus
``clarify``.  The workspace owns one mock OAuth world (`oauth_world.py`) on an
ephemeral port and one S1..S5d adjudicator (`grading/`).

Every tool result carries the world ``witness`` (OAuth endpoint hit counts) so
the reward-side process gate can corroborate anchor claims, and a bounded
``progress_score`` from grading the trajectory so far.  Modelled on
``examples/agentic/office/office_tools.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading.messages import Trajectory  # noqa: E402
from grading.step_table import AnalyzerCfg, StepAnalyzer, StepReport  # noqa: E402
import oauth_env  # noqa: E402
import oauth_gate  # noqa: E402
from oauth_world import MockOAuthServer  # noqa: E402

TOOLS = json.loads(Path(__file__).with_name("oauth_tool_schemas.json").read_text(encoding="utf-8"))

OUTPUT_CAP = 4_000
READ_CAP = 20_000
SKILL_CAP = 12_000
CMD_TIMEOUT_S = 90

# hermes clarify-tool contract constants (tools/clarify_tool.py)
MAX_CHOICES = 4
MAX_QUESTIONS = 5
RECOMMENDED_LABEL = "(Recommended)"
TIMEOUT_RESPONSE = (
    "The user did not provide a response within the time limit. "
    "Use your best judgement to make the choice and proceed."
)


# ---------------------------------------------------------------------------
# clarify argument normalization (ported from hermes tools/clarify_tool.py)
# ---------------------------------------------------------------------------

def _flatten_choice(c: Any) -> str:
    """Coerce a choice into its user-facing display string (LLMs sometimes
    emit dict-shaped choices like {"description": ...})."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, dict):
        for key in ("label", "description", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(c, (list, tuple)):
        return " ".join(_flatten_choice(x) for x in c).strip()
    return str(c).strip()


def mark_recommended(choices: list[str]) -> list[str]:
    """Label the first choice as the agent's recommendation (schema says order
    best-first).  Idempotent; a lone choice isn't a recommendation."""
    if len(choices) < 2:
        return choices
    first = str(choices[0]).strip()
    if first != strip_recommended(first):
        return choices
    return [f"{first} {RECOMMENDED_LABEL}"] + list(choices[1:])


def strip_recommended(text: str) -> str:
    """Remove the recommendation label from a resolved answer."""
    stripped = str(text).strip()
    if stripped.casefold().endswith(RECOMMENDED_LABEL.casefold()):
        return stripped[: -len(RECOMMENDED_LABEL)].strip()
    return stripped


def _parse_multi_select_response(raw_response: Any) -> list[str]:
    if isinstance(raw_response, list):
        return [str(r).strip() for r in raw_response if str(r).strip()]
    raw = str(raw_response).strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        except json.JSONDecodeError:
            pass
    return [s.strip() for s in raw.split(",") if s.strip()]


def _normalize_questions(questions: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate + normalize the ``questions`` batch.  Returns
    ``(normalized, error)``; an empty list means "no batch" (fall through)."""
    if not isinstance(questions, list):
        return None, "questions must be an array of question objects."
    if not questions:
        return None, None
    if len(questions) > MAX_QUESTIONS:
        return None, f"questions supports at most {MAX_QUESTIONS} items."

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict):
            return None, f"questions[{index}] must be an object with a 'question'."

        text = str(item.get("question") or "").strip()
        if not text:
            return None, f"questions[{index}].question must be non-empty text."

        choices = item.get("choices")
        if choices is not None:
            if not isinstance(choices, list):
                return None, f"questions[{index}].choices must be a list."
            choices = [s for s in (_flatten_choice(c) for c in choices) if s]
            if len(choices) > MAX_CHOICES:
                choices = choices[:MAX_CHOICES]
            if not choices:
                choices = None

        model_id = str(item.get("id") or "").strip() or None

        normalized.append({
            "qid": f"q{index}",
            "id": model_id,
            "question": text,
            "choices": mark_recommended(list(choices)) if choices else None,
            "choices_offered": list(choices) if choices else None,
            "multi_select": bool(item.get("multi_select")) and bool(choices),
        })

    return normalized, None


def _clean_batch_answer(entry: dict[str, Any], raw: Any) -> Any:
    if entry["multi_select"]:
        return [strip_recommended(r) for r in _parse_multi_select_response(raw)]
    return strip_recommended(raw)


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OAuthWorkspace:
    """One episode's isolated workspace: fixtures + mock world + grader."""

    record: dict
    root: Path
    spec: dict
    server: MockOAuthServer
    analyzer: StepAnalyzer

    @classmethod
    def from_record(cls, record: dict) -> "OAuthWorkspace":
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="areno-oauth-"))
        try:
            paths = oauth_env.materialize_workspace(record, root)
            spec = oauth_env.parse_task_spec(dict(record))
            server = MockOAuthServer(spec["server"], paths, port=0).start()
            analyzer = StepAnalyzer(_workspace_vocab(), _workspace_table(), AnalyzerCfg())
            return cls(record=dict(record), root=root, spec=spec, server=server, analyzer=analyzer)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    def close(self) -> None:
        self.server.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    # -- process environment for terminal children --------------------------
    def child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["HERMES_HOME"] = (self.root / ".hermes").as_posix()
        env["GOOGLE_OAUTH_BASE_URL"] = self.server.base_url
        return env

    # -- grading -------------------------------------------------------------
    def step_report(self, messages: list[dict[str, Any]]) -> StepReport:
        """Adjudicate the episode trajectory so far."""
        traj = Trajectory.from_dict({"messages": list(messages), "meta": {"driver": "areno"}})
        return self.analyzer.analyze(traj)

    def _with_grade(self, result: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Attach progress / witness evidence to a tool result.

        ``progress_score`` is the mean of the S-step scores (S4 normalized to
        0..1) over the trajectory so far; ``solved`` marks a confirmed
        AUTHENTICATED anchor (the S5d row).
        """
        report = self.step_report(messages)
        result["progress_score"] = round(_normalized_progress(report), 4)
        if report.score_of("S5d") >= 1.0:
            result["solved"] = True
        result["witness"] = dict(self.server.machine.hits)
        return result


def _workspace_vocab():
    from oauth_vocab import GOOGLE_EMAIL_VOCAB
    return GOOGLE_EMAIL_VOCAB


def _workspace_table():
    from oauth_steps import OAUTH_TABLE
    return OAUTH_TABLE


def _normalized_progress(report: StepReport) -> float:
    if not report.step_scores:
        return 0.0
    scaled = []
    for row in report.step_scores:
        s = float(row.get("score", 0.0))
        scaled.append(min(s, 5.0) / 5.0 if row.get("name") == "S4" else min(s, 1.0))
    return sum(scaled) / len(scaled)


# ---------------------------------------------------------------------------
# tool implementations
# ---------------------------------------------------------------------------

def _terminal(workspace: OAuthWorkspace, arguments: dict[str, Any]) -> dict[str, Any]:
    command = str(arguments.get("command", ""))
    if not command.strip():
        return {"error": "terminal requires a non-empty 'command'."}
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace.root,
            env=workspace.child_env(),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=CMD_TIMEOUT_S,
        )
        return {
            "output": (proc.stdout or "")[:OUTPUT_CAP],
            "error": (proc.stderr or "")[:OUTPUT_CAP],
            "exit_code": proc.returncode,
            "success": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"terminal: command timed out after {CMD_TIMEOUT_S}s",
                "exit_code": 124, "success": False}
    except OSError as exc:
        return {"error": f"terminal: {exc}", "success": False}


def read_file(workspace: OAuthWorkspace, arguments: dict[str, Any]) -> dict[str, Any]:
    path = _safe_path(workspace, arguments)
    if path is None:
        return {"error": f"invalid path: {arguments.get('path')!r}", "success": False}
    if not path.is_file():
        return {"error": f"file not found: {path}", "success": False}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": f"read failed: {exc}", "success": False}
    return {"path": path.relative_to(workspace.root).as_posix(),
            "content": text[:READ_CAP], "success": True}


def write_file(workspace: OAuthWorkspace, arguments: dict[str, Any]) -> dict[str, Any]:
    path = _safe_path(workspace, arguments)
    content = arguments.get("content")
    if path is None:
        return {"error": f"invalid path: {arguments.get('path')!r}", "success": False}
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return {"error": f"write failed: {exc}", "success": False}
    return {"path": path.relative_to(workspace.root).as_posix(),
            "bytes": len(content.encode("utf-8")), "success": True}


def skill_view(workspace: OAuthWorkspace, arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments.get("name", "")).strip()
    skills_root = workspace.root / ".hermes" / "skills"
    match = next((p for p in sorted(skills_root.rglob("SKILL.md")) if p.parent.name == name), None)
    if match is None:
        known = sorted({p.parent.name for p in skills_root.rglob("SKILL.md")})
        return {"success": False, "name": name,
                "error": f"unknown skill {name!r}; available: {', '.join(known)}"}
    return {"success": True, "name": name,
            "content": match.read_text(encoding="utf-8", errors="replace")[:SKILL_CAP]}


def clarify(workspace: OAuthWorkspace, arguments: dict[str, Any]) -> dict[str, Any]:
    """Ask the (deterministic scripted) user.  Routes each question against the
    task_spec's ``clarify_answers`` (exact same regex keys the process gate
    judges obligations with); unmatched questions surface the hermes timeout
    sentinel so the agent decides for itself."""
    questions = arguments.get("questions")
    if not questions and arguments.get("question"):
        questions = [{"question": arguments.get("question"),
                      "choices": arguments.get("choices"),
                      "multi_select": arguments.get("multi_select", False)}]
    normalized, error = _normalize_questions(questions)
    if error:
        return {"error": error}
    if not normalized:
        return {"error": "No question provided. Pass questions=[{question: '...', "
                         "choices?: [...], multi_select?: bool}, ...] — a single question "
                         "is a one-entry array."}

    answers_map = dict(workspace.spec["user"].get("clarify_answers") or {})
    answers: dict[str, Any] = {}
    timed_out = False
    for entry in normalized:
        routed = _route_answer(entry["question"], answers_map)
        if routed is None:
            timed_out = True
            routed = ""
        answers[entry["qid"]] = routed

    responses = []
    for entry in normalized:
        row: dict[str, Any] = {}
        if entry["id"]:
            row["id"] = entry["id"]
        row["question"] = entry["question"]
        row["choices_offered"] = entry["choices_offered"]
        raw = answers.get(entry["qid"])
        row["user_response"] = _clean_batch_answer(entry, raw) if raw else ""
        responses.append(row)

    result: dict[str, Any] = {"responses": responses}
    if timed_out:
        result["timed_out"] = True
    return result


def _route_answer(question: str, answers_map: dict[str, str]) -> str | None:
    for key, answer in answers_map.items():
        if oauth_gate.matches(question, key):
            return answer
    return None


# ---------------------------------------------------------------------------
# dispatch + argument helpers
# ---------------------------------------------------------------------------

def run_tool(workspace: OAuthWorkspace, name: str, arguments: dict[str, Any],
             messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Dispatch one tool call; grades the trajectory-so-far onto the result."""
    messages = list(messages or [])
    if name == "terminal":
        result = _terminal(workspace, arguments)
    elif name == "read_file":
        result = read_file(workspace, arguments)
    elif name == "write_file":
        result = write_file(workspace, arguments)
    elif name == "skill_view":
        result = skill_view(workspace, arguments)
    elif name == "clarify":
        result = clarify(workspace, arguments)
    else:
        result = {"error": f"unknown tool: {name}", "success": False}

    # Grading sees the transcript including this call's own result projected in
    # (the agent loop appends the real tool message right after), so the final
    # --check turn gets its AUTHENTICATED credit -- and the solved flag it
    # produces -- in the same step.
    if messages:
        call_id = None
        last = messages[-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            call_id = last["tool_calls"][0].get("id")
        projected = [*messages, {"role": "tool", "tool_call_id": call_id, "name": name,
                                 "content": json.dumps(result, ensure_ascii=False, sort_keys=True)}]
        result = workspace._with_grade(result, projected)
    return result


def _safe_path(workspace: OAuthWorkspace, arguments: dict[str, Any]) -> Path | None:
    """Workspace-relative path resolution: absolutes and escapes rejected."""
    raw = str(arguments.get("path", "") or "")
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return None
    root = workspace.root.resolve()
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        return None
    return path


def decode_tool_arguments(raw: Any) -> dict[str, Any]:
    """Model-emitted arguments: dict passthrough, JSON string decode, {}."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}
