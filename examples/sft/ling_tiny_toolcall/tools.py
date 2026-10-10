"""Hermes-faithful action space + offline harness backend for the tool-call SFT example.

This module IS the action space the trained Ling-3.0-tiny must match at deploy
time. Earlier revisions of this example invented a small ``list_emails`` /
``read_email`` / ``search_emails`` / ``send_email`` toolset; that action space
does not exist in Hermes, so trajectories collected under it transfer poorly to
the real agent. This revision ports the *actual* Hermes tools the Gmail-login
workflow uses -- ``skill_view``, ``skills_list``, ``clarify``, and ``terminal`` --
with the real schema text, and backs ``terminal`` with a small deterministic
fake shell that emulates the Google Workspace skill's setup script and
``google_api.py`` CLI.

Design constraints (Harness-Zero action-space match):

  * Tool names, argument schemas, and result JSON shapes mirror Hermes
    (``tools/skills_tool.py``, ``tools/clarify_tool.py``, ``tools/terminal_tool.py``).
  * ``TERMINAL_SCHEMA`` / ``SKILLS_LIST_SCHEMA`` / ``SKILL_VIEW_SCHEMA`` /
    ``CLARIFY_SCHEMA`` carry the same description text as Hermes so the served
    tool prompt the student sees at collect time matches deployment.
  * Nothing here imports third-party code: stdlib only, so the CPU test harness
    runs anywhere.

The fake shell does NOT execute arbitrary commands. It pattern-matches the
handful of command shapes the skill documents (``setup.py`` subcommands and
``google_api.py`` verbs) and replays canned, deterministic output. That keeps the
trajectory reproducible and offline while preserving the exact command strings a
real Hermes agent would emit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Tool schemas (ported from Hermes; description text preserved)
# ---------------------------------------------------------------------------

TERMINAL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": (
            "Execute shell commands on a Linux environment. Filesystem, current "
            "working directory, and exported environment variables persist between "
            "calls. Reserve terminal for: builds, installs, git, processes, scripts, "
            "network, package managers, and anything that needs a shell."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute on the VM",
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in the background, returning a session_id.",
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait (default: 180).",
                    "minimum": 1,
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for this command (absolute path).",
                },
                "pty": {
                    "type": "boolean",
                    "description": "Run in pseudo-terminal (PTY) mode for interactive CLI tools.",
                    "default": False,
                },
            },
            "required": ["command"],
        },
    },
}

SKILLS_LIST_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "skills_list",
        "description": (
            "List available skills (name + description). Use skill_view(name) to "
            "load full content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter to narrow results",
                }
            },
            "required": [],
        },
    },
}

SKILL_VIEW_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "skill_view",
        "description": (
            "Skills allow for loading information about specific tasks and "
            "workflows, as well as scripts and templates. Load a skill's full "
            "content or access its linked files (references, templates, scripts). "
            "First call returns SKILL.md content plus a 'linked_files' dict showing "
            "available references/templates/scripts. To access those, call again "
            "with file_path parameter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill name (use skills_list to see available skills)."
                    ),
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "OPTIONAL: Path to a linked file within the skill "
                        "(e.g., 'references/api.md'). Omit to get the main SKILL.md "
                        "content."
                    ),
                },
            },
            "required": ["name"],
        },
    },
}

CLARIFY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "clarify",
        "description": (
            "Ask the user a question when you need clarification, feedback, or a "
            "decision before proceeding. Provide up to 4 choices; the user picks "
            "one or types their own answer via a 5th 'Other' option. List the "
            "choice you recommend FIRST. Omit choices for an open-ended question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The question itself, and ONLY the question. Do NOT embed "
                        "the answer options here."
                    ),
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 4 selectable choices. Omit for open-ended.",
                },
            },
            "required": ["question"],
        },
    },
}

# The exact toolset the Hermes CLI exposes for this workflow. Kept minimal: the
# Gmail-login episode only needs skill discovery, a decision prompt, and a shell
# to drive the Google Workspace skill scripts.
TOOLS: list[dict[str, Any]] = [
    SKILLS_LIST_SCHEMA,
    SKILL_VIEW_SCHEMA,
    CLARIFY_SCHEMA,
    TERMINAL_SCHEMA,
]

_TOOL_NAMES = {tool["function"]["name"] for tool in TOOLS}


class ToolCallError(ValueError):
    """Raised when an assistant turn names an unknown tool or supplies bad arguments."""


# ---------------------------------------------------------------------------
# Skill registry (reads the real SKILL.md files vendored under skills/)
# ---------------------------------------------------------------------------

SKILLS_ROOT = Path(__file__).resolve().parent / "skills"


@dataclass(frozen=True)
class SkillRecord:
    """One vendored skill: where its files live and the frontmatter we index."""

    name: str
    rel_path: str
    description: str
    tags: list[str]
    related_skills: list[str]
    skill_dir: Path

    @property
    def category(self) -> str:
        # category is the first path segment, e.g. "productivity" / "email"
        return self.rel_path.split("/", 1)[0] if "/" in self.rel_path else ""


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into (frontmatter dict, body). Minimal YAML subset.

    The vendored skills use a flat ``key: value`` frontmatter plus a nested
    ``metadata.hermes.tags`` list. Rather than depend on PyYAML we parse just the
    keys this example needs (name, description, metadata hermes tags/related).
    """

    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    raw, body = parts[1], parts[2]
    meta: dict[str, Any] = {}
    current_list_key: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("  ") and current_list_key and line.strip().startswith("-"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            # could be a nested block; remember it for a following tags list
            current_list_key = key
            continue
        if key == "name":
            meta["name"] = value.strip('"')
        elif key == "description":
            meta["description"] = value.strip().strip('"')
        elif key == "tags" and value.startswith("["):
            meta["tags"] = [t.strip() for t in value.strip("[]").split(",") if t.strip()]
        elif key == "related_skills" and value.startswith("["):
            meta["related_skills"] = [
                t.strip() for t in value.strip("[]").split(",") if t.strip()
            ]
    return meta, body


def _discover_skills(root: Path = SKILLS_ROOT) -> list[SkillRecord]:
    """Find every ``<category>/<name>/SKILL.md`` under the vendored skills tree."""

    records: list[SkillRecord] = []
    if not root.is_dir():
        return records
    for skill_md in sorted(root.glob("*/*/SKILL.md")):
        skill_dir = skill_md.parent
        rel_path = f"{skill_dir.parent.name}/{skill_dir.name}/SKILL.md"
        meta, _ = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        records.append(
            SkillRecord(
                name=meta.get("name") or skill_dir.name,
                rel_path=rel_path,
                description=meta.get("description", ""),
                tags=list(meta.get("tags", [])),
                related_skills=list(meta.get("related_skills", [])),
                skill_dir=skill_dir,
            )
        )
    return records


# ---------------------------------------------------------------------------
# The fake shell: a deterministic replay of the Google Workspace setup flow
# ---------------------------------------------------------------------------


# The fixture client-secret path the simulated user hands back, and a fake
# account. These are fixed so every collected episode is identical.
FIXTURE_CLIENT_SECRET = "~/Downloads/client_secret_example.apps.googleusercontent.com.json"
FIXTURE_ACCOUNT = "sha7tang@gmail.com"
FIXTURE_AUTH_URL = (
    "https://accounts.google.com/o/oauth2/auth?response_type=code"
    "&client_id=524404221282-example.apps.googleusercontent.com"
    "&redirect_uri=http%3A%2F%2Flocalhost%3A1"
    "&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fgmail.readonly"
    "+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar"
    "&state=EXAMPLE_STATE&code_challenge=EXAMPLE_CHALLENGE"
    "&code_challenge_method=S256&access_type=offline&prompt=consent"
)

# The fake unread-mail listing the terminal backend returns for the Gmail query.
FIXTURE_UNREAD: list[dict[str, Any]] = [
    {
        "id": "1a11efa219fdd936",
        "threadId": "1a11efa219fdd936",
        "from": "McKinsey Weekend Read <publishing@email.mckinsey.com>",
        "to": FIXTURE_ACCOUNT,
        "subject": "Is your organization standing between AI and value?",
        "date": "Fri, 9 Oct 2026 04:44:30 +0000",
        "snippet": "Big ideas to explore this weekend",
        "labels": ["UNREAD", "CATEGORY_UPDATES", "INBOX"],
    },
    {
        "id": "1a11e554e743d131",
        "threadId": "1a11e554e743d131",
        "from": "Upwork Notification <donotreply@upwork.com>",
        "to": FIXTURE_ACCOUNT,
        "subject": "New job alert: AI and Automation Developer",
        "date": "Fri, 09 Oct 2026 01:44:28 +0000",
        "snippet": "This job was just posted and matches your alert settings.",
        "labels": ["UNREAD", "CATEGORY_UPDATES", "INBOX"],
    },
]


def _find_auth_code(command: str) -> str | None:
    """Pull the OAuth ``code=...`` value out of a ``--auth-code`` command."""

    match = re.search(r"code=([A-Za-z0-9/_.\-]+)", command)
    return match.group(1) if match else None


def _terminal_google_workspace(command: str) -> dict[str, Any] | None:
    """Replay the documented Google Workspace commands; None if unrecognized.

    Matches only the command shapes the ``google-workspace`` SKILL.md documents
    (``setup.py`` subcommands and ``google_api.py`` verbs). Everything else falls
    through to the generic shell handler.
    """

    # --- setup.py subcommands -------------------------------------------------
    if "setup.py" in command:
        if "--check" in command and "google_token.json" not in command:
            return {"output": "NOT_AUTHENTICATED: No token at ~/.hermes/google_token.json",
                    "exit_code": 1, "error": None}
        if "--client-secret" in command:
            return {
                "output": (
                    "OK: Client secret saved to "
                    "~/.hermes/google_client_secret.json"
                ),
                "exit_code": 0,
                "error": None,
            }
        if "--auth-url" in command:
            return {"output": FIXTURE_AUTH_URL, "exit_code": 0, "error": None}
        if "--auth-code" in command:
            code = _find_auth_code(command)
            if code:
                return {
                    "output": (
                        "OK: Authenticated. Token saved to "
                        "~/.hermes/google_token.json\n"
                        "AUTHENTICATED: Token valid at ~/.hermes/google_token.json"
                    ),
                    "exit_code": 0,
                    "error": None,
                }
            return {"output": "ERROR: no authorization code in the pasted value",
                    "exit_code": 1, "error": None}

    # --- google_api.py verbs --------------------------------------------------
    if "google_api.py" in command or " gmail " in f" {command} ":
        if re.search(r"\bgmail\s+search\b", command):
            import json as _json

            return {
                "output": _json.dumps(FIXTURE_UNREAD, ensure_ascii=False, indent=2),
                "exit_code": 0,
                "error": None,
            }
        if re.search(r"\bgmail\s+get\b", command):
            import json as _json

            message = dict(FIXTURE_UNREAD[0])
            message["body"] = "Big ideas to explore this weekend."
            return {
                "output": _json.dumps(message, ensure_ascii=False, indent=2),
                "exit_code": 0,
                "error": None,
            }

    return None


def run_terminal_command(command: str) -> dict[str, Any]:
    """Execute one command against the deterministic fake shell.

    Returns a Hermes-shaped terminal result: ``{"output", "exit_code", "error"}``.
    Only the documented Google Workspace command shapes produce real-looking
    output; anything else gets a benign "command not found" so a mis-specified
    call is visible in the trajectory instead of silently succeeding.
    """

    command = (command or "").strip()
    if not command:
        raise ToolCallError("terminal: empty command")
    handled = _terminal_google_workspace(command)
    if handled is not None:
        return handled
    # A bare `which gws` / version probe the model may try first.
    if command.startswith("which ") or " --version" in command:
        return {"output": f"/bin/bash: {command.split()[0]}: command not found",
                "exit_code": 1, "error": None}
    return {
        "output": f"/bin/bash: {command}: command not found",
        "exit_code": 127,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Tool execution + skill_view payloads
# ---------------------------------------------------------------------------


def _skill_view_payload(record: SkillRecord, file_path: str | None) -> dict[str, Any]:
    """Build the Hermes-shaped ``skill_view`` result for a skill or linked file.

    Mirrors the result dict assembled in ``tools/skills_tool.py``: the SKILL.md
    content plus a ``linked_files`` map, ``readiness_status``, and the
    ``setup_needed`` flags. The google-workspace skill reports ``setup_needed``
    (missing credential files), which is exactly the signal that drives the login
    workflow.
    """

    if file_path:
        target = (record.skill_dir / file_path).resolve()
        # keep the read inside the skill directory
        if not str(target).startswith(str(record.skill_dir.resolve())) or not target.is_file():
            return {"success": False, "error": f"linked file not found: {file_path}"}
        return {
            "success": True,
            "name": record.name,
            "file_path": file_path,
            "content": target.read_text(encoding="utf-8"),
        }

    content = (record.skill_dir / "SKILL.md").read_text(encoding="utf-8")
    linked = {}
    for sub in ("references", "templates", "assets", "scripts"):
        sub_dir = record.skill_dir / sub
        if sub_dir.is_dir():
            linked[sub] = sorted(p.name for p in sub_dir.iterdir() if p.is_file())
    # The Google Workspace skill needs credential files this offline harness never
    # has, so it reports setup_needed -- the trigger for the login workflow.
    setup_needed = record.name == "google-workspace"
    payload: dict[str, Any] = {
        "success": True,
        "name": record.name,
        "description": record.description,
        "tags": record.tags,
        "related_skills": record.related_skills,
        "content": content,
        "path": record.rel_path,
        "linked_files": linked or None,
        "usage_hint": (
            "To view linked files, call skill_view(name, file_path) where file_path "
            "is e.g. 'references/api.md' or 'assets/config.yaml'"
            if linked
            else None
        ),
        "missing_credential_files": (
            ["google_token.json", "google_client_secret.json"] if setup_needed else []
        ),
        "setup_needed": setup_needed,
        "readiness_status": "setup_needed" if setup_needed else "available",
    }
    if setup_needed:
        payload["setup_note"] = (
            "Setup needed before using this skill: missing file google_token.json, "
            "file google_client_secret.json."
        )
    return payload


def execute_tool(name: str, arguments: dict[str, Any], env: "AgentEnv") -> dict[str, Any]:
    """Dispatch one parsed tool call to the offline harness.

    ``name`` must be one of the exported ``TOOLS``; unknown names raise so the
    collector/converter can classify the turn as a format collapse. Argument
    coercion is intentionally permissive: the collector parses the JSON, then
    passes plain Python values through.
    """

    if name not in _TOOL_NAMES:
        raise ToolCallError(f"unknown tool name {name!r}; expected one of {sorted(_TOOL_NAMES)}")
    try:
        if name == "terminal":
            return run_terminal_command(str(arguments.get("command", "")))
        if name == "skills_list":
            category = arguments.get("category")
            return env.skills_list(category=category)
        if name == "skill_view":
            return env.skill_view(
                name=str(arguments.get("name", "")),
                file_path=arguments.get("file_path"),
            )
        if name == "clarify":
            return env.clarify(
                question=str(arguments.get("question", "")),
                choices=arguments.get("choices"),
            )
    except TypeError as exc:
        raise ToolCallError(f"bad arguments for {name!r}: {exc}") from exc
    raise ToolCallError(f"unhandled tool {name!r}")


def tool_result_to_text(result: dict[str, Any]) -> str:
    """Render a tool result for the ``tool`` role message in a trajectory.

    Uses stable, compact JSON so the SFT converter reproduces identical prompt
    text across runs (the seed data relies on determinism).
    """

    import json

    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Environment: the harness's scripted decision-maker for clarify()
# ---------------------------------------------------------------------------


@dataclass
class AgentEnv:
    """Per-episode harness state.

    ``decisions`` is a FIFO of canned user answers consumed by ``clarify``; the
    Nth clarify call returns the Nth decision. This is the simulated user for the
    deterministic upload flow: the episode is a fixed script, so the harness -- not
    a live human -- supplies each answer.
    """

    skills: list[SkillRecord] = field(default_factory=lambda: _discover_skills())
    decisions: list[str] = field(default_factory=list)
    clarify_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def default(cls) -> "AgentEnv":
        """Env for the Gmail-login episode: user picks the gws route, then gives the path."""

        return cls(
            decisions=[
                "邮箱 + 日历/云盘/文档等（OAuth 授权）",
                f"文件路径是：{FIXTURE_CLIENT_SECRET}",
            ]
        )

    def _find(self, name: str) -> SkillRecord | None:
        for record in self.skills:
            if record.name == name:
                return record
        return None

    def skills_list(self, *, category: str | None = None) -> dict[str, Any]:
        records = self.skills
        if category:
            records = [r for r in records if r.category == category]
        return {
            "success": True,
            "skills": [
                {
                    "name": r.name,
                    "description": r.description,
                    "category": r.category,
                    "tags": r.tags,
                }
                for r in records
            ],
        }

    def skill_view(self, *, name: str, file_path: str | None = None) -> dict[str, Any]:
        record = self._find(name)
        if record is None:
            return {"success": False, "error": f"skill not found: {name}"}
        return _skill_view_payload(record, file_path)

    def clarify(self, *, question: str, choices: list[str] | None = None) -> dict[str, Any]:
        self.clarify_log.append({"question": question, "choices": choices})
        if not self.decisions:
            return {
                "question": question,
                "choices_offered": choices or [],
                "user_response": "",
            }
        response = self.decisions.pop(0)
        return {
            "question": question,
            "choices_offered": choices or [],
            "user_response": response,
        }


# ---------------------------------------------------------------------------
# Hermes tool-call delimiters (on-wire format; assembled from code points)
# ---------------------------------------------------------------------------
#
# The real on-wire tool-call tokens are the Hermes ``<tool_call>{json}</tool_call>``
# block. They are assembled from code points so no literal angle-bracket marker
# appears as a source literal, keeping the format contract in one place for every
# consumer (build_dataset.py, mail_graph.py, collect_trajectories.py).

TOOL_CALL_OPEN = "".join([chr(60) * 3, "tool_call", chr(62)])
TOOL_CALL_CLOSE = "".join([chr(60), "/", "tool_call", chr(62)])
TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CALL_CLOSE), re.DOTALL
)


def parse_hermes_tool_calls(text: str) -> list[dict[str, Any]]:
    """Return parsed Hermes tool-call dicts in ``text`` (empty if none / malformed).

    Each matched block must decode to JSON with a string ``name`` and an object
    ``arguments``; anything else is treated as a format collapse.
    """

    import json

    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        calls.append({"name": name, "arguments": arguments})
    return calls


def render_hermes_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Render one tool call as the exact on-wire Hermes block the model must emit."""

    import json

    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"


__all__ = [
    "AgentEnv",
    "CLARIFY_SCHEMA",
    "FIXTURE_ACCOUNT",
    "FIXTURE_CLIENT_SECRET",
    "SKILLS_LIST_SCHEMA",
    "SKILL_VIEW_SCHEMA",
    "SKILLS_ROOT",
    "SkillRecord",
    "TERMINAL_SCHEMA",
    "TOOLS",
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "ToolCallError",
    "execute_tool",
    "parse_hermes_tool_calls",
    "render_hermes_tool_call",
    "run_terminal_command",
    "tool_result_to_text",
]