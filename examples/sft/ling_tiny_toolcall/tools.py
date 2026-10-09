"""Mail action space for the LangGraph tool-call distillation example.

The tool schemas and the ``MailEnv`` result format here ARE the action space the
trained Ling-3.0-tiny must match at deploy time. Harness-Zero's central warning is
that trajectories collected under one action space transfer poorly to a target
with a different one, so the names, argument schemas, and JSON result shapes
defined in this file are the contract shared by the collector
(``collect_trajectories.py``), the SFT converter (``build_dataset.py``), and the
serving-time probe in the README — they never diverge.

Only the Python standard library is used so this module imports anywhere, including
the CPU test harness, without optional dependencies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# OpenAI function-calling tool schemas. Mirrors the Hermes-style action space the
# report's section 2.2 specifies: list / read / search / send. The ``execute_tool``
# table below is the single source of truth for both the name set and the router.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_emails",
            "description": "List emails in a mail folder, newest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Folder name, e.g. inbox / sent.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of emails to return.",
                    },
                },
                "required": ["folder", "limit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_email",
            "description": "Return the subject, sender, body, and attachments of one email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "ID of the email to read.",
                    },
                },
                "required": ["email_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search emails whose subject or body contains the keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Case-insensitive substring to match.",
                    },
                    "folder": {
                        "type": "string",
                        "description": "Folder to search; defaults to inbox when omitted.",
                    },
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to one or more recipients.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recipient addresses.",
                    },
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_NAMES = {tool["function"]["name"] for tool in TOOLS}


class ToolCallError(ValueError):
    """Raised when an assistant turn names an unknown tool or supplies bad arguments."""


@dataclass
class Email:
    """One fixture email in the in-memory mailbox."""

    id: str
    folder: str
    sender: str
    subject: str
    body: str
    attachments: list[str] = field(default_factory=list)


def _default_env() -> MailEnv:
    """A small, deterministic mailbox used by the seed data and the collector."""

    return MailEnv(
        emails=[
            Email(
                id="e1",
                folder="inbox",
                sender="alice@antgroup.com",
                subject="Q3 financial review meeting",
                body="Hi, the Q3 review is scheduled for Thursday 10am. Please bring the latest P&L.",
            ),
            Email(
                id="e2",
                folder="inbox",
                sender="hr@antgroup.com",
                subject="Annual benefits enrollment open",
                body="Benefits enrollment is open until next Friday. Log in to the portal to confirm your selections.",
            ),
            Email(
                id="e3",
                folder="inbox",
                sender="bob@antgroup.com",
                subject="Re: budget approval",
                body="Approved up to 50k. Please file the requisition through the finance portal.",
            ),
            Email(
                id="e4",
                folder="inbox",
                sender="corp@antgroup.com",
                subject="Office closed next Monday",
                body="The office will be closed next Monday for the holiday. No meetings should be scheduled.",
            ),
            Email(
                id="e5",
                folder="sent",
                sender="me@antgroup.com",
                subject="Re: Q3 financial review meeting",
                body="Got it, I will join with the P&L on Thursday.",
            ),
        ]
    )


class MailEnv:
    """In-memory executor for the mail tools.

    Results are plain ``dict`` values so they serialize straight to JSON tool
    results in the trajectory. The executor only reads from ``self.emails`` so a
    fresh ``MailEnv`` per task keeps trajectories deterministic.
    """

    def __init__(self, emails: list[Email] | None = None) -> None:
        self.emails = list(emails) if emails is not None else []
        self.sent: list[dict[str, Any]] = []

    @classmethod
    def default(cls) -> "MailEnv":
        return _default_env()

    def list_emails(self, *, folder: str, limit: int) -> dict[str, Any]:
        emails = [email for email in self.emails if email.folder == folder][:limit]
        return {
            "folder": folder,
            "count": len(emails),
            "emails": [
                {"id": email.id, "sender": email.sender, "subject": email.subject}
                for email in emails
            ],
        }

    def read_email(self, *, email_id: str) -> dict[str, Any]:
        for email in self.emails:
            if email.id == email_id:
                return {
                    "id": email.id,
                    "folder": email.folder,
                    "sender": email.sender,
                    "subject": email.subject,
                    "body": email.body,
                    "attachments": list(email.attachments),
                }
        return {"error": f"email_id {email_id!r} not found"}

    def search_emails(self, *, keyword: str, folder: str | None = None) -> dict[str, Any]:
        needle = keyword.lower()
        matches = []
        for email in self.emails:
            if folder is not None and email.folder != folder:
                continue
            if needle in email.subject.lower() or needle in email.body.lower():
                matches.append(
                    {"id": email.id, "sender": email.sender, "subject": email.subject}
                )
        return {"keyword": keyword, "folder": folder, "count": len(matches), "emails": matches}

    def send_email(self, *, to: list[str], subject: str, body: str) -> dict[str, Any]:
        record = {"to": list(to), "subject": subject, "body": body, "status": "sent"}
        self.sent.append(record)
        return {"status": "sent", "recipients": len(to)}


def execute_tool(name: str, arguments: dict[str, Any], env: MailEnv) -> dict[str, Any]:
    """Dispatch one parsed tool call to ``MailEnv``.

    ``name`` must be one of the exported ``TOOLS``; unknown names raise so the
    collector/converter can classify the turn as a format collapse (see report
    section 5.2). Argument coercion is intentionally permissive: the collector
    parses the Hermes JSON, then passes plain Python values through.
    """

    if name not in _TOOL_NAMES:
        raise ToolCallError(f"unknown tool name {name!r}; expected one of {sorted(_TOOL_NAMES)}")
    handler: Callable[..., dict[str, Any]]
    if name == "list_emails":
        handler = env.list_emails
    elif name == "read_email":
        handler = env.read_email
    elif name == "search_emails":
        handler = env.search_emails
    else:  # send_email
        handler = env.send_email
    try:
        return handler(**arguments)
    except TypeError as exc:
        raise ToolCallError(f"bad arguments for {name!r}: {exc}") from exc


def tool_result_to_text(result: dict[str, Any]) -> str:
    """Render a tool result for the ``tool`` role message in a trajectory.

    Stable, compact JSON so the SFT converter reproduces identical ``prompt``
    text across runs (the seed data relies on determinism).
    """

    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Hermes tool-call delimiters and parser (shared by the converter + collector)
# ---------------------------------------------------------------------------
#
# The on-wire tokens are assembled from code points so no literal angle-bracket
# marker ever appears in this source file -- keeps the action-space contract in
# one place and lets every consumer (build_dataset.py, mail_graph.py,
# collect_trajectories.py) reuse the exact same delimiters.

TOOL_CALL_OPEN = "".join([chr(60) * 3, "tool_call", chr(62)])
TOOL_CALL_CLOSE = "".join([chr(60), "/", "tool_call", chr(62)])
TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CALL_CLOSE), re.DOTALL
)


def parse_hermes_tool_calls(text: str) -> list[dict[str, Any]]:
    """Return parsed Hermes tool-call dicts in ``text`` (empty if none / malformed).

    Each matched block must decode to JSON with a string ``name`` and an object
    ``arguments``; anything else is treated as a format collapse. Shared by the
    collector (to drive ToolExecution) and the converter (to validate + render).
    """

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

    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"


__all__ = [
    "Email",
    "MailEnv",
    "TOOLS",
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "ToolCallError",
    "execute_tool",
    "parse_hermes_tool_calls",
    "render_hermes_tool_call",
    "tool_result_to_text",
]
