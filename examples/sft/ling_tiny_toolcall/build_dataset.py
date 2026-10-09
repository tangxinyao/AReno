"""Build data/{train,eval}.jsonl for the Ling-3.0-tiny Hermes tool-call SFT example.

This is the trajectory->SFT converter + filter from the report's section 4
(data conversion) and section 5.2/5.3 (quality filtering and the over-calling
risk). Pipeline:

1. A LangGraph workflow (mail_graph.py) runs Ling-3.0-tiny as the student over a
   set of mail tasks and records a full messages transcript per task
   (collect_trajectories.py emits this, one task per JSONL row).
2. This script filters those transcripts -- keep only Verifier-marked-done runs in
   which every assistant tool-call turn is well-formed Hermes (valid JSON, known
   tool name) -- and converts each surviving trajectory into ONE SFT row per
   assistant step. Tool-result spans are context, not labels, so they live in the
   prompt prefix; only the assistant's own turns are trained.
3. Negative (no-tool) chit-chat rows are added so the model does not learn "always
   call a tool" (the Llama-3.2-1B irrelevance-collapse failure mode from the report).

The default mode runs offline with NO collected raw_trajectories.jsonl: the small
set of hand-authored trajectories embedded as SEED_TRAJECTORIES + SEED_NO_TOOL_TASKS
below produces the committed data/{train,eval}.jsonl deterministically. Re-running
with optional collector output (--trajectories) replaces those seed trajectories;
the conversion code path is unchanged.

The on-wire Hermes tool-call tokens are assembled from code points (not written as
source literals) so this file does not embed the raw angle-bracket markers, which
keeps the source free of literal tool-call delimiters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import (
    MailEnv,
    TOOLS,
    TOOL_CALL_RE,
    ToolCallError,
    execute_tool,
    parse_hermes_tool_calls,
    render_hermes_tool_call,
    tool_result_to_text,
)

OUT = Path(__file__).resolve().parent / "data"

SYSTEM_PROMPT = (
    "You are a mail assistant. Use the provided tools to read the user's mail and "
    "answer questions about it. Emit one tool call per turn in the Hermes tool-call "
    "format. When you already have the answer, reply in plain text with no tool call."
)


def assert_valid_tool_call(text: str) -> list[dict[str, Any]]:
    """Raise if ``text`` contains any tool-call block that is not valid Hermes.

    Used as the training-data format guard (report section 5.2: every tool-call
    block must be fully legal JSON). A well-formed block returns its parsed call;
    a malformed block (bad JSON, unknown tool name, missing args) raises so the
    caller can drop the turn. The regex and delimiters are shared with the
    collector so collect-time and convert-time agree on what counts as valid.
    """

    known = {tool["function"]["name"] for tool in TOOLS}
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed tool-call JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("tool-call payload is not an object")
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str):
            raise ValueError("tool-call payload missing string name")
        if name not in known:
            raise ValueError(f"tool-call names unknown tool {name!r}")
        if not isinstance(arguments, dict):
            raise ValueError("tool-call payload missing arguments object")
    return parse_hermes_tool_calls(text)


def _assistant_turn_text(message: dict[str, Any]) -> str:
    """The supervised text for one assistant message: its tool-call block or prose.

    A tool-call message is rendered back into the exact on-wire Hermes block the
    model must learn to emit (via the shared renderer). Prose messages (the final
    natural-language answer) pass through unchanged.
    """

    if message.get("tool_calls"):
        blocks = []
        for call in message["tool_calls"]:
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            blocks.append(render_hermes_tool_call(function["name"], arguments))
        return "\n".join(blocks)
    return str(message.get("content") or "")


def _messages_to_prompt_text(messages: list[dict[str, Any]], *, tools: list[dict[str, Any]]) -> str:
    """Flatten the conversation up to the current turn into one text prompt.

    Stable, deterministic rendering so re-running the seed build leaves the
    committed jsonl byte-identical. Mirrors the action space the served model
    sees: the system prompt, the tool schemas as JSON, then the messages by role,
    with tool results drawn from ``tool_result_to_text``.
    """

    tool_specs = json.dumps(
        [tool["function"] for tool in tools], ensure_ascii=False, separators=(",", ":")
    )
    lines = [f"system: {SYSTEM_PROMPT}", f"system_tools: {tool_specs}"]
    for message in messages:
        role = message["role"]
        if role == "assistant":
            text = _assistant_turn_text(message)
            lines.append(f"assistant: {text}")
        elif role == "tool":
            lines.append(f"tool[{message.get('name', '')}]: {message.get('content', '')}")
        else:
            lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)


def clean_trajectory(trajectory: dict[str, Any], *, env: MailEnv) -> bool:
    """True if the trajectory is a Verifier-done run with all-valid tool calls.

    This is the report's section 5.3 positive selection: keep only completed,
    format-clean runs. Invalid JSON, unknown tool names, or a Verifier
    not-done flag drop the whole trajectory. Tool-result values are replayed
    through ``MailEnv`` so the recorded results must match the action space.
    """

    if not trajectory.get("done"):
        return False
    messages = trajectory.get("messages") or []
    if not messages or messages[0].get("role") != "user":
        return False
    assistant_turns = [m for m in messages if m.get("role") == "assistant"]
    if not any(m.get("tool_calls") for m in assistant_turns):
        return False
    for message in assistant_turns:
        text = _assistant_turn_text(message)
        if not text.strip():
            return False
        if message.get("tool_calls"):
            try:
                assert_valid_tool_call(text)
            except ValueError:
                return False
            for call in message["tool_calls"]:
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                try:
                    execute_tool(call["function"]["name"], args, env)
                except (ToolCallError, TypeError):
                    return False
    return True


def trajectory_to_rows(
    trajectory: dict[str, Any], *, tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Expand one trajectory into one SFT row per assistant step (step-wise).

    Each row's prompt is the conversation rendered up through the latest tool result;
    the response is the next assistant turn -- either a tool-call block or the
    final natural-language answer. Tool-result messages stay in the prompt as
    context; they are never the supervised target.
    """

    messages = list(trajectory["messages"])
    rows: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    step = 0
    for message in messages:
        if message["role"] != "assistant":
            context.append(message)
            continue
        response = _assistant_turn_text(message)
        if not response.strip():
            context.append(message)
            continue
        prompt = _messages_to_prompt_text(context, tools=tools)
        rows.append(
            {
                "prompt": prompt,
                "response": response.strip(),
                "lang": trajectory.get("lang", "en"),
                "trajectory_id": trajectory["task_id"],
                "step": step,
            }
        )
        step += 1
        context.append(message)
    return rows


def no_tool_rows(tasks: list[dict[str, Any]], *, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows teaching the model to answer directly when no tool is needed.

    The report cites a Llama-3.2-1B study where SFT over an all-tool-call dataset
    collapsed irrelevance accuracy because the model learned "tool available ->
    call it." These rows supply the missing negative: plain prompts whose expected
    response contains no tool-call block. Same system+tools prefix so the action
    space is present, but the target is a short direct answer.
    """

    tool_specs = json.dumps(
        [tool["function"] for tool in tools], ensure_ascii=False, separators=(",", ":")
    )
    rows: list[dict[str, Any]] = []
    for task in tasks:
        prompt = "\n".join(
            [
                f"system: {SYSTEM_PROMPT}",
                f"system_tools: {tool_specs}",
                f"user: {task['prompt']}",
            ]
        )
        rows.append(
            {
                "prompt": prompt,
                "response": task["response"],
                "lang": task.get("lang", "en"),
                "trajectory_id": task["task_id"],
                "step": 0,
            }
        )
    return rows


def task_to_eval_row(task: dict[str, Any], *, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Eval probe: system+tools+user prompt, reference = expected final answer.

    Tests recall on the action space, not prompt memorization (the eval task ids
    are disjoint from every train task id).
    """

    tool_specs = json.dumps(
        [tool["function"] for tool in tools], ensure_ascii=False, separators=(",", ":")
    )
    prompt = "\n".join(
        [
            f"system: {SYSTEM_PROMPT}",
            f"system_tools: {tool_specs}",
            f"user: {task['prompt']}",
        ]
    )
    return {
        "prompt": prompt,
        "reference": task["answer"],
        "lang": task.get("lang", "en"),
        "task_id": task["task_id"],
    }


# ---------------------------------------------------------------------------
# Seed data (embedded so the bundle imports + tests run with no GPU collector)
# ---------------------------------------------------------------------------
#
# Each seed trajectory is the exact messages transcript a successful collector
# run would produce for a small mail task, hand-authored against the MailEnv
# fixtures in tools.py so recorded tool results match the environment. Replace
# SEED_TRAJECTORIES with real collector output (--trajectories) at scale; the
# conversion code path is unchanged.

SEED_TRAJECTORIES: list[dict[str, Any]] = [
    {
        "task_id": "t1",
        "lang": "en",
        "done": True,
        "messages": [
            {"role": "user", "content": "Read the email about the Q3 financial review meeting."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "search_emails", "arguments": {"keyword": "Q3 financial review"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search_emails",
                "content": tool_result_to_text(
                    MailEnv.default().search_emails(keyword="Q3 financial review")
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {"name": "read_email", "arguments": {"email_id": "e1"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "read_email",
                "content": tool_result_to_text(MailEnv.default().read_email(email_id="e1")),
            },
            {
                "role": "assistant",
                "content": "The Q3 financial review meeting is scheduled for Thursday at 10am; "
                "Alice asked you to bring the latest P&L.",
            },
        ],
    },
    {
        "task_id": "t2",
        "lang": "en",
        "done": True,
        "messages": [
            {"role": "user", "content": "What is the latest email in my inbox?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "list_emails", "arguments": {"folder": "inbox", "limit": 1}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "list_emails",
                "content": tool_result_to_text(
                    MailEnv.default().list_emails(folder="inbox", limit=1)
                ),
            },
            {
                "role": "assistant",
                "content": "The latest email is from corp@antgroup.com, subject "
                "\"Office closed next Monday\".",
            },
        ],
    },
    {
        "task_id": "t3",
        "lang": "en",
        "done": True,
        "messages": [
            {"role": "user", "content": "Find anything about budget approval and tell me the decision."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "search_emails", "arguments": {"keyword": "budget approval"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search_emails",
                "content": tool_result_to_text(
                    MailEnv.default().search_emails(keyword="budget approval")
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {"name": "read_email", "arguments": {"email_id": "e3"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "read_email",
                "content": tool_result_to_text(MailEnv.default().read_email(email_id="e3")),
            },
            {
                "role": "assistant",
                "content": "Bob approved the budget up to 50k and asked you to file the requisition "
                "through the finance portal.",
            },
        ],
    },
    {
        "task_id": "t4",
        "lang": "zh",
        "done": True,
        "messages": [
            {"role": "user", "content": "帮我看一下收件箱里关于假期办公室关闭的邮件。"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "search_emails", "arguments": {"keyword": "office closed"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search_emails",
                "content": tool_result_to_text(
                    MailEnv.default().search_emails(keyword="office closed")
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {"name": "read_email", "arguments": {"email_id": "e4"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "read_email",
                "content": tool_result_to_text(MailEnv.default().read_email(email_id="e4")),
            },
            {
                "role": "assistant",
                "content": "下周一办公室因假期关闭,不要安排会议。",
            },
        ],
    },
]

SEED_NO_TOOL_TASKS: list[dict[str, Any]] = [
    {"task_id": "n1", "lang": "en", "prompt": "Thanks!", "response": "You're welcome."},
    {
        "task_id": "n2",
        "lang": "en",
        "prompt": "What can you do?",
        "response": "I can read and search your mail with the list_emails, read_email, "
        "search_emails, and send_email tools, and summarize what I find.",
    },
    {
        "task_id": "n3",
        "lang": "en",
        "prompt": "Who wrote Romeo and Juliet?",
        "response": "William Shakespeare wrote Romeo and Juliet.",
    },
    {"task_id": "n4", "lang": "zh", "prompt": "你好。", "response": "你好,有什么我可以帮你的吗?"},
]

SEED_EVAL_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "v1",
        "lang": "en",
        "prompt": "Read Alice's email about the review meeting and tell me when it is.",
        "answer": "Thursday at 10am.",
    },
    {
        "task_id": "v2",
        "lang": "en",
        "prompt": "Search for the benefits enrollment email and tell me the deadline.",
        "answer": "Next Friday.",
    },
    {
        "task_id": "v3",
        "lang": "zh",
        "prompt": "帮我看看收件箱里最新的会议邮件是什么时间。",
        "answer": "周四上午 10 点。",
    },
]


def build(
    *,
    trajectories: list[dict[str, Any]],
    no_tool_tasks: list[dict[str, Any]],
    eval_tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter, expand, and split trajectories + negatives into train/eval rows."""

    env = MailEnv.default()
    train: list[dict[str, Any]] = []
    for trajectory in trajectories:
        if not clean_trajectory(trajectory, env=env):
            continue
        train.extend(trajectory_to_rows(trajectory, tools=TOOLS))
    train.extend(no_tool_rows(no_tool_tasks, tools=TOOLS))
    evals = [task_to_eval_row(task, tools=TOOLS) for task in eval_tasks]
    return train, evals


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert collected mail-agent trajectories into Ling-3.0-tiny tool-call SFT data.",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=None,
        help="Optional JSONL of raw trajectories from collect_trajectories.py. "
        "Omit to build from the embedded seed data.",
    )
    args = parser.parse_args(argv)

    if args.trajectories is not None:
        trajectories = _load_jsonl(args.trajectories)
    else:
        trajectories = SEED_TRAJECTORIES
    no_tool_tasks = SEED_NO_TOOL_TASKS
    eval_tasks = SEED_EVAL_TASKS

    train, evals = build(
        trajectories=trajectories, no_tool_tasks=no_tool_tasks, eval_tasks=eval_tasks
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        with (OUT / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train={len(train)} eval={len(evals)} trajectories={len(trajectories)}")


if __name__ == "__main__":
    main()
