"""Build data/{train,eval}.jsonl for the Ling-3.0-tiny Hermes tool-call SFT example.

This is the trajectory->SFT converter + filter. Pipeline:

1. A LangGraph workflow (mail_graph.py) runs Ling-3.0-tiny as the student over a
   set of Hermes tasks and records a full messages transcript per task
   (collect_trajectories.py emits this, one task per JSONL row).
2. This script filters those transcripts -- keep only Verifier-marked-done runs in
   which every assistant tool-call turn is well-formed Hermes (valid JSON, known
   tool name) -- and converts each surviving trajectory into ONE SFT row per
   assistant step. Tool-result spans are context, not labels, so they live in the
   prompt prefix; only the assistant's own turns are trained.
3. Negative (no-tool) chit-chat rows are added so the model does not learn "always
   call a tool" (the irrelevance-collapse failure mode).

The default mode runs offline with NO collected raw_trajectories.jsonl: the small
set of hand-authored trajectories embedded as SEED_TRAJECTORIES + SEED_NO_TOOL_TASKS
below produces the committed data/{train,eval}.jsonl deterministically. Re-running
with optional collector output (--trajectories) replaces those seed trajectories;
the conversion code path is unchanged.

The action space is the real Hermes toolset -- ``skills_list``, ``skill_view``,
``clarify``, ``terminal`` (see tools.py). The canonical episode is the Gmail-login
workflow: the user asks to log in to Gmail, the model loads both email skills
(``google-workspace`` and ``himalaya``), asks the user which route to take, then
drives the Google Workspace OAuth setup through ``terminal``.

The on-wire Hermes tool-call tokens are assembled from code points (not written as
source literals) so this file does not embed the raw angle-bracket markers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import (
    AgentEnv,
    FIXTURE_ACCOUNT,
    FIXTURE_CLIENT_SECRET,
    TOOLS,
    TOOL_CALL_RE,
    ToolCallError,
    execute_tool,
    parse_hermes_tool_calls,
    render_hermes_tool_call,
    tool_result_to_text,
)

OUT = Path(__file__).resolve().parent / "data"


def _prompt_fragments() -> tuple[str, str, str]:
    """Import the system-prompt fragments from mail_graph regardless of sys.path.

    ``build_dataset`` runs both as a script (``python build_dataset.py``, where
    ``mail_graph`` is importable because the file's directory is on ``sys.path``)
    and via the CPU test loader (which imports this module by absolute path and
    then restores ``sys.path``). The absolute-path fallback keeps both paths
    working without duplicating the persona strings.
    """

    try:
        from mail_graph import _SKILLS_PREAMBLE, HERMES_AGENT_HELP_GUIDANCE, SOUL_IDENTITY

        return SOUL_IDENTITY, HERMES_AGENT_HELP_GUIDANCE, _SKILLS_PREAMBLE
    except ModuleNotFoundError:
        import importlib.util
        import sys

        module_path = Path(__file__).resolve().parent / "mail_graph.py"
        spec = importlib.util.spec_from_file_location("ling_tiny_toolcall_mail_graph", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.SOUL_IDENTITY, module.HERMES_AGENT_HELP_GUIDANCE, module._SKILLS_PREAMBLE


def _system_prompt_text() -> str:
    """The identity + help-guidance + skills preamble block used as the prompt prefix."""

    soul, help_guidance, skills_preamble = _prompt_fragments()
    return "\n\n".join([soul, help_guidance, skills_preamble])


def assert_valid_tool_call(text: str) -> list[dict[str, Any]]:
    """Raise if ``text`` contains any tool-call block that is not valid Hermes.

    Used as the training-data format guard: every tool-call block must be fully
    legal JSON. A well-formed block returns its parsed call; a malformed block
    (bad JSON, unknown tool name, missing args) raises so the caller can drop the
    turn. The regex and delimiters are shared with the collector so collect-time
    and convert-time agree on what counts as valid.
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
    # The system text is assembled here without the live env (the prompt prefix
    # stores the identity + tool schemas; the per-episode skills index is injected
    # by the served template at train time).
    system = _system_prompt_text()
    lines = [f"system: {system}", f"system_tools: {tool_specs}"]
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


def clean_trajectory(trajectory: dict[str, Any], *, env: AgentEnv) -> bool:
    """True if the trajectory is a Verifier-done run with all-valid tool calls.

    Positive selection: keep only completed, format-clean runs. Invalid JSON,
    unknown tool names, or a Verifier not-done flag drop the whole trajectory.
    Tool results are replayed through ``AgentEnv`` so the recorded results must
    match the action space.
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

    Each row's prompt is the conversation rendered up through the latest tool
    result; the response is the next assistant turn -- either a tool-call block or
    the final natural-language answer. Tool-result messages stay in the prompt as
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

    The action space is present (system + tools prefix), but the target is a short
    direct answer with no tool-call block -- the missing negative that guards
    against "tool available -> call it" collapse.
    """

    tool_specs = json.dumps(
        [tool["function"] for tool in tools], ensure_ascii=False, separators=(",", ":")
    )
    system = _system_prompt_text()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        prompt = "\n".join(
            [
                f"system: {system}",
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
    system = _system_prompt_text()
    prompt = "\n".join(
        [
            f"system: {system}",
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
# Each seed trajectory is the exact messages transcript a successful collector run
# would produce, hand-authored against the ``AgentEnv`` harness in tools.py so
# recorded tool results match the environment. Replace SEED_TRAJECTORIES with real
# collector output (--trajectories) at scale; the conversion code path is unchanged.
#
# The canonical episode is the Gmail-login flow (t1): skill_view both email
# skills -> clarify (which route) -> clarify (client-secret path) -> terminal
# --client-secret -> terminal --auth-url -> prose (send the URL) -> [user pastes
# the code] -> terminal --auth-code -> final prose (logged in). The remaining seed
# tasks are shorter mail questions over the same action space.


def _gw_skill_view_payload() -> dict[str, Any]:
    """The skill_view result for google-workspace (setup_needed, per the harness)."""

    return AgentEnv.default().skill_view(name="google-workspace")


def _himalaya_skill_view_payload() -> dict[str, Any]:
    return AgentEnv.default().skill_view(name="himalaya")


def _tool_message(tool_call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": tool_result_to_text(result),
    }


def _seed_gmail_login() -> dict[str, Any]:
    """The canonical Gmail-login episode (see the example README for the prose)."""

    auth_url = _auth_url_for_seed()
    auth_code_url = (
        "http://localhost:1/?state=EXAMPLE_STATE&iss=https://accounts.google.com"
        "&code=4/0AXlqoi70OgsQBmtvCP029CYQuOs6-BgaW8GzPnnLqmDUf_mUKi2RCI-N0buDTsGbksJp9A"
        "&scope=https://www.googleapis.com/auth/gmail.readonly"
    )
    return {
        "task_id": "t1",
        "lang": "zh",
        "done": True,
        "messages": [
            {"role": "user", "content": "帮我登录我的谷歌邮箱"},
            {
                "role": "assistant",
                "content": "我先查看一下可用的 Google 邮箱工具技能，了解正确的认证方式。",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "skill_view", "arguments": {"name": "google-workspace"}},
                    },
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {"name": "skill_view", "arguments": {"name": "himalaya"}},
                    },
                ],
            },
            _tool_message("call_1", "skill_view", _gw_skill_view_payload()),
            _tool_message("call_2", "skill_view", _himalaya_skill_view_payload()),
            {
                "role": "assistant",
                "content": "有两个功能类似的邮箱技能，先确认你的需求。",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_3",
                        "function": {
                            "name": "clarify",
                            "arguments": {
                                "question": "你只需要登录/收发 Gmail 邮件，还是也想用日历、云盘、文档等其他 Google 服务？",
                                "choices": [
                                    "只要邮箱（IMAP/SMTP，2分钟搞定）",
                                    "邮箱 + 日历/云盘/文档等（OAuth 授权）",
                                ],
                            },
                        },
                    }
                ],
            },
            _tool_message(
                "call_3",
                "clarify",
                {
                    "question": "你只需要登录/收发 Gmail 邮件，还是也想用日历、云盘、文档等其他 Google 服务？",
                    "choices_offered": [
                        "只要邮箱（IMAP/SMTP，2分钟搞定）",
                        "邮箱 + 日历/云盘/文档等（OAuth 授权）",
                    ],
                    "user_response": "邮箱 + 日历/云盘/文档等（OAuth 授权）",
                },
            ),
            {
                "role": "assistant",
                "content": "好，走 OAuth 授权路线。我需要你的 Google Cloud OAuth 客户端凭证（client_secret JSON）。",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_4",
                        "function": {
                            "name": "clarify",
                            "arguments": {
                                "question": "Google Cloud 的 OAuth 客户端凭证（client_secret JSON）文件路径是什么？",
                            },
                        },
                    }
                ],
            },
            _tool_message(
                "call_4",
                "clarify",
                {
                    "question": "Google Cloud 的 OAuth 客户端凭证（client_secret JSON）文件路径是什么？",
                    "choices_offered": [],
                    "user_response": f"文件路径是：{FIXTURE_CLIENT_SECRET}",
                },
            ),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_5",
                        "function": {
                            "name": "terminal",
                            "arguments": {
                                "command": (
                                    "python ~/.hermes/skills/productivity/google-workspace/"
                                    f"scripts/setup.py --client-secret {FIXTURE_CLIENT_SECRET}"
                                )
                            },
                        },
                    }
                ],
            },
            _tool_message(
                "call_5",
                "terminal",
                {
                    "output": "OK: Client secret saved to ~/.hermes/google_client_secret.json",
                    "exit_code": 0,
                    "error": None,
                },
            ),
            {
                "role": "assistant",
                "content": "凭证已保存。现在生成授权链接，你需要在浏览器里打开它完成授权。",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_6",
                        "function": {
                            "name": "terminal",
                            "arguments": {
                                "command": (
                                    "python ~/.hermes/skills/productivity/google-workspace/"
                                    "scripts/setup.py --auth-url"
                                )
                            },
                        },
                    }
                ],
            },
            _tool_message(
                "call_6",
                "terminal",
                {"output": auth_url, "exit_code": 0, "error": None},
            ),
            {
                "role": "assistant",
                "content": (
                    "请在浏览器打开下面的链接完成授权，然后把你跳转到的完整 URL 粘贴回来：\n\n"
                    + auth_url
                ),
            },
            {"role": "user", "content": auth_code_url},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_7",
                        "function": {
                            "name": "terminal",
                            "arguments": {
                                "command": (
                                    "python ~/.hermes/skills/productivity/google-workspace/"
                                    f"scripts/setup.py --auth-code '{auth_code_url}'"
                                )
                            },
                        },
                    }
                ],
            },
            _tool_message(
                "call_7",
                "terminal",
                {
                    "output": (
                        "OK: Authenticated. Token saved to ~/.hermes/google_token.json\n"
                        "AUTHENTICATED: Token valid at ~/.hermes/google_token.json"
                    ),
                    "exit_code": 0,
                    "error": None,
                },
            ),
            {
                "role": "assistant",
                "content": (
                    f"登录完成，你的邮箱账号是 {FIXTURE_ACCOUNT}，令牌保存在 "
                    "~/.hermes/google_token.json 并会自动刷新。现在可以帮你读邮件、"
                    "发邮件、查日程、传云盘文件等，需要做什么直接说。"
                ),
            },
        ],
    }


def _auth_url_for_seed() -> str:
    from tools import FIXTURE_AUTH_URL

    return FIXTURE_AUTH_URL


def _seed_unread_digest() -> dict[str, Any]:
    """A shorter follow-up episode: list unread mail via the API script."""

    gmail_search = (
        "python ~/.hermes/skills/productivity/google-workspace/scripts/google_api.py "
        "gmail search \"is:unread\" --max 3"
    )
    return {
        "task_id": "t2",
        "lang": "zh",
        "done": True,
        "messages": [
            {"role": "user", "content": "看看我的谷歌邮箱里有哪些未读邮件。"},
            {
                "role": "assistant",
                "content": "我先加载 Google Workspace 技能，了解查询邮件的命令。",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "skill_view", "arguments": {"name": "google-workspace"}},
                    }
                ],
            },
            _tool_message("call_1", "skill_view", _gw_skill_view_payload()),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {"name": "terminal", "arguments": {"command": gmail_search}},
                    }
                ],
            },
            _tool_message(
                "call_2",
                "terminal",
                {
                    "output": json.dumps(
                        [
                            {
                                "id": "1a11efa219fdd936",
                                "from": "McKinsey Weekend Read <publishing@email.mckinsey.com>",
                                "subject": "Is your organization standing between AI and value?",
                                "date": "Fri, 9 Oct 2026 04:44:30 +0000",
                                "labels": ["UNREAD", "CATEGORY_UPDATES", "INBOX"],
                            },
                            {
                                "id": "1a11e554e743d131",
                                "from": "Upwork Notification <donotreply@upwork.com>",
                                "subject": "New job alert: AI and Automation Developer",
                                "date": "Fri, 09 Oct 2026 01:44:28 +0000",
                                "labels": ["UNREAD", "CATEGORY_UPDATES", "INBOX"],
                            },
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "exit_code": 0,
                    "error": None,
                },
            ),
            {
                "role": "assistant",
                "content": (
                    "你有 2 封未读邮件：一封来自 McKinsey Weekend Read（主题：Is your "
                    "organization standing between AI and value?），另一封来自 Upwork "
                    "Notification（主题：New job alert: AI and Automation Developer）。"
                ),
            },
        ],
    }


def _seed_reply_by_skill_route() -> dict[str, Any]:
    """A himalaya-route episode: the user picks the IMAP/SMTP path instead."""

    return {
        "task_id": "t3",
        "lang": "zh",
        "done": True,
        "messages": [
            {"role": "user", "content": "我想在终端里收邮件，用 himalaya 怎么配置？"},
            {
                "role": "assistant",
                "content": "我先加载 himalaya 技能，看它的配置说明。",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "skill_view", "arguments": {"name": "himalaya"}},
                    }
                ],
            },
            _tool_message("call_1", "skill_view", _himalaya_skill_view_payload()),
            {
                "role": "assistant",
                "content": (
                    "himalaya 需要一个配置文件 ~/.config/himalaya/config.toml，里面填 IMAP/SMTP "
                    "服务器和账号。最快的方式是运行 `himalaya account configure` 向导，它会逐步问你"
                    "邮箱地址、服务器和密码（密码建议用 `pass` 或系统钥匙串保存）。配置里的 folder.aliases "
                    "需要把 inbox/sent/drafts/trash 映射到服务端的真实文件夹名，Gmail 用户尤其要注意。"
                ),
            },
        ],
    }


def _seed_refuse_no_login() -> dict[str, Any]:
    """An episode where the user only wants one known message (no full login)."""

    return {
        "task_id": "t4",
        "lang": "zh",
        "done": True,
        "messages": [
            {"role": "user", "content": "帮我看下谷歌邮箱里最近的未读邮件标题就行，不用登录那么麻烦。"},
            {
                "role": "assistant",
                "content": "我先看下有没有更轻量的邮箱技能。",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "skill_view", "arguments": {"name": "himalaya"}},
                    }
                ],
            },
            _tool_message("call_1", "skill_view", _himalaya_skill_view_payload()),
            {
                "role": "assistant",
                "content": (
                    "不管是 himalaya 还是 Google Workspace，都需要先授权一次才能读你的邮箱——"
                    "没有免登录的路径。himalaya 用 Gmail 应用专用密码，2 分钟；Google Workspace 用 "
                    "OAuth，能顺带用日历/云盘。你选哪个？"
                ),
            },
        ],
    }


SEED_TRAJECTORIES: list[dict[str, Any]] = [
    _seed_gmail_login(),
    _seed_unread_digest(),
    _seed_reply_by_skill_route(),
    _seed_refuse_no_login(),
]

SEED_NO_TOOL_TASKS: list[dict[str, Any]] = [
    {"task_id": "n1", "lang": "en", "prompt": "Thanks!", "response": "You're welcome."},
    {
        "task_id": "n2",
        "lang": "zh",
        "prompt": "你能做什么？",
        "response": (
            "我可以帮你读/发 Gmail 邮件、查日历、传云盘文件、编辑文档表格。需要的话我也可以"
            "在终端里跑命令、查看技能说明或加载脚本。"
        ),
    },
    {
        "task_id": "n3",
        "lang": "en",
        "prompt": "Who wrote Romeo and Juliet?",
        "response": "William Shakespeare wrote Romeo and Juliet.",
    },
    {"task_id": "n4", "lang": "zh", "prompt": "你好。", "response": "你好，有什么我可以帮你的吗？"},
]

SEED_EVAL_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "v1",
        "lang": "zh",
        "prompt": "帮我登录我的谷歌邮箱。",
        "answer": "需要先确认你要哪些 Google 服务，然后提供 client_secret JSON 完成 OAuth 授权。",
    },
    {
        "task_id": "v2",
        "lang": "zh",
        "prompt": "我只想用终端收 Gmail，怎么配置？",
        "answer": "加载 himalaya 技能，用 Gmail 应用专用密码配置 IMAP/SMTP。",
    },
    {
        "task_id": "v3",
        "lang": "en",
        "prompt": "List my unread Gmail messages.",
        "answer": "Load the google-workspace skill and run google_api.py gmail search \"is:unread\".",
    },
]


def build(
    *,
    trajectories: list[dict[str, Any]],
    no_tool_tasks: list[dict[str, Any]],
    eval_tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter, expand, and split trajectories + negatives into train/eval rows."""

    env = AgentEnv.default()
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
        description="Convert collected Hermes tool-call trajectories into Ling-3.0-tiny SFT data.",
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