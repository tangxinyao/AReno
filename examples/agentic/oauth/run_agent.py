"""Native OAuth agent loop with a dynamic 10k context budget.

Skeleton shared with ``examples/agentic/office/run_agent.py``; the OAuth
swaps are the tool set (terminal/read_file/write_file/skill_view/clarify),
the workspace fixture (mock OAuth world per episode), the prompts and the
solved predicate (a confirmed AUTHENTICATED anchor via S5d, relayed by the
tool results as ``solved``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn
from areno.api.tokenizer import apply_chat_template_with_options, normalize_token_ids

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oauth_tools import (TOOLS, OAuthWorkspace, decode_tool_arguments, run_tool)  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = """You are a Google email connection agent working in an isolated workspace.
The workspace root for this task is {workspace}. Every tool resolves relative paths from this
root, and the workspace's own profile home (.hermes) holds the installed skills, credentials,
and tokens. The user's goal is to connect their Google email: figure out what the user needs,
follow the matching skill's setup procedure with the terminal tool, and verify the result with
the skill's own check command before answering. If a decision genuinely changes the procedure
(scope, protection enrollment, existing credentials), ask via the clarify tool instead of
guessing; do not re-ask what the user already answered. Create no files unrelated to the task.
Before every tool call write one short sentence saying what you are about to do and what you
expect from it, including any failure you already expect. There is no outbound network: only
this workspace and its local endpoints exist."""

FINAL_PROMPT = "Stop using tools. Briefly state the connection status confirmed by the check command and anything the user must still do themselves."
FINAL_RESPONSE_MAX_TOKENS = 256
TOKEN_BUDGET_ENV = "ARENO_OAUTH_TOKEN_BUDGET"


async def run_agent(ctx, batch) -> AgentTrajectory:
    """Run OAuth episodes until solved or the next turn would approach 10k tokens."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("The OAuth agentic demo requires `openai` and `httpx`.") from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=None,
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    workspaces: list[OAuthWorkspace] = []
    try:
        # Each episode's workspace owns one mock OAuth world on its own
        # ephemeral port (bound per workspace, so episodes never share state).
        workspaces = [OAuthWorkspace.from_record(dict(item.record)) for item in items]
        grouped = await asyncio.gather(
            *(
                _run_episode(item=item, workspace=workspace, client=client, tokenizer=ctx.get_tokenizer())
                for item, workspace in zip(items, workspaces, strict=True)
            )
        )
        return AgentTrajectory(turns=[turn for episode in grouped for turn in episode])
    finally:
        for workspace in workspaces:
            workspace.close()
        await client.close()


async def _run_episode(item, workspace: OAuthWorkspace, client, tokenizer) -> list[AgentTrajectoryTurn]:
    record = dict(item.record)
    context_budget = _context_budget(record)
    max_turns = int(record.get("max_turns", 12))
    messages = [{"role": "system", "content": _system_prompt(workspace)}, {"role": "user", "content": item.prompt}]
    turns = []
    finish_next = False

    for turn_index in range(max_turns):
        must_finish = finish_next or turn_index == max_turns - 1
        request_messages = [*messages, {"role": "user", "content": FINAL_PROMPT}] if must_finish else list(messages)
        request_tools = [] if must_finish else TOOLS
        context_tokens = _context_token_count(tokenizer, request_messages, tools=request_tools)
        remaining_tokens = context_budget - context_tokens
        if remaining_tokens <= 0:
            logger.warning(
                "OAuth context exhausted task=%s turn=%d context_tokens=%d budget=%d",
                record.get("id"),
                turn_index + 1,
                context_tokens,
                context_budget,
            )
            break
        kwargs = {
            "model": "policy",
            "messages": request_messages,
            "stream": False,
        }
        request_max_tokens = _request_max_tokens(must_finish=must_finish, remaining_tokens=remaining_tokens)
        if request_max_tokens is not None:
            kwargs["max_tokens"] = request_max_tokens
        if not must_finish:
            kwargs.update({"tools": TOOLS, "tool_choice": "auto"})
        response = await client.chat.completions.create(**kwargs)
        if _completion_was_truncated(response):
            logger.warning(
                "Dropping length-truncated OAuth turn task=%s turn=%d max_tokens=%s",
                record.get("id"),
                turn_index + 1,
                request_max_tokens if request_max_tokens is not None else "session",
            )
            break
        turn = AgentTrajectoryTurn(
            item=item,
            messages=request_messages,
            response=response,
            tools=[] if must_finish else TOOLS,
        )
        turns.append(turn)
        if must_finish:
            break

        assistant = _assistant_message(response)
        messages.append(assistant)
        call = _first_tool_call(assistant)
        if call is None:
            break
        # The tools block (subprocess, loopback HTTP, trajectory grading), so
        # run them off the event loop: the other episodes of this rollout batch
        # share it and would otherwise wait out every command.
        result = await asyncio.to_thread(
            run_tool,
            workspace,
            call["function"]["name"],
            decode_tool_arguments(call["function"].get("arguments")),
            messages=list(messages),
        )
        content = json.dumps(result, ensure_ascii=False, sort_keys=True)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": content,
            }
        )
        solved = bool(result.get("solved"))
        next_context_tokens = _context_token_count(tokenizer, messages, tools=TOOLS)
        finish_next = solved or next_context_tokens >= context_budget - FINAL_RESPONSE_MAX_TOKENS
        logger.debug(
            "OAuth turn task=%s turn=%d context_tokens=%d solved=%s",
            record.get("id"),
            turn_index + 1,
            next_context_tokens,
            solved,
        )
    return turns


def _system_prompt(workspace: OAuthWorkspace) -> str:
    """Bind the model-visible prompt to this episode's actual workspace."""

    return SYSTEM_PROMPT.format(workspace=workspace.root.as_posix())


def _completion_was_truncated(response) -> bool:
    """Return whether the server stopped before completing this response."""

    choices = getattr(response, "choices", None) or []
    return bool(choices and getattr(choices[0], "finish_reason", None) == "length")


def _request_max_tokens(*, must_finish: bool, remaining_tokens: int) -> int | None:
    """Use session max-new-tokens for tools; cap only the final text turn."""

    if not must_finish:
        return None
    return min(FINAL_RESPONSE_MAX_TOKENS, remaining_tokens)


def _context_budget(record: dict[str, Any]) -> int:
    """Resolve the runtime context budget, allowing an environment override."""

    raw_budget = os.getenv(TOKEN_BUDGET_ENV)
    budget = int(raw_budget) if raw_budget is not None else int(record.get("context_budget", 10_000))
    if budget < 1:
        raise ValueError(f"{TOKEN_BUDGET_ENV} must be a positive integer")
    return budget


def _assistant_message(response) -> dict[str, Any]:
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": decode_tool_arguments(call.function.arguments),
                },
            }
            for call in (message.tool_calls or [])[:1]
        ],
    }


def _first_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    calls = message.get("tool_calls") or []
    return calls[0] if calls else None


def _context_token_count(tokenizer, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]]) -> int:
    """Count the exact next-request tokens with the rollout tokenizer."""

    encoded = apply_chat_template_with_options(
        tokenizer,
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(normalize_token_ids(encoded))
