"""LangGraph workflow / harness for the Hermes Gmail-login tool-call SFT example.

This is the report's "Planner -> Router -> ToolExecution -> Verifier -> Summarizer"
topology. Each node runs the student model (Ling-3.0-tiny, via an OpenAI-compatible
chat client) and records the result into the shared ``StateGraph`` messages list.
A ``MemorySaver`` checkpointer captures every transition, so the checkpoint history
IS the executed trajectory; ``collect_trajectories.py`` replays it.

The system prompt is a faithful, offline port of the Hermes CLI system prompt for
this workflow: the SOUL.md identity block, the help guidance, and the skills index
(``## Skills (mandatory)`` + ``<available_skills>``). Building it here -- rather
than inventing a toy prompt -- is the Harness-Zero action-space-match constraint:
the student must see at collect time the same instructions and tool schemas it
will see at deployment.

LangGraph is imported lazily so this module imports cleanly without it installed;
only the collector actually runs the graph. The chat client uses only the standard
library so the example never needs the ``openai`` PyPI package -- any
OpenAI-compatible endpoint (including ``areno serve``) works.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from tools import (
    AgentEnv,
    TOOLS,
    ToolCallError,
    execute_tool,
    parse_hermes_tool_calls,
    tool_result_to_text,
)

MAX_STEPS = 8  # cap a runaway think-act-observe loop before it breaks the budget

# ---------------------------------------------------------------------------
# System prompt (ported from Hermes: SOUL.md identity + skills index)
# ---------------------------------------------------------------------------

# The SOUL.md identity block Hermes prepends to every session. Vendored verbatim
# from ~/.hermes/SOUL.md so the collected episodes carry the real persona text.
SOUL_IDENTITY = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below. "
    "Be targeted and efficient in your exploration and investigations."
)

# The Hermes-self-help guidance block (agent/prompt_builder.py:HERMES_AGENT_HELP_GUIDANCE).
HERMES_AGENT_HELP_GUIDANCE = (
    "You run on Hermes Agent (by Nous Research). When the user needs help with "
    "Hermes itself -- configuring, setting up, using, extending, or "
    "troubleshooting it -- or when you need to understand your own features, "
    "tools, or capabilities, load the `hermes-agent` skill with "
    "skill_view(name='hermes-agent') for additional guidance and proven "
    "workflows."
)

# The shared preamble of the Hermes skills index (agent/prompt_builder.py).
_SKILLS_PREAMBLE = (
    "## Skills (mandatory)\n"
    "Before replying, scan the skills below. If a skill matches or is even "
    "partially relevant to your task, you MUST load it with skill_view(name) and "
    "follow its instructions. Err on the side of loading -- it is always better "
    "to have context you don't need than to miss critical steps, pitfalls, or "
    "established workflows. Skills contain specialized knowledge -- API "
    "endpoints, tool-specific commands, and proven workflows that outperform "
    "general-purpose approaches. Load the skill even if you think you could "
    "handle the task with basic tools like web_search or terminal. Skills also "
    "encode the user's preferred approach, conventions, and quality standards "
    "for tasks like code review, planning, and testing -- load them even for "
    "tasks you already know how to do, because the skill defines how it should "
    "be done here.\n"
    "If a skill has issues, fix it with skill_manage(action='patch').\n"
    "After difficult/iterative tasks, offer to save as a skill."
)


def build_skills_system_prompt(env: AgentEnv) -> str:
    """Build the ``## Skills (mandatory)`` block with the vendored skills index.

    Mirrors ``agent/prompt_builder.py:build_skills_system_prompt``: the preamble,
    then an ``<available_skills>`` block grouped by category. The example vendors
    two email skills (``himalaya`` and ``google-workspace``) whose overlapping
    descriptions are exactly what makes the model ask the user to choose.
    """

    by_category: dict[str, list[Any]] = {}
    for record in env.skills:
        by_category.setdefault(record.category, []).append(record)
    lines: list[str] = []
    for category in sorted(by_category):
        lines.append(f"  {category}:")
        for record in sorted(by_category[category], key=lambda r: r.name):
            lines.append(f"    - {record.name}: {record.description}")
    return (
        _SKILLS_PREAMBLE
        + "\n\n<available_skills>\n"
        + "\n".join(lines)
        + "\n</available_skills>\n\n"
        + "Only proceed without loading a skill if genuinely none are relevant "
        "to the task."
    )


def build_system_prompt(env: AgentEnv) -> str:
    """Assemble the full Hermes-style system prompt for this workflow."""

    return "\n\n".join(
        [
            SOUL_IDENTITY,
            HERMES_AGENT_HELP_GUIDANCE,
            build_skills_system_prompt(env),
        ]
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class MailState:
    """Mutable workflow state shared across nodes.

    Only plain, JSON-serializable values live here: the LangGraph ``MemorySaver``
    checkpointer serializes state to msgpack, so the per-episode ``AgentEnv`` must
    NOT be a field. It is threaded in as a build-time closure value instead.

    ``messages`` is the OpenAI-style transcript the collector exports verbatim.
    ``done`` is the Verifier's verdict -- a trajectory is kept for training only
    when ``done`` is True and every assistant tool-call turn parsed cleanly.
    """

    user_query: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    done: bool = False


# ---------------------------------------------------------------------------
# OpenAI-compatible client (stdlib only, no ``openai`` dependency)
# ---------------------------------------------------------------------------


class OpenAIClient:
    """Minimal OpenAI /v1/chat/completions client over urllib.

    Only the fields the workflow needs are implemented: messages, tools, and
    max_tokens. ``tools`` is passed straight through so the served endpoint
    renders the schemas into the prompt the same way deployment will.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "areno-agentic") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if tools is not None:
            body["tools"] = tools
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # surface server error text for debugging
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"chat request failed ({exc.code}): {detail}") from exc


# ---------------------------------------------------------------------------
# Nodes: each is a pure function of state that appends one message
# ---------------------------------------------------------------------------


def _build_request_messages(state: MailState, *, env: AgentEnv) -> list[dict[str, Any]]:
    """Prepend the system message to the recorded transcript for the next call."""

    return [{"role": "system", "content": build_system_prompt(env)}, *state.messages]


def _assistant_message_from_response(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAI chat response into a transcript message.

    If the endpoint parsed tool calls (``message.tool_calls``), keep them; otherwise
    treat the assistant content as prose. The collector stores ``content`` too so
    the converter can render either form back into on-wire Hermes blocks.

    ``areno serve`` models launched with ``--disable-thinking`` may still emit the
    visible answer in ``reasoning_content`` while leaving ``content`` empty. On a
    pure prose turn (no tool calls) that would drop the final answer, so fall back
    to ``reasoning_content`` for the content. Tool-call turns are left alone: their
    payload is the tool call itself, and splicing thinking text into the content
    would pollute the training sample.
    """

    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    if not tool_calls and not content.strip():
        content = message.get("reasoning_content") or ""
    normalized = {"role": "assistant", "content": content}
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    return normalized


def planner_node(state: MailState, *, client: OpenAIClient, env: AgentEnv) -> MailState:
    """Kick off the think-act-observe loop: seed the transcript with the user query."""

    del client, env
    if state.messages:
        return state  # only seeds on the first transition
    state.messages.append({"role": "user", "content": state.user_query})
    return state


def router_node(state: MailState, *, client: OpenAIClient, env: AgentEnv) -> MailState:
    """Ask the student for the next action given the transcript so far.

    This is the "decide which tool to call / whether to answer" step. One
    assistant message is appended (tool-call block or prose answer).
    """

    if state.step >= MAX_STEPS:
        state.done = False
        return state
    response = client.chat(_build_request_messages(state, env=env), tools=TOOLS, max_tokens=512)
    state.messages.append(_assistant_message_from_response(response))
    state.step += 1
    return state


def tool_execution_node(state: MailState, *, client: OpenAIClient, env: AgentEnv) -> MailState:
    """Execute any tool call in the latest assistant message; append tool results.

    If the latest assistant turn has no tool call, this is a no-op -- the Verifier
    then decides whether the prose answer completes the task.
    """

    del client
    last = state.messages[-1] if state.messages else None
    if not last or last.get("role") != "assistant":
        return state
    if last.get("tool_calls"):
        for index, call in enumerate(last["tool_calls"]):
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            try:
                result = execute_tool(function["name"], arguments, env)
            except (ToolCallError, TypeError) as exc:
                result = {"error": str(exc)}
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"call_{state.step}_{index}",
                    "name": function["name"],
                    "content": tool_result_to_text(result),
                }
            )
    return state


def verifier_node(state: MailState, *, client: OpenAIClient, env: AgentEnv) -> MailState:
    """Classify the run: last assistant turn with prose (no tool call) means done.

    The Verifier decides "task complete" vs "keep calling." It sets the ``done``
    flag only; routing is decided by the graph's conditional edges. When the loop
    lands on a prose answer within budget, that prose IS the summary and no extra
    turn is needed -- the Summarizer only acts as a forced backstop when the
    budget is spent before any prose answer appeared.
    """

    del client, env
    last = state.messages[-1] if state.messages else None
    over_budget = state.step >= MAX_STEPS
    state.done = bool(
        last
        and last.get("role") == "assistant"
        and not last.get("tool_calls")
        and not over_budget
    )
    return state


def summarizer_node(state: MailState, *, client: OpenAIClient, env: AgentEnv) -> MailState:
    """Forced backstop: ask for one prose answer when the loop ended without one.

    Execute-Summarize split: Execute (Planner/Router/ToolExecution above) produced
    the trajectory; this Summarize turn produces the prose final answer the eval
    compares against -- a single forced response, tools disabled, so the model
    answers from gathered observations instead of looping on another tool call.
    """

    response = client.chat(_build_request_messages(state, env=env), tools=None, max_tokens=512)
    state.messages.append(_assistant_message_from_response(response))
    state.done = not state.messages[-1].get("tool_calls")
    return state


# ---------------------------------------------------------------------------
# Graph construction (langgraph imported lazily)
# ---------------------------------------------------------------------------


def build_graph(client: OpenAIClient, env: AgentEnv) -> Any:
    """Compile the Planner->Router->ToolExecution->Verifier->Summarizer graph.

    ``env`` is the per-episode harness (skills + scripted decisions); it is a
    build-time closure value, NOT a state field, because the ``MemorySaver``
    checkpointer serializes state and ``AgentEnv`` is not msgpack-serializable.
    Build a fresh graph per episode to swap env.

    ``langgraph`` is imported here so the module imports without it installed.
    Conditional edges implement the loop:

      router -> (tool call?) tool_execution -> router                 # keep acting
              -> (prose/done) verifier -> END                         # that prose is the summary
      (budget spent, no prose)           verifier -> summarizer -> END  # forced backstop answer
    """

    try:
        from langgraph.graph import END, StateGraph
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:  # pragma: no cover - exercised by the collector path
        raise ImportError(
            "mail_graph requires langgraph: pip install langgraph (or "
            "langgraph + langgraph-checkpoint). The rest of the example bundle "
            "does not need it."
        ) from exc

    def _after_router(state: MailState) -> str:
        """The Router turn produced either a tool call (-> execute) or prose (-> verify)."""

        last = state.messages[-1] if state.messages else None
        if state.step >= MAX_STEPS:
            return "verifier"
        if last and last.get("role") == "assistant" and last.get("tool_calls"):
            return "tool_execution"
        return "verifier"

    def _after_tool_execution(state: MailState) -> str:
        """After executing a tool, loop back to the Router for the next think step."""

        if state.step >= MAX_STEPS:
            return "verifier"
        return "router"

    def _after_verifier(state: MailState) -> str:
        """Done means the prose answer is in already -> exit; otherwise force a Summary backstop."""

        return END if state.done else "summarizer"

    graph = StateGraph(MailState)
    graph.add_node("planner", lambda s: planner_node(s, client=client, env=env))
    graph.add_node("router", lambda s: router_node(s, client=client, env=env))
    graph.add_node("tool_execution", lambda s: tool_execution_node(s, client=client, env=env))
    graph.add_node("verifier", lambda s: verifier_node(s, client=client, env=env))
    graph.add_node("summarizer", lambda s: summarizer_node(s, client=client, env=env))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "router")
    graph.add_conditional_edges(
        "router",
        _after_router,
        {"tool_execution": "tool_execution", "verifier": "verifier"},
    )
    graph.add_conditional_edges(
        "tool_execution",
        _after_tool_execution,
        {"router": "router", "verifier": "verifier"},
    )
    graph.add_conditional_edges(
        "verifier",
        _after_verifier,
        {"summarizer": "summarizer", END: END},
    )
    graph.add_edge("summarizer", END)
    compiled = graph.compile(checkpointer=MemorySaver())
    return compiled


__all__ = [
    "MailState",
    "MAX_STEPS",
    "OpenAIClient",
    "build_graph",
    "build_system_prompt",
    "build_skills_system_prompt",
    "SOUL_IDENTITY",
    "HERMES_AGENT_HELP_GUIDANCE",
]