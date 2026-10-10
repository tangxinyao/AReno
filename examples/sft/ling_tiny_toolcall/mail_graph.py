"""LangGraph workflow / harness for the mail tool-call distillation example.

This is the report's section 2.1 "Planner -> ToolRouter -> ToolExecution ->
Verifier -> Summarizer" topology. The workflow IS the harness: each node runs the
student model (Ling-3.0-tiny, via an OpenAI-compatible chat client) and records
the result into the shared ``StateGraph`` messages list. A ``MemorySaver``
checkpointer captures every state transition, so the checkpoint history IS the
executed trajectory (report section 3.1) -- ``collect_trajectories.py`` replays it.

LangGraph is imported lazily so this module (and the whole example bundle) imports
cleanly without it installed; only the collector actually runs the graph. The
chat client is also constructed lazily and uses only the standard library so the
example never needs the ``openai`` PyPI package -- any OpenAI-compatible endpoint
(including ``areno serve``) works.

Why route the collector through ``areno serve``'s OpenAI-compatible endpoint: that
endpoint renders the ``tools`` field into the prompt through the chat template
(``areno/api/openai_chat.py:65``), exactly as production serving does. Keeping the
action space the model sees at collect time identical to the one it sees at
deploy time is the Harness-Zero action-space-match constraint (report 5.1).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from tools import (
    MailEnv,
    TOOLS,
    ToolCallError,
    execute_tool,
    parse_hermes_tool_calls,
    tool_result_to_text,
)

MAX_STEPS = 6  # cap a runaway think-act-observe loop before it breaks the budget


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class MailState:
    """Mutable workflow state shared across nodes.

    Only plain, JSON-serializable values live here: the LangGraph ``MemorySaver``
    checkpointer serializes state to msgpack, so the per-task ``MailEnv`` (with
    Python objects and the HTTP client it is not part of) must NOT be a field. It
    is threaded in as a build-time closure value instead (see ``build_graph``).

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

    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
             max_tokens: int = 256) -> dict[str, Any]:
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
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # surface server error text for debugging
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"chat request failed ({exc.code}): {detail}") from exc


# ---------------------------------------------------------------------------
# Nodes: each is a pure function of state that appends one message
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a mail assistant. Use the provided tools to read the user's mail and "
    "answer questions about it. Emit one tool call per turn in the Hermes tool-call "
    "format. When you already have the answer, reply in plain text with no tool call."
)


def _build_request_messages(state: MailState) -> list[dict[str, Any]]:
    """Prepend the system message to the recorded transcript for the next call."""

    return [{"role": "system", "content": _SYSTEM}, *state.messages]


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


def planner_node(state: MailState, *, client: OpenAIClient) -> MailState:
    """Kick off the think-act-observe loop: seed the transcript with the user query."""

    if state.messages:
        return state  # only seeds on the first transition
    state.messages.append({"role": "user", "content": state.user_query})
    return state


def router_node(state: MailState, *, client: OpenAIClient) -> MailState:
    """Ask the student for the next action given the transcript so far.

    This is the report's "decide which tool to call / whether to answer" step.
    One assistant message is appended (tool-call block or prose answer).
    """

    if state.step >= MAX_STEPS:
        state.done = False
        return state
    response = client.chat(_build_request_messages(state), tools=TOOLS, max_tokens=256)
    state.messages.append(_assistant_message_from_response(response))
    state.step += 1
    return state


def tool_execution_node(state: MailState, *, client: OpenAIClient, env: MailEnv) -> MailState:
    """Execute any tool call in the latest assistant message; append tool results.

    If the latest assistant turn has no tool call, this is a no-op -- the Verifier
    then decides whether the prose answer completes the task. ``env`` is the
    per-task mailbox bound in at graph build time, not a state field.
    """

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


def verifier_node(state: MailState, *, client: OpenAIClient) -> MailState:
    """Classify the run: last assistant turn with prose (no tool call) means done.

    The report's Verifier decides "task complete" vs "keep calling." It sets the
    ``done`` flag only; routing is decided by the graph's conditional edges
    (``build_graph``). When the loop lands on a prose answer within budget, that prose
    IS the summary and no extra turn is needed -- the Summarizer only acts as a
    forced backstop when the budget is spent before any prose answer appeared.
    """

    del client  # the Verifier never calls the model; it only reads the transcript
    last = state.messages[-1] if state.messages else None
    over_budget = state.step >= MAX_STEPS
    state.done = bool(
        last
        and last.get("role") == "assistant"
        and not last.get("tool_calls")
        and not over_budget
    )
    return state


def summarizer_node(state: MailState, *, client: OpenAIClient) -> MailState:
    """Forced backstop: ask for one prose answer when the loop ended without one.

    FlowMind's Execute-Summarize split: Execute (Planner/Router/ToolExecution above)
    produced the trajectory; this Summarize turn produces the prose final answer the
    eval compares against -- a single forced response, tools disabled, so the model
    answers from gathered observations instead of looping on another tool call. This
    node is only reached when the loop was aborted with no prose answer in view.
    """

    response = client.chat(_build_request_messages(state), tools=None, max_tokens=256)
    state.messages.append(_assistant_message_from_response(response))
    state.done = not state.messages[-1].get("tool_calls")
    return state


# ---------------------------------------------------------------------------
# Graph construction (langgraph imported lazily)
# ---------------------------------------------------------------------------


def build_graph(client: OpenAIClient, env: MailEnv) -> Any:
    """Compile the Planner->Router->ToolExecution->Verifier->Summarizer graph.

    ``env`` is the per-task mailbox; it is a build-time closure value, NOT a state
    field, because the ``MemorySaver`` checkpointer serializes state and ``MailEnv``
    is not msgpack-serializable. Build a fresh graph per task to swap env.

    ``langgraph`` is imported here so the module imports without it installed. The
    topology matches the report's section 2.1: each Router turn drives one think
    step; ToolExecution runs any tools called; Verifier classifies done/not-done;
    Summarizer is a forced-backstop final answer only when the budget is spent
    before a prose answer appeared. Conditional edges implement the loop:

      router -> (tool call?) tool_execution -> router                   # keep acting
              -> (prose/done) verifier -> END                           # that prose is the summary
      (budget spent, no prose)           verifier -> summarizer -> END   # forced backstop answer

    A ``MemorySaver`` checkpointer records every transition, so the checkpoint
    history is the executed trajectory (report section 3.1).
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
    graph.add_node("planner", lambda s: planner_node(s, client=client))
    graph.add_node("router", lambda s: router_node(s, client=client))
    graph.add_node("tool_execution", lambda s: tool_execution_node(s, client=client, env=env))
    graph.add_node("verifier", lambda s: verifier_node(s, client=client))
    graph.add_node("summarizer", lambda s: summarizer_node(s, client=client))

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
]
