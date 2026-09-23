from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sft" / "opc"
EOS = "<|role_end|>"


@pytest.fixture(autouse=True)
def _clean_opc_env(monkeypatch):
    """Default to no compression knobs and B mode, so tests are independent of
    whatever ARENO_OPC_* vars happen to be exported (except ARENO_OPC_TOKENIZER,
    the explicit opt-in that gates the real-tokenizer tests)."""

    for var in (
        "ARENO_OPC_DROP_REASONING",
        "ARENO_OPC_INCLUDE_SYSTEM_PROMPT",
        "ARENO_OPC_MAX_HISTORY_MESSAGES",
        "ARENO_OPC_MAX_TOOL_CHARS",
        "ARENO_OPC_PHASES",
        "ARENO_OPC_C_MODE",
        "ARENO_OPC_MAX_SEQ_TOKENS",
    ):
        monkeypatch.delenv(var, raising=False)


class FakeLingTokenizer:
    """Char-level tokenizer with a small Python chat template shaped like Ling's.

    It is not the real jinja template; it only has the properties the loader
    relies on (prefix-stable turns, merged OBSERVATION blocks, EOS-closed
    turns), so CPU tests exercise span/mask logic rather than exact markup.
    """

    eos_token = EOS
    eos_token_id = 0
    chat_template = "fake"

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        out = []
        if not messages or messages[0]["role"] != "system":
            out.append(f"<role>SYSTEM</role>detailed thinking on{EOS}")
        i = 0
        while i < len(messages):
            message = messages[i]
            role = message["role"]
            if role == "tool":
                blocks = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    blocks.append(f"<tool_response>{messages[i]['content']}</tool_response>")
                    i += 1
                out.append("<role>OBSERVATION</role>" + "".join(blocks) + EOS)
                continue
            if role == "system":
                out.append(f"<role>SYSTEM</role>{message['content']}{EOS}")
            elif role == "user":
                out.append(f"<role>HUMAN</role>{message['content']}{EOS}")
            elif role == "assistant":
                body = f"<think>{message.get('reasoning_content', '')}</think>{message['content']}"
                for call in message.get("tool_calls", []):
                    args = "".join(
                        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
                        for key, value in call["function"]["arguments"].items()
                    )
                    body += f"<tool_call>{call['function']['name']}{args}</tool_call>"
                out.append(f"<role>ASSISTANT</role>\n{body}{EOS}")
            i += 1
        if add_generation_prompt:
            out.append("<role>ASSISTANT</role>\n<think>")
        return "".join(out)

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def _load_loader(tokenizer=None):
    path = EXAMPLE_DIR / "dataset_loader.py"
    spec = importlib.util.spec_from_file_location("sft_opc_dataset_loader_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if tokenizer is not None:
        module._load_tokenizer = lambda: tokenizer
    return module


def _load(path, tokenizer=None):
    loader = _load_loader(tokenizer or FakeLingTokenizer())
    return loader.load_training_dataset(str(path), default_loader=lambda _: [])


def _write_trial(root: Path, name: str, messages: list[dict], *, reward=1.0, result=True, **session) -> Path:
    trial = root / name
    agent = trial / "agent"
    agent.mkdir(parents=True)
    (agent / "hermes-session.jsonl").write_text(json.dumps({"messages": messages, **session}, ensure_ascii=False))
    if result:
        (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": reward}}}))
    return trial


def _basic_messages() -> list[dict]:
    return [
        {"role": "user", "content": "盘点 2026 年收入。"},
        {
            "role": "assistant",
            "content": "先看看机器上有什么。",
            "reasoning_content": "Explore first.",
            "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "ls /app"}'}}],
        },
        {"role": "tool", "tool_name": "terminal", "content": '{"output": "orders.csv"}'},
        {"role": "assistant", "content": "盘完了，共 963,000 元。"},
    ]


def _turn_messages(n_turns: int) -> list[dict]:
    messages = [{"role": "user", "content": "任务 prompt。"}]
    for i in range(n_turns):
        messages.append(
            {
                "role": "assistant",
                "content": f"第{i}步动作。",
                "tool_calls": [{"function": {"name": "terminal", "arguments": json.dumps({"command": f"step {i}"})}}],
            }
        )
        messages.append({"role": "tool", "tool_name": "terminal", "content": json.dumps({"output": f"step {i} result"})})
    messages.append({"role": "assistant", "content": "完成。"})
    return messages


def _decode(tokens, mask=None):
    return "".join(chr(t) for i, t in enumerate(tokens) if mask is None or mask[i])


def test_opc_loader_filters_by_reward_and_expands_per_turn(tmp_path):
    _write_trial(tmp_path, "passed-trial", _basic_messages(), reward=1.0)
    _write_trial(tmp_path, "failed-trial", _basic_messages(), reward=0.0)

    records = _load(tmp_path)

    # Failed trial dropped; two assistant turns expand into two rows.
    assert [r["source_trial"] for r in records] == ["passed-trial", "passed-trial"]
    assert records[0]["prompt"].endswith("<role>ASSISTANT</role>\n<think>")
    assert records[0]["prompt"].count("<role>HUMAN</role>") == 1
    # Target = the template's rendering of the turn after the generation prompt, minus EOS.
    assert records[0]["response"] == (
        "Explore first.</think>先看看机器上有什么。"
        "<tool_call>terminal<arg_key>command</arg_key><arg_value>ls /app</arg_value></tool_call>"
    )
    assert "<role>OBSERVATION</role><tool_response>" in records[1]["prompt"]
    assert records[1]["response"] == "</think>盘完了，共 963,000 元。"


def test_opc_loader_rows_concatenate_to_template_rendering(tmp_path):
    tokenizer = FakeLingTokenizer()
    _write_trial(tmp_path, "t", _basic_messages())
    loader = _load_loader(tokenizer)
    records = loader.load_training_dataset(str(tmp_path), default_loader=lambda _: [])

    messages = loader._chat_messages({"messages": _basic_messages()}, include_system_prompt=False, drop_reasoning=False)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    assert records[-1]["prompt"] + records[-1]["response"] + EOS == full


def test_opc_loader_drops_result_json_without_reward(tmp_path):
    trial = _write_trial(tmp_path, "crashed", _basic_messages(), result=False)
    (trial / "result.json").write_text(json.dumps({"verifier_result": None}))
    _write_trial(tmp_path, "ok", _basic_messages())

    assert {r["source_trial"] for r in _load(tmp_path)} == {"ok"}


def test_opc_loader_single_trial_dir_still_filters_reward(tmp_path):
    trial = _write_trial(tmp_path, "failed", _basic_messages(), reward=0.0)

    assert _load(trial) == []  # was kept via the looser layout fallback before


def test_opc_loader_can_drop_reasoning_and_accept_single_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENO_OPC_DROP_REASONING", "1")
    trial = _write_trial(tmp_path, "passed-trial", _basic_messages())

    records = _load(trial / "agent" / "hermes-session.jsonl")

    assert records[0]["response"].startswith("</think>先看看机器上有什么。")
    assert "Explore first." not in records[0]["response"]
    assert records[0]["source_trial"] == "passed-trial"


def test_opc_loader_skips_inactive_and_compacted_messages(tmp_path):
    messages = _basic_messages()
    messages.insert(1, {"role": "assistant", "content": "被回滚的一步。", "active": 0})
    messages.insert(2, {"role": "user", "content": "被压缩的消息。", "compacted": 1})
    _write_trial(tmp_path, "t", messages)

    records = _load(tmp_path)

    assert len(records) == 2
    assert all("被回滚" not in r["prompt"] + r["response"] and "被压缩" not in r["prompt"] for r in records)


def test_opc_loader_sliding_window_keeps_task_prompt(tmp_path, monkeypatch):
    _write_trial(tmp_path, "win", _turn_messages(8), system_prompt="You are Hermes.")
    monkeypatch.setenv("ARENO_OPC_MAX_HISTORY_MESSAGES", "2")
    monkeypatch.setenv("ARENO_OPC_INCLUDE_SYSTEM_PROMPT", "1")

    records = _load(tmp_path)

    assert len(records) == 9  # 8 tool turns + final answer
    last_prompt = records[-1]["prompt"]
    assert "You are Hermes." in last_prompt
    assert "任务 prompt。" in last_prompt  # window keeps the task even with a system prompt in front
    assert "第0步动作。" not in last_prompt  # early turns slid out of context
    assert "step 7 result" in last_prompt  # recent observation kept


def test_opc_loader_tool_char_cap_only_touches_context(tmp_path, monkeypatch):
    messages = [
        {"role": "user", "content": "查一下。"},
        {
            "role": "assistant",
            "content": "查看大文件。",
            "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"content": "y" * 500})}}],
        },
        {"role": "tool", "tool_name": "write_file", "content": json.dumps({"output": "x" * 5000})},
        {"role": "assistant", "content": "看完了。"},
    ]
    _write_trial(tmp_path, "big", messages)
    monkeypatch.setenv("ARENO_OPC_MAX_TOOL_CHARS", "100")

    records = _load(tmp_path)

    assert "…<truncated " in records[1]["prompt"]  # tool result and history tool args capped
    assert "y" * 500 not in records[1]["prompt"]
    assert len(records[1]["prompt"]) < 1200
    assert "y" * 500 in records[0]["response"]  # target stays verbatim
    assert "truncated" not in records[0]["response"]


def test_opc_loader_phase_split_tags_genre_and_index(tmp_path, monkeypatch):
    def turn(content, tool, flat=False):
        call = {"name": tool, "arguments": "{}"} if flat else {"function": {"name": tool, "arguments": "{}"}}
        return {"role": "assistant", "content": content, "tool_calls": [call]}

    messages = [
        {"role": "user", "content": "任务 prompt。"},
        turn("找文件。", "search_files"),  # explore
        turn("读文件。", "read_file", flat=True),  # read (flat tool-call format)
        turn("写脚本。", "write_file"),  # implement
        turn("跑脚本。", "execute_code"),  # run
        {"role": "assistant", "content": "交付报告。"},  # report (no tool)
    ]
    _write_trial(tmp_path, "phased", messages)
    monkeypatch.setenv("ARENO_OPC_PHASES", "1")

    records = _load(tmp_path)

    assert [r["phase_genre"] for r in records] == ["explore", "read", "implement", "run", "report"]
    assert [r["phase_index"] for r in records] == [1, 2, 3, 4, 5]  # contiguous runs numbered from 1


def test_split_phases_output_loads_back_through_loader(tmp_path):
    _write_trial(tmp_path / "data", "t", _basic_messages())
    loader = _load_loader(FakeLingTokenizer())
    records = loader.load_training_dataset(str(tmp_path / "data"), default_loader=lambda _: [])
    row_file = tmp_path / "run.jsonl"
    row_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    assert _load(row_file) == records


def test_opc_loader_rejects_bad_int_env(tmp_path, monkeypatch):
    _write_trial(tmp_path, "t", _basic_messages())
    monkeypatch.setenv("ARENO_OPC_MAX_SEQ_TOKENS", "32k")

    with pytest.raises(ValueError, match="ARENO_OPC_MAX_SEQ_TOKENS"):
        _load(tmp_path)


def test_opc_loader_rejects_prefix_unstable_template(tmp_path, monkeypatch):
    class DropsPastReasoning(FakeLingTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
            messages = [
                {k: v for k, v in m.items() if k != "reasoning_content"} if i != last else m
                for i, m in enumerate(messages)
            ]
            return super().apply_chat_template(messages, **kwargs)

    messages = _basic_messages()
    messages[3]["reasoning_content"] = "收尾。"
    _write_trial(tmp_path, "t", messages)
    monkeypatch.setenv("ARENO_OPC_C_MODE", "1")

    with pytest.raises(ValueError, match="prefix-stable"):
        _load(tmp_path, DropsPastReasoning())


def test_opc_loader_c_chunk_ranges():
    loader = _load_loader()
    assert loader._chunk_ranges(300, [0] * 100 + [1] * 100 + [2] * 100, 150) == [(0, 100), (100, 200), (200, 300)]
    assert loader._chunk_ranges(400, [0] * 400, 150) == [(0, 150), (150, 300), (300, 400)]  # hard-split one huge part
    assert loader._chunk_ranges(5, [0, 0, 0, 0, 0], 32768) == [(0, 5)]


def test_opc_loader_c_mode_masks_exactly_assistant_spans(tmp_path, monkeypatch):
    _write_trial(tmp_path, "t", _basic_messages())
    monkeypatch.setenv("ARENO_OPC_C_MODE", "1")

    records = _load(tmp_path)

    assert len(records) == 1
    row = records[0]
    assert row["prompt_mask"] == [not t for t in row["loss_mask"]]
    target = _decode(row["tokens"], row["loss_mask"])
    assert target == (
        "Explore first.</think>先看看机器上有什么。"
        "<tool_call>terminal<arg_key>command</arg_key><arg_value>ls /app</arg_value></tool_call>"
        f"{EOS}</think>盘完了，共 963,000 元。{EOS}"
    )
    context = _decode(row["tokens"], row["prompt_mask"])
    assert "<role>HUMAN</role>" in context and "orders.csv" in context


def test_opc_loader_c_mode_chunks_repeat_task_and_cut_at_turns(tmp_path, monkeypatch):
    _write_trial(tmp_path, "t", _turn_messages(8))
    monkeypatch.setenv("ARENO_OPC_C_MODE", "1")
    monkeypatch.setenv("ARENO_OPC_MAX_SEQ_TOKENS", "400")

    records = _load(tmp_path)

    assert len(records) > 1
    preamble = f"<role>SYSTEM</role>detailed thinking on{EOS}<role>HUMAN</role>任务 prompt。{EOS}"
    for row in records:
        assert len(row["tokens"]) <= 400
        text = _decode(row["tokens"])
        assert text.startswith(preamble)  # every chunk keeps the task
        body = text[len(preamble) :]
        # Cuts land exactly on turn boundaries (no token of the next message left behind).
        assert body.startswith("<role>ASSISTANT</role>") or body.startswith("<role>OBSERVATION</role>")
        assert not any(row["loss_mask"][: len(preamble)])


def test_opc_loader_c_mode_keeps_tool_call_targets_verbatim(tmp_path, monkeypatch):
    messages = _basic_messages()
    messages[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": "z" * 300})
    messages[2]["content"] = "o" * 300
    _write_trial(tmp_path, "t", messages)
    monkeypatch.setenv("ARENO_OPC_C_MODE", "1")
    monkeypatch.setenv("ARENO_OPC_MAX_TOOL_CHARS", "50")

    row = _load(tmp_path)[0]

    assert "z" * 300 in _decode(row["tokens"], row["loss_mask"])  # target untouched
    assert "o" * 300 not in _decode(row["tokens"])  # observation capped


# --- Real Ling tokenizer (opt-in: ARENO_OPC_TOKENIZER=<dir>) ----------------------


def _real_tokenizer_dir() -> str:
    pytest.importorskip("transformers")
    tokenizer_dir = os.environ.get("ARENO_OPC_TOKENIZER")
    if not tokenizer_dir:
        pytest.skip("set ARENO_OPC_TOKENIZER=<dir> to exercise the real tokenizer")
    return tokenizer_dir


def test_opc_loader_b_mode_real_tokenizer_matches_template(tmp_path):
    _real_tokenizer_dir()
    from areno.api.data_utils import prompt_response_to_tokens_and_mask

    _write_trial(tmp_path, "ling", _basic_messages())
    loader = _load_loader()
    tokenizer = loader._load_tokenizer()
    records = loader.load_training_dataset(str(tmp_path), default_loader=lambda _: [])

    messages = loader._chat_messages({"messages": _basic_messages()}, include_system_prompt=False, drop_reasoning=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    last = records[-1]
    assert full.rstrip().startswith(last["prompt"] + last["response"])
    assert full.rstrip().endswith(tokenizer.eos_token)

    # Through the trainer's own encoding path: the template is applied once, EOS is the stop token.
    tokens, _ = prompt_response_to_tokens_and_mask(
        last["prompt"], last["response"], tokenizer, tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(tokens)
    assert decoded.count("<role>HUMAN</role>") == 1
    assert tokens[-1] == tokenizer.eos_token_id


def test_opc_loader_c_mode_end_to_end(tmp_path, monkeypatch):
    _real_tokenizer_dir()
    _write_trial(tmp_path, "ling", _basic_messages())
    monkeypatch.setenv("ARENO_OPC_C_MODE", "1")
    monkeypatch.setenv("ARENO_OPC_MAX_SEQ_TOKENS", "50000")

    loader = _load_loader()
    records = loader.load_training_dataset(str(tmp_path), default_loader=lambda _: [])

    assert records
    tokenizer = loader._load_tokenizer()
    for record in records:
        assert len(record["tokens"]) == len(record["prompt_mask"]) == len(record["loss_mask"])
        assert any(record["loss_mask"]) and not all(record["loss_mask"])  # context + target both present
        target = tokenizer.decode([t for t, m in zip(record["tokens"], record["loss_mask"]) if m])
        assert "盘完了" in target and "orders.csv" not in target
