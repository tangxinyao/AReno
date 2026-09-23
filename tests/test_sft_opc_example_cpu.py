from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sft" / "opc"


@pytest.fixture(autouse=True)
def _clean_opc_env(monkeypatch):
    """Default to no compression knobs and B mode, so tests are independent of
    whatever ARENO_OPC_* vars happen to be exported (except ARENO_OPC_TOKENIZER,
    the explicit opt-in that gates the tokenizer end-to-end test)."""

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


def _load_loader():
    path = EXAMPLE_DIR / "dataset_loader.py"
    spec = importlib.util.spec_from_file_location("sft_opc_dataset_loader_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_session(tmp_path, *, reward: float, name: str) -> Path:
    trial = tmp_path / name
    agent = trial / "agent"
    agent.mkdir(parents=True)
    session = {
        "system_prompt": "You are Hermes Agent...",
        "messages": [
            {"role": "user", "content": "盘点 2026 年收入。"},
            {
                "role": "assistant",
                "content": "先看看机器上有什么。",
                "reasoning_content": "Explore first.",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": '{"command": "ls /app"}'}}
                ],
            },
            {"role": "tool", "tool_name": "terminal", "content": '{"output": "orders.csv"}'},
            {"role": "assistant", "content": "盘完了，共 963,000 元。"},
        ],
    }
    (agent / "hermes-session.jsonl").write_text(json.dumps(session, ensure_ascii=False))
    result = {"verifier_result": {"rewards": {"reward": reward}}}
    (trial / "result.json").write_text(json.dumps(result))
    return trial


def test_opc_loader_filters_by_reward_and_expands_per_turn(tmp_path, monkeypatch):
    monkeypatch.delenv("ARENO_OPC_DROP_REASONING", raising=False)
    _write_session(tmp_path, reward=1.0, name="passed-trial")
    _write_session(tmp_path, reward=0.0, name="failed-trial")

    loader = _load_loader()
    records = loader.load_training_dataset(str(tmp_path), default_loader=lambda _: [])

    # Failed trial dropped; two assistant turns expand into two rows.
    assert [r["source_trial"] for r in records] == ["passed-trial", "passed-trial"]
    assert records[0]["prompt"].startswith("detailed thinking on<|role_end|>")
    assert records[0]["prompt"].endswith("<role>ASSISTANT</role>\n thinking")
    # Target keeps reasoning, then " response", content, then Ling tool format.
    assert records[0]["response"].startswith("Explore first.")
    assert " response先看看机器上有什么。" in records[0]["response"]
    assert "<tool_call>terminal\n<arg_key>command</arg_key>\n<arg_value>ls /app</arg_value>\n</tool_call>" in records[0][
        "response"
    ]
    # The second row's prompt carries the tool result as a <tool_response> block.
    assert "<role>OBSERVATION</role>" in records[1]["prompt"]
    assert "<tool_response>" in records[1]["prompt"]
    assert records[1]["response"] == " response盘完了，共 963,000 元。"  # no reasoning -> just " response"


def test_opc_loader_can_drop_reasoning_and_accept_single_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENO_OPC_DROP_REASONING", "1")
    trial = _write_session(tmp_path, reward=1.0, name="passed-trial")
    session_file = trial / "agent" / "hermes-session.jsonl"

    loader = _load_loader()
    records = loader.load_training_dataset(str(session_file), default_loader=lambda _: [])

    assert records[0]["response"].startswith(" response先看看机器上有什么。")
    assert "Explore first." not in records[0]["response"]
    assert "<tool_call>" in records[0]["response"]


def _write_session_with_turns(tmp_path, *, n_turns: int, name: str) -> Path:
    trial = tmp_path / name
    agent = trial / "agent"
    agent.mkdir(parents=True)
    messages = [{"role": "user", "content": "任务 prompt。"}]
    for i in range(n_turns):
        messages.append(
            {
                "role": "assistant",
                "content": f"第{i}步动作。",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": json.dumps({"command": f"step {i}"})}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_name": "terminal", "content": json.dumps({"output": f"step {i} result"})})
    messages.append({"role": "assistant", "content": "完成。"})
    (agent / "hermes-session.jsonl").write_text(json.dumps({"messages": messages}, ensure_ascii=False))
    result = {"verifier_result": {"rewards": {"reward": 1.0}}}
    (trial / "result.json").write_text(json.dumps(result))
    return trial


def test_opc_loader_sliding_window_keeps_task_prompt(tmp_path, monkeypatch):
    _write_session_with_turns(tmp_path, n_turns=8, name="win")
    monkeypatch.setenv("ARENO_OPC_MAX_HISTORY_MESSAGES", "2")

    loader = _load_loader()
    records = loader.load_training_dataset(str(tmp_path), default_loader=lambda _: [])

    assert len(records) == 9  # 8 tool turns + final answer
    last_prompt = records[-1]["prompt"]
    assert "任务 prompt。" in last_prompt   # sliding window must keep the task
    assert "第0步动作。" not in last_prompt  # early turns slid out of context
    assert "step 7 result" in last_prompt   # recent observation kept


def test_opc_loader_tool_char_cap_only_touches_context(tmp_path, monkeypatch):
    trial = tmp_path / "big"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    big_output = json.dumps({"output": "x" * 5000})
    messages = [
        {"role": "user", "content": "查一下。"},
        {
            "role": "assistant",
            "content": "查看大文件。",
            "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "cat big"}'}}],
        },
        {"role": "tool", "tool_name": "terminal", "content": big_output},
        {"role": "assistant", "content": "看完了。"},
    ]
    (agent / "hermes-session.jsonl").write_text(json.dumps({"messages": messages}, ensure_ascii=False))
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))

    monkeypatch.setenv("ARENO_OPC_MAX_TOOL_CHARS", "100")
    loader = _load_loader()
    records = loader.load_training_dataset(str(tmp_path), default_loader=lambda _: [])

    assert "…<truncated " in records[1]["prompt"]  # big tool result capped in context
    assert len(records[1]["prompt"]) < 1200         # 5000-char output no longer inlined
    # target stays verbatim
    assert (
        records[0]["response"]
        == " response查看大文件。\n<tool_call>terminal\n<arg_key>command</arg_key>\n<arg_value>cat big</arg_value>\n</tool_call>"
    )
    assert records[1]["response"] == " response看完了。"


def test_opc_loader_phase_split_tags_genre_and_index(tmp_path, monkeypatch):
    trial = tmp_path / "phased"
    agent = trial / "agent"
    agent.mkdir(parents=True)

    def turn(content, tool):
        return {"role": "assistant", "content": content, "tool_calls": [{"function": {"name": tool, "arguments": "{}"}}]}

    messages = [
        {"role": "user", "content": "任务 prompt。"},
        turn("找文件。", "search_files"),                       # explore
        turn("读文件。", "read_file"),                          # read
        turn("写脚本。", "write_file"),                         # implement
        turn("跑脚本。", "execute_code"),                       # run
        {"role": "assistant", "content": "交付报告。"},         # report (no tool)
    ]
    (agent / "hermes-session.jsonl").write_text(json.dumps({"messages": messages}, ensure_ascii=False))
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))

    monkeypatch.setenv("ARENO_OPC_PHASES", "1")
    loader = _load_loader()
    records = loader.load_training_dataset(str(tmp_path), default_loader=lambda _: [])

    assert [r["phase_genre"] for r in records] == ["explore", "read", "implement", "run", "report"]
    assert [r["phase_index"] for r in records] == [1, 2, 3, 4, 5]  # contiguous runs numbered from 1


def _write_ling_session(tmp_path) -> None:
    trial = tmp_path / "ling"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    messages = [
        {"role": "user", "content": "盘点 2026 年收入。"},
        {
            "role": "assistant",
            "content": "先看看机器。",
            "reasoning_content": "探索一下环境。",
            "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "ls /app"}'}}
            ],
        },
        {"role": "tool", "tool_name": "terminal", "content": '{"output": "orders.csv"}'},
        {"role": "assistant", "content": "盘完了，共 963,000 元。"},
    ]
    (agent / "hermes-session.jsonl").write_text(json.dumps({"messages": messages}, ensure_ascii=False))
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))


def test_opc_loader_c_chunk_ranges():
    loader = _load_loader()
    assert loader._chunk_ranges(300, [0] * 100 + [1] * 100 + [2] * 100, 150) == [(0, 100), (100, 200), (200, 300)]
    assert loader._chunk_ranges(400, [0] * 400, 150) == [(0, 150), (150, 300), (300, 400)]  # hard-split one huge part
    assert loader._chunk_ranges(5, [0, 0, 0, 0, 0], 32768) == [(0, 5)]


def test_opc_loader_c_ling_parts_target_flags():
    loader = _load_loader()
    messages = [
        {"role": "user", "content": "任务。"},
        {
            "role": "assistant",
            "content": "查。",
            "reasoning_content": "想。",
            "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "ls"}'}}],
        },
        {"role": "tool", "tool_name": "terminal", "content": "out"},
        {"role": "assistant", "content": "完成。"},
    ]
    parts = loader._build_ling_parts(messages)

    def flag_of(substring):
        for text, target in parts:
            if substring in text:
                return target
        raise AssertionError(f"no part contains {substring!r}")

    assert flag_of("detailed thinking on") is False          # header = context
    assert flag_of("<role>HUMAN</role>") is False            # user = context
    assert flag_of("<role>ASSISTANT</role>\n thinking") is False  # gen prefix = context
    assert flag_of(" response查。") is True                  # assistant turn = target
    assert flag_of("<tool_call>terminal") is True            # tool call inside target
    assert flag_of("<role>OBSERVATION</role>") is False      # tool result = context
    assert flag_of("<|role_end|>") is not None
    assert flag_of("完成。") is True                          # final answer = target


def test_opc_loader_c_mode_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    tokenizer_dir = os.environ.get("ARENO_OPC_TOKENIZER")
    if not tokenizer_dir:
        pytest.skip("set ARENO_OPC_TOKENIZER=<dir> to exercise the real tokenizer")
    _write_ling_session(tmp_path)
    monkeypatch.setenv("ARENO_OPC_C_MODE", "1")
    monkeypatch.setenv("ARENO_OPC_TOKENIZER", tokenizer_dir)
    monkeypatch.setenv("ARENO_OPC_MAX_SEQ_TOKENS", "50000")

    loader = _load_loader()
    records = loader.load_training_dataset(str(tmp_path), default_loader=lambda _: [])

    assert records
    for record in records:
        assert len(record["tokens"]) == len(record["prompt_mask"]) == len(record["loss_mask"])
        assert any(record["loss_mask"]) and not all(record["loss_mask"])  # context + target both present