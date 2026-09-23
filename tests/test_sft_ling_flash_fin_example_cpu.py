from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "sft" / "ling_flash_fin" / "data"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA_DIR / name).read_text(encoding="utf-8").splitlines()]


def test_train_rows_match_sft_prompt_response_schema():
    rows = _rows("train.jsonl")

    assert rows
    assert all(row["prompt"].strip() and row["response"].strip() for row in rows)
    assert {row["lang"] for row in rows} == {"zh", "en"}


def test_eval_questions_are_held_out_from_train():
    train_prompts = {row["prompt"] for row in _rows("train.jsonl")}
    evals = _rows("eval.jsonl")

    assert {row["lang"] for row in evals} == {"zh", "en"}
    assert all(row["reference"].strip() for row in evals)
    assert not train_prompts & {row["prompt"] for row in evals}
