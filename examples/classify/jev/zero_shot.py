"""Zero-shot baseline: a plain causal LM answers Jev questions through its LM head.

No training, no score head. Each question becomes one chat prompt that lists the
state, the question and the candidates, and asks for a one-token answer:

    choice -> option letter (A, B, ...)    noul -> "yes" / "no"    score -> level digit

The prediction is the LM's next-token distribution renormalized over the
candidate tokens (one packed forward per batch of questions, `NextTokenScorer`).
`label_mass` is the probability the LM put on the candidate tokens at all
(format adherence). `--answer-prefix auto` picks, on the first 64 questions,
the prefix that maximizes label_mass; that uses no gold labels.

    python examples/classify/jev/zero_shot.py --model <Ling-3.0-tiny dir> \
        --records ~/data/jev-records/typed-decisions --split test --output zs.json --dump zs.dump.jsonl

Metrics are the same as evaluate.py (acc / ce / kl / tv / brier / conf / overconf / ece).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import candidate_ids  # noqa: E402
from evaluate import question_metrics, report  # noqa: E402

PREFIX_CHOICES = ("", "\n", "Answer: ", "\nAnswer: ")


def log_sum_exp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values))


def load_questions(records: str, split: str) -> list[dict]:
    rows = []
    with (Path(records).expanduser() / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for qid, question in record["request"]["questions"].items():
                target = record["targets"].get(qid)
                if target is None:
                    continue
                ids = candidate_ids(question)
                rows.append(
                    {
                        "state": record["request"]["state"],
                        "question": question,
                        "ids": ids,
                        "target": [float(target[i]) for i in ids],
                        "type": question["type"],
                        "group": record.get("source_group", ""),
                    }
                )
    return rows


def build_prompt(row: dict) -> tuple[str, list[str]] | None:
    """User message and the one-token answer label for each candidate (candidate order)."""

    question, kind, ids = row["question"], row["type"], row["ids"]
    lines = [
        "Read the state and answer the question.",
        "",
        "State:",
        row["state"],
        "",
        f"Question: {question['instructions']}",
    ]
    if kind == "choice":
        if len(ids) > 26:
            return None
        labels = list(string.ascii_uppercase[: len(ids)])
        lines.append("Options:")
        for label, key in zip(labels, ids, strict=True):
            description = str(question["criteria"][key])
            same = description.strip() == key.strip()
            lines.append(f"{label}. {key}" if same else f"{label}. {key}: {description}")
        lines.append("Reply with the letter of the best option only.")
        return "\n".join(lines), labels
    if kind == "noul":
        criteria = question.get("criteria") or {}
        if criteria.get("true"):
            lines.append(f"yes means: {criteria['true']}")
        if criteria.get("false"):
            lines.append(f"no means: {criteria['false']}")
        lines.append("Reply with yes or no only.")
        return "\n".join(lines), ["no", "yes"]  # candidate order is [false, true]
    if len(ids) > 10:
        return None
    lines.append("Levels:")
    for index, description in enumerate(question["criteria"]):
        lines.append(f"{index}. {description}")
    lines.append("Reply with the level number only.")
    return "\n".join(lines), [str(index) for index in range(len(ids))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="causal LM checkpoint directory (e.g. Ling-3.0-tiny)")
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="seeded random subset (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--answer-prefix", default="auto", help='"auto" or a literal string appended after the chat prompt'
    )
    parser.add_argument("--enable-thinking", action="store_true", help="keep the chat template's thinking mode on")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--attn-backend", choices=["flash", "native"], default="flash")
    parser.add_argument("--batch", type=int, default=32, help="questions per scoring call")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dump", default=None, help="per-question candidate log-probs + gold (jsonl) for calibrate.py")
    args = parser.parse_args()

    import torch

    from areno.api.tokenizer import (
        apply_chat_template_with_options,
        configure_chat_template_enable_thinking,
        normalize_token_ids,
    )
    from areno.experimental.classify.scorer import NextTokenScorer

    rows = load_questions(args.records, args.split)
    if args.limit and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)
    scorer = NextTokenScorer(args.model, attn_backend=args.attn_backend)
    tokenizer = scorer.tokenizer
    configure_chat_template_enable_thinking(tokenizer, True if args.enable_thinking else False)

    def label_token(label: str) -> int:
        ids = normalize_token_ids(tokenizer.encode(label, add_special_tokens=False))
        return int(ids[0])

    prepared, skipped = [], 0
    for row in rows:
        built = build_prompt(row)
        if built is None:
            skipped += 1
            continue
        text, labels = built
        prompt = normalize_token_ids(
            apply_chat_template_with_options(
                tokenizer, [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
            )
        )
        tokens = [label_token(label) for label in labels]
        if len(prompt) > args.max_prompt_tokens or len(set(tokens)) != len(tokens):
            skipped += 1
            continue
        prepared.append((row, prompt, tokens))

    def encode_prefix(prefix: str) -> list[int]:
        return normalize_token_ids(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else []

    def score(items, prefix_ids):
        logprobs = scorer.next_token_logprobs([p + prefix_ids for _, p, _ in items], [t for _, _, t in items])
        return [lp.float().tolist() for lp in logprobs]

    prefix = args.answer_prefix
    if prefix == "auto":
        probe = prepared[:64]
        masses = {}
        for choice in PREFIX_CHOICES:
            values = score(probe, encode_prefix(choice))
            masses[choice] = sum(math.exp(log_sum_exp(v)) for v in values) / max(len(values), 1)
        prefix = max(masses, key=masses.get)
        print("answer prefix label_mass on 64 probe questions:", {repr(k): round(v, 4) for k, v in masses.items()})
    print(f"answer prefix: {prefix!r}; scoring {len(prepared)} questions (skipped {skipped})", flush=True)
    prefix_ids = encode_prefix(prefix)

    results, dumped, masses_by_type = [], [], defaultdict(list)
    for start in range(0, len(prepared), args.batch):
        batch = prepared[start : start + args.batch]
        for (row, _, _), values in zip(batch, score(batch, prefix_ids), strict=True):
            probs = torch.softmax(torch.tensor(values), dim=0).tolist()
            metrics = question_metrics(probs, row["target"])
            metrics["label_mass"] = math.exp(log_sum_exp(values))
            results.append({"type": row["type"], "group": row["group"], **metrics})
            masses_by_type[row["type"]].append(metrics["label_mass"])
            if args.dump:
                dumped.append({"type": row["type"], "group": row["group"], "logits": values, "target": row["target"]})
        if (start // args.batch) % 20 == 0:
            print(f"  {start + len(batch)}/{len(prepared)}", flush=True)

    meta = {"model": args.model, "records": args.records, "split": args.split, "mode": "zero-shot LM head"}
    summary = report(results, {**meta, "answer_prefix": prefix, "skipped": skipped})
    summary["label_mass"] = {kind: sum(v) / len(v) for kind, v in masses_by_type.items()}
    print("mean label_mass by type:", {k: round(v, 3) for k, v in summary["label_mass"].items()})
    if args.output:
        Path(args.output).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.dump:
        with Path(args.dump).expanduser().open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"meta": meta}) + "\n")
            for item in dumped:
                handle.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
