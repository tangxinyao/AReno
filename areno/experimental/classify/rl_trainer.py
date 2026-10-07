"""PPO trainer for the sequence-scoring (classification) head.

Dataset rows describe one decision each:

- `{"prompt": str, "candidates": [str, ...], "chosen": int, "old_logp": float,
   "advantage": float, "sft_target": [float, ...] | None}`, where candidate
  path `i` is `encode(prompt) + encode(candidates[i])`; or
- `{"candidate_tokens": [[int, ...], ...], "chosen": int, "old_logp": float,
   "advantage": float, "sft_target": [float, ...] | None}`.

`sft_target`, when supplied, is a probability distribution over candidates
that anchors the policy with a cross-entropy term in `classify_rl_loss_fn`.
If omitted on any row, the SFT anchor is disabled for the whole dataset
(sequence-label keys must be uniform inside a pack).

Each candidate path becomes one `TrainSequence`; the actor score head reads
its last token and `classify_rl_loss_fn` normalizes the logits of a decision
with a single softmax. Packing follows `ClassifyTrainer`: whole decisions in
first-fit-decreasing bins, bin count padded to a multiple of DP size, bins
interleaved so the engine's strided DP split lands each bin on the right rank.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import areno.api
from areno.api.dashboard import record_dashboard_state


@dataclass(slots=True)
class EncodedRLDecision:
    """Token paths plus PPO metadata for one decision."""

    leaves: list[list[int]]
    chosen: int
    old_logp: float
    advantage: float
    sft_target: list[float] | None = None

    @property
    def cost(self) -> int:
        return sum(len(leaf) for leaf in self.leaves)


class ClassifyRLTrainer:
    """Offline PPO loop over pre-scored decisions on the score head."""

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        del reward_fn
        from areno.experimental.classify.rl_config import ClassifyRLTrainerConfig

        if not isinstance(config, ClassifyRLTrainerConfig):
            raise TypeError(
                "classify_rl requires ClassifyRLTrainerConfig (it enables the actor score head); "
                "build the trainer through the SDK"
            )
        self.config = config
        self.areno = instance
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def fit(self) -> None:
        self.areno.init()
        try:
            self._fit_initialized()
        finally:
            self.areno.close()

    def _fit_initialized(self) -> None:
        tokenizer = self.areno.get_tokenizer()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_token_id = int(pad_token_id or 0)
        decisions, skipped = encode_decisions(self.dataset, tokenizer, max_seq_len=self.config.max_seq_len)
        self.logger.info("stage=classify_rl_dataset decisions=%d skipped_too_long=%d", len(decisions), skipped)
        if not decisions:
            raise ValueError("classify_rl dataset produced no decisions within max_seq_len")
        has_sft = any(decision.sft_target is not None for decision in decisions)
        if has_sft and not all(decision.sft_target is not None for decision in decisions):
            raise ValueError("classify_rl dataset mixes decisions with and without sft_target")
        dp_size = self.areno.dp_size()
        rng = random.Random(self.config.seed)
        step = 0
        saved_step = None
        for epoch in range(self.config.epochs):
            record_dashboard_state(self.areno, stage="epoch_start", epoch=epoch, step=step, role="policy")
            order = list(range(len(decisions)))
            rng.shuffle(order)
            for start in range(0, len(order), self.config.batch_size):
                chunk = [decisions[index] for index in order[start : start + self.config.batch_size]]
                rows, mini_bs = build_rl_step_rows(
                    chunk,
                    dp_size=dp_size,
                    microbatch_tokens=self.config.microbatch_tokens,
                    pad_token_id=pad_token_id,
                    include_sft_target=has_sft,
                )
                record_dashboard_state(self.areno, stage="train_start", epoch=epoch, step=step, role="policy")
                train_start = time.perf_counter()
                result = self.areno.train(rows, self.loss_fn, mini_bs=mini_bs, gradient_accumulation_steps=None)
                if isinstance(result, dict):
                    result["policy_train_wall_time_s"] = time.perf_counter() - train_start
                    result["classify_rl_microbatches"] = len(rows) // mini_bs
                record_dashboard_state(self.areno, stage="train_end", epoch=epoch, step=step, role="policy")
                self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                step += 1
                if self._maybe_save(step):
                    saved_step = step
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    break
            if self.config.max_steps is not None and step >= self.config.max_steps:
                break
        if self.config.save_path is not None and saved_step != step:
            self._save(step)

    def _maybe_save(self, step: int) -> bool:
        if self.config.save_path is None or step % self.config.save_interval != 0:
            return False
        self._save(step)
        return True

    def _save(self, step: int) -> None:
        # The backbone is written in HF layout; the engine adds score_head.safetensors.
        ckpt_path = str(Path(self.config.save_path) / f"step_{step:06d}")
        self.logger.info("step=%d stage=save_checkpoint_start path=%s", step, ckpt_path)
        saved_path = self.areno.save_checkpoint(ckpt_path)
        self.logger.info("step=%d stage=save_checkpoint_end path=%s", step, saved_path)


def encode_decisions(dataset, tokenizer, *, max_seq_len: int) -> tuple[list[EncodedRLDecision], int]:
    """Tokenize every row; drop decisions whose longest path exceeds `max_seq_len`."""

    decisions = []
    skipped = 0
    for index in range(len(dataset)):
        decision = encode_decision(dataset[index], tokenizer)
        if max(len(leaf) for leaf in decision.leaves) > max_seq_len:
            skipped += 1
            continue
        decisions.append(decision)
    return decisions, skipped


def encode_decision(record: Any, tokenizer) -> EncodedRLDecision:
    """Normalize one dataset row into candidate token paths plus PPO metadata."""

    record = dict(record)
    if "candidate_tokens" in record:
        leaves = [[int(token) for token in leaf] for leaf in record["candidate_tokens"]]
    elif "prompt" in record and "candidates" in record:
        prefix = _encode(tokenizer, str(record["prompt"]))
        leaves = [prefix + _encode(tokenizer, str(candidate)) for candidate in record["candidates"]]
    else:
        raise ValueError(
            "classify_rl rows need `prompt` + `candidates` or `candidate_tokens`, plus `chosen`, `old_logp`, `advantage`"
        )
    if len(leaves) < 2:
        raise ValueError("classify_rl decisions need at least two candidates")
    if any(len(leaf) < 1 for leaf in leaves):
        raise ValueError("classify_rl candidate paths must be non-empty")

    chosen = int(record["chosen"])
    if not 0 <= chosen < len(leaves):
        raise ValueError(f"chosen={chosen} out of range for {len(leaves)} candidates")
    old_logp = float(record["old_logp"])
    advantage = float(record["advantage"])
    if not math.isfinite(old_logp) or not math.isfinite(advantage):
        raise ValueError("old_logp and advantage must be finite")

    sft_target = record.get("sft_target")
    if sft_target is not None:
        sft_target = [float(value) for value in sft_target]
        if len(sft_target) != len(leaves):
            raise ValueError(f"sft_target has {len(sft_target)} entries for {len(leaves)} candidates")
        if any(not math.isfinite(value) or value < 0.0 for value in sft_target):
            raise ValueError("sft_target entries must be non-negative and finite")
        if abs(math.fsum(sft_target) - 1.0) > 1e-4:
            raise ValueError("sft_target must sum to 1")

    return EncodedRLDecision(
        leaves=leaves, chosen=chosen, old_logp=old_logp, advantage=advantage, sft_target=sft_target
    )


def build_rl_step_rows(
    decisions: list[EncodedRLDecision],
    *,
    dp_size: int,
    microbatch_tokens: int,
    pad_token_id: int,
    include_sft_target: bool,
) -> tuple[list, int]:
    """Lay out one optimizer step for PPO; returns `(rows, mini_bs)`."""

    if not decisions:
        raise ValueError("classify_rl step needs at least one decision")
    bins: list[list[int]] = []
    bin_costs: list[int] = []
    for index in sorted(range(len(decisions)), key=lambda item: decisions[item].cost, reverse=True):
        cost = decisions[index].cost
        slot = next((b for b, used in enumerate(bin_costs) if used + cost <= microbatch_tokens), None)
        if slot is None:
            bins.append([])
            bin_costs.append(0)
            slot = len(bins) - 1
        bins[slot].append(index)
        bin_costs[slot] += cost
    while len(bins) % dp_size:
        bins.append([])
    num_microbatches = len(bins) // dp_size
    weight = dp_size * num_microbatches / len(decisions)
    bin_rows = [
        _bin_rows(decisions, members, weight, pad_token_id, include_sft_target) for members in bins
    ]
    rows_per_bin = max(len(rows) for rows in bin_rows)
    padding = _padding_row(pad_token_id, weight, include_sft_target)
    rows = []
    for micro in range(num_microbatches):
        group = bin_rows[micro * dp_size : (micro + 1) * dp_size]
        for position in range(rows_per_bin):
            for rank_rows in group:
                rows.append(rank_rows[position] if position < len(rank_rows) else padding)
    return rows, rows_per_bin * dp_size


def _bin_rows(
    decisions: list[EncodedRLDecision],
    members: list[int],
    weight: float,
    pad_token_id: int,
    include_sft_target: bool,
) -> list:
    rows = []
    for decision_id in members:
        decision = decisions[decision_id]
        for candidate_index, leaf in enumerate(decision.leaves):
            labels = {
                "group": float(decision_id),
                "chosen": 1.0 if candidate_index == decision.chosen else 0.0,
                "old_logp": decision.old_logp,
                "advantage": decision.advantage,
                "weight": weight,
            }
            if include_sft_target:
                labels["sft_target"] = (
                    decision.sft_target[candidate_index] if decision.sft_target is not None else 0.0
                )
            rows.append(_train_row(leaf, pad_token_id, labels))
    return rows


def _padding_row(pad_token_id: int, weight: float, include_sft_target: bool):
    labels = {
        "group": -1.0,
        "chosen": 0.0,
        "old_logp": 0.0,
        "advantage": 0.0,
        "weight": weight,
    }
    if include_sft_target:
        labels["sft_target"] = 0.0
    return _train_row([pad_token_id, pad_token_id], pad_token_id, labels)


def _train_row(tokens: list[int], pad_token_id: int, labels: dict[str, float]):
    zeros = [0.0] * len(tokens)
    return areno.api.TrainSequence(
        tokens=list(tokens),
        prompt_mask=[True] * len(tokens),
        logprobs=zeros,
        advantages=zeros,
        eos_token_id=pad_token_id,
        sequence_labels=labels,
    )


def _encode(tokenizer, text: str) -> list[int]:
    try:
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(token) for token in tokenizer.encode(text)]


__all__ = [
    "ClassifyRLTrainer",
    "EncodedRLDecision",
    "build_rl_step_rows",
    "encode_decision",
    "encode_decisions",
]
