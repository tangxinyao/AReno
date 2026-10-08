"""Trainer config for grouped-softmax PPO on classification heads."""

from __future__ import annotations

from dataclasses import dataclass

from areno.api.trainer_config import TrainerConfig


@dataclass(slots=True)
class ClassifyRLTrainerConfig(TrainerConfig):
    """Settings for `ClassifyRLTrainer`.

    `batch_size` counts decisions per optimizer step. Each decision's candidate
    paths are packed whole into per-DP-rank microbatches of at most
    `microbatch_tokens` real tokens; `mini_bs` and
    `gradient_accumulation_steps` are derived per step and ignored here.
    Decisions whose longest candidate path exceeds `max_seq_len` are skipped.

    `epochs` doubles as the number of PPO passes over the dataset. For
    stability keep `batch_size` close to the full dataset so each epoch is a
    single data pass.
    """

    clip_range: float = 0.2
    entropy_weight: float = 0.0
    sft_weight: float = 0.0
    microbatch_tokens: int = 24000
    max_seq_len: int = 1536
    score_head_lr: float = 2.0e-4
    score_head_warmup_steps: int = 0
    freeze_backbone: bool = False
    seed: int = 17

    def __post_init__(self) -> None:
        TrainerConfig.__post_init__(self)
        if self.backend != "cuda":
            raise ValueError("classify_rl training is only supported by the CUDA backend")
        if self.lora is not None:
            raise ValueError("classify_rl training does not support native LoRA")
        if not (0.0 < self.clip_range < 1.0):
            raise ValueError("clip_range must be in (0, 1)")
        if self.entropy_weight < 0.0:
            raise ValueError("entropy_weight must be non-negative")
        if self.sft_weight < 0.0:
            raise ValueError("sft_weight must be non-negative")
        if self.microbatch_tokens < 1:
            raise ValueError("microbatch_tokens must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.score_head_lr <= 0.0:
            raise ValueError("score_head_lr must be positive")
        if self.score_head_warmup_steps < 0:
            raise ValueError("score_head_warmup_steps must be non-negative")
        if self.freeze_backbone:
            # Head-only training. Force the backbone LR to 0 so the optimizer
            # cannot update it even if upstream code forwards a nonzero value
            # from `--backbone-lr`. Score head still trains at score_head_lr.
            self.optimizer_lr = 0.0

    def optimizer_config(self) -> dict:
        """Add the score-head LR group and head-only warmup."""

        config = TrainerConfig.optimizer_config(self)
        config["score_head_lr"] = self.score_head_lr
        config["score_head_warmup_steps"] = self.score_head_warmup_steps
        config["freeze_backbone"] = self.freeze_backbone
        return config

    def cuda_config(self):
        """Enable the actor score head on the CUDA engine."""

        config = TrainerConfig.cuda_config(self)
        config.runtime["score_head"] = True
        return config


__all__ = ["ClassifyRLTrainerConfig"]
