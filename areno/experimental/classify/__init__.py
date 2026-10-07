"""Experimental grouped-softmax classification.

`--algo classify` trains the actor as a sequence scorer: a small head on the
last token's hidden state yields one logit per candidate path, and the
candidates of a question share one softmax fitted with cross-entropy plus
Brier loss. `--algo classify_rl` reuses the same score head and grouped
softmax, but swaps the loss for a PPO-clip objective with optional SFT CE
anchor and entropy bonus. Both algorithms require their respective config
dataclass, which turns on `RuntimeConfig.score_head`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from areno.api.algorithms import AlgorithmSpec, register_algorithm
from areno.api.trainer_config import TrainerConfig
from areno.experimental.classify.config import ClassifyTrainerConfig
from areno.experimental.classify.loss import classify_loss_fn
from areno.experimental.classify.rl_config import ClassifyRLTrainerConfig
from areno.experimental.classify.rl_loss import classify_rl_loss_fn


def _load_classify_trainer() -> type:
    from areno.experimental.classify.trainer import ClassifyTrainer

    return ClassifyTrainer


def _load_classify_rl_trainer() -> type:
    from areno.experimental.classify.rl_trainer import ClassifyRLTrainer

    return ClassifyRLTrainer


def _bind_classify_loss(config: TrainerConfig, loss_fn: Callable) -> Callable:
    return partial(loss_fn, brier_weight=float(getattr(config, "brier_weight", 0.5)))


def _bind_classify_rl_loss(config: TrainerConfig, loss_fn: Callable) -> Callable:
    return partial(
        loss_fn,
        clip_range=float(getattr(config, "clip_range", 0.2)),
        entropy_weight=float(getattr(config, "entropy_weight", 0.0)),
        sft_weight=float(getattr(config, "sft_weight", 0.0)),
    )


register_algorithm(
    AlgorithmSpec(
        name="classify",
        trainer_cls=_load_classify_trainer,
        default_loss_fn=classify_loss_fn,
        requires_rollout=False,
        loss_fn_factory=_bind_classify_loss,
        experimental=True,
    )
)


register_algorithm(
    AlgorithmSpec(
        name="classify_rl",
        trainer_cls=_load_classify_rl_trainer,
        default_loss_fn=classify_rl_loss_fn,
        requires_rollout=False,
        loss_fn_factory=_bind_classify_rl_loss,
        experimental=True,
    )
)


__all__ = [
    "ClassifyRLTrainerConfig",
    "ClassifyTrainerConfig",
    "classify_loss_fn",
    "classify_rl_loss_fn",
]
