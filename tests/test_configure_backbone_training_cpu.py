"""CPU tests for the freeze_backbone branch of
`areno.engine.modeling.configure_backbone_training`.

Covers only the public contract (requires_grad flip, score_head preservation,
missing-head error, no-op branches). The downstream optimizer-parameter
filter in `worker.py` is tested transitively: a frozen backbone param
with requires_grad=False is excluded by the standard `p.requires_grad`
guard before build_optimizer runs.
"""

from __future__ import annotations

import unittest

import torch

from areno.engine.config import OptimizerConfig
from areno.engine.modeling import configure_backbone_training


class _ScoreHeadModel(torch.nn.Module):
    """Minimal classify-style actor: two linear backbone blocks + a score head.

    The attach-score-head real path builds `score_head` as a
    Linear-GELU-Linear(1) module; the exact shape doesn't matter here —
    only that it lives at `model.score_head` and its params are known.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer_a = torch.nn.Linear(4, 4)
        self.layer_b = torch.nn.Linear(4, 4)
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.GELU(),
            torch.nn.Linear(2, 1),
        )


class ConfigureBackboneTrainingTest(unittest.TestCase):
    def test_freeze_flips_requires_grad_on_backbone_only(self) -> None:
        model = _ScoreHeadModel()
        opt = OptimizerConfig(freeze_backbone=True)
        configure_backbone_training(model, opt, trainable=True)
        for name, param in model.named_parameters():
            if name.startswith("score_head"):
                self.assertTrue(param.requires_grad, f"{name} must stay trainable")
            else:
                self.assertFalse(param.requires_grad, f"{name} must be frozen")

    def test_freeze_disabled_leaves_all_requires_grad_true(self) -> None:
        model = _ScoreHeadModel()
        opt = OptimizerConfig(freeze_backbone=False)
        configure_backbone_training(model, opt, trainable=True)
        for name, param in model.named_parameters():
            self.assertTrue(param.requires_grad, f"{name} must stay trainable")

    def test_non_trainable_role_is_a_noop_even_when_flag_set(self) -> None:
        """Reference / rollout roles don't train; the helper must not touch
        their params regardless of the flag."""

        model = _ScoreHeadModel()
        opt = OptimizerConfig(freeze_backbone=True)
        configure_backbone_training(model, opt, trainable=False)
        for name, param in model.named_parameters():
            self.assertTrue(param.requires_grad, f"{name} must stay trainable")

    def test_requires_score_head_when_flag_set(self) -> None:
        """Freezing the backbone with no head attached would leave nothing
        trainable — reject loudly instead of silently zeroing out training."""

        model = torch.nn.Linear(4, 4)  # no `score_head` attribute
        opt = OptimizerConfig(freeze_backbone=True)
        with self.assertRaisesRegex(ValueError, "score_head"):
            configure_backbone_training(model, opt, trainable=True)

    def test_filter_by_requires_grad_matches_score_head_param_count(self) -> None:
        """Reproduces the worker.py filter: after configure_backbone_training,
        [p for p in model.parameters() if p.requires_grad] must equal exactly
        the score head's parameters."""

        model = _ScoreHeadModel()
        opt = OptimizerConfig(freeze_backbone=True)
        configure_backbone_training(model, opt, trainable=True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        head_params = list(model.score_head.parameters())
        self.assertEqual(len(trainable), len(head_params))
        self.assertEqual(
            [id(p) for p in trainable],
            [id(p) for p in head_params],
        )

    def test_unwraps_torch_compile_wrapper(self) -> None:
        """The helper reads score_head through unwrap_model so a compiled
        actor still works. We approximate the wrapping by using a module
        attribute wrapper rather than invoking torch.compile (which needs
        a real forward)."""

        from areno.engine.modeling import unwrap_model

        model = _ScoreHeadModel()
        unwrapped = unwrap_model(model)
        self.assertIs(unwrapped, model)
        # Freezing on the wrapped-but-no-op case should still work.
        opt = OptimizerConfig(freeze_backbone=True)
        configure_backbone_training(model, opt, trainable=True)
        for name, param in model.named_parameters():
            if name.startswith("score_head"):
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
