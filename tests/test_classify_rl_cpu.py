"""CPU tests for experimental grouped-softmax PPO training (`--algo classify_rl`)."""

from __future__ import annotations

import math
import unittest

import torch

from areno.api.backend.cuda.training import make_train_pack
from areno.engine.config import OptimizerConfig, RuntimeConfig
from areno.engine.runtime.common import split_data_pack_by_dp
from areno.experimental.classify.rl_config import ClassifyRLTrainerConfig
from areno.experimental.classify.rl_loss import classify_rl_loss_fn, grouped_ppo_loss
from areno.experimental.classify.rl_trainer import (
    EncodedRLDecision,
    build_rl_step_rows,
    encode_decision,
)


def _ref_ppo(logits, chosen, old_logp, advantage, clip_range):
    """Minimal single-decision PPO-clip objective for cross-checking."""

    log_probs = torch.log_softmax(logits, dim=-1)
    logp_new = log_probs[chosen]
    ratio = torch.exp(logp_new - old_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
    return -torch.minimum(unclipped, clipped)


def _ref_entropy(logits):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum()


def _ref_ce(logits, target):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum()


class GroupedPPOLossTest(unittest.TestCase):
    def _label_tensors(self, groups, chosens, old_logps, advantages, weights, *, sft_targets=None):
        return {
            "group": torch.tensor(groups, dtype=torch.float32),
            "chosen": torch.tensor(chosens, dtype=torch.float32),
            "old_logp": torch.tensor(old_logps, dtype=torch.float32),
            "advantage": torch.tensor(advantages, dtype=torch.float32),
            "weight": torch.tensor(weights, dtype=torch.float32),
            **({"sft_target": torch.tensor(sft_targets, dtype=torch.float32)} if sft_targets is not None else {}),
        }

    def test_matches_hand_computed_ppo_and_ignores_padding(self):
        torch.manual_seed(0)
        # Two decisions (3 and 4 candidates) with a padding row sandwiched in.
        scores = torch.randn(8, requires_grad=True)
        groups = [0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0]
        chosens = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        old_logp_a = -1.3
        old_logp_b = -0.5
        advantage_a = 0.8
        advantage_b = -0.4
        old_logps = [old_logp_a, old_logp_a, old_logp_a, 0.0, old_logp_b, old_logp_b, old_logp_b, old_logp_b]
        advantages = [advantage_a, advantage_a, advantage_a, 0.0, advantage_b, advantage_b, advantage_b, advantage_b]
        weights = [0.5] * 8

        clip_range = 0.2
        loss, metrics = grouped_ppo_loss(
            scores,
            group=torch.tensor(groups),
            chosen=torch.tensor(chosens),
            old_logp=torch.tensor(old_logps),
            advantage=torch.tensor(advantages),
            weight=torch.tensor(weights),
            sft_target=None,
            clip_range=clip_range,
            entropy_weight=0.0,
            sft_weight=0.0,
        )
        loss.backward()
        grad = scores.grad.clone()

        ref_scores = scores.detach().clone().requires_grad_(True)
        ref_loss = _ref_ppo(ref_scores[[0, 1, 2]], chosen=1, old_logp=old_logp_a, advantage=advantage_a, clip_range=clip_range)
        ref_loss = ref_loss + _ref_ppo(ref_scores[[4, 5, 6, 7]], chosen=2, old_logp=old_logp_b, advantage=advantage_b, clip_range=clip_range)
        (ref_loss * 0.5).backward()

        torch.testing.assert_close(loss.detach(), ref_loss.detach() * 0.5)
        torch.testing.assert_close(grad, ref_scores.grad)
        self.assertEqual(float(grad[3]), 0.0)
        self.assertEqual(float(metrics["rl_questions_per_rank"]), 2.0)
        # ratio_a = exp(logp_new - old_logp); at init logp_new ~= old_logp only by chance,
        # so just sanity-check the recorded mean is finite and clip fraction in [0, 1].
        self.assertTrue(math.isfinite(float(metrics["rl_ratio_mean"])))
        self.assertGreaterEqual(float(metrics["rl_clip_frac"]), 0.0)
        self.assertLessEqual(float(metrics["rl_clip_frac"]), 1.0)

    def test_entropy_term_subtracts_group_entropy(self):
        scores = torch.tensor([2.0, 0.5, -1.0, 1.0, 0.0], requires_grad=True)
        groups = [0.0, 0.0, 0.0, 1.0, 1.0]
        chosens = [1.0, 0.0, 0.0, 0.0, 1.0]
        old_logps = [-0.1] * 5
        advantages = [0.0] * 5  # drop the PG term so only entropy drives the loss
        weights = [1.0] * 5

        loss, metrics = grouped_ppo_loss(
            scores,
            group=torch.tensor(groups),
            chosen=torch.tensor(chosens),
            old_logp=torch.tensor(old_logps),
            advantage=torch.tensor(advantages),
            weight=torch.tensor(weights),
            sft_target=None,
            clip_range=0.2,
            entropy_weight=0.25,
            sft_weight=0.0,
        )
        expected = -0.25 * (_ref_entropy(scores[:3]) + _ref_entropy(scores[3:]))
        torch.testing.assert_close(loss, expected)
        self.assertGreater(float(metrics["rl_entropy"]), 0.0)

    def test_sft_target_adds_cross_entropy_anchor(self):
        scores = torch.tensor([1.5, -0.2, 0.4], requires_grad=True)
        groups = [0.0, 0.0, 0.0]
        chosens = [1.0, 0.0, 0.0]
        old_logps = [-0.8, -0.8, -0.8]
        advantages = [0.0, 0.0, 0.0]
        weights = [1.0, 1.0, 1.0]
        sft_target = [0.6, 0.3, 0.1]

        loss, metrics = grouped_ppo_loss(
            scores,
            group=torch.tensor(groups),
            chosen=torch.tensor(chosens),
            old_logp=torch.tensor(old_logps),
            advantage=torch.tensor(advantages),
            weight=torch.tensor(weights),
            sft_target=torch.tensor(sft_target),
            clip_range=0.2,
            entropy_weight=0.0,
            sft_weight=0.5,
        )
        expected = 0.5 * _ref_ce(scores, torch.tensor(sft_target))
        torch.testing.assert_close(loss, expected)
        self.assertGreater(float(metrics["rl_sft_ce"]), 0.0)

    def test_all_padding_rank_returns_connected_zero(self):
        scores = torch.randn(4, requires_grad=True)
        labels = self._label_tensors(
            groups=[-1.0] * 4, chosens=[0.0] * 4, old_logps=[0.0] * 4, advantages=[0.0] * 4, weights=[1.0] * 4
        )
        loss, metrics = classify_rl_loss_fn(
            {"sequence_labels": labels}, scores, clip_range=0.2, entropy_weight=0.1, sft_weight=0.0
        )
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.equal(scores.grad, torch.zeros(4)))
        self.assertEqual(float(metrics["rl_questions_per_rank"]), 0.0)

    def test_loss_fn_rejects_missing_keys(self):
        pack = {"sequence_labels": {"group": torch.tensor([0.0, 0.0])}}
        with self.assertRaisesRegex(ValueError, "classify_rl loss requires"):
            classify_rl_loss_fn(pack, torch.tensor([1.0, 0.0]))


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]


class EncodeDecisionTest(unittest.TestCase):
    def test_prefix_plus_candidate_tokens_and_metadata(self):
        decision = encode_decision(
            {
                "prompt": "ab",
                "candidates": ["c", "de"],
                "chosen": 1,
                "old_logp": -0.7,
                "advantage": 0.3,
                "sft_target": [0.4, 0.6],
            },
            _Tokenizer(),
        )
        self.assertEqual(decision.leaves, [[97, 98, 99], [97, 98, 100, 101]])
        self.assertEqual(decision.chosen, 1)
        self.assertAlmostEqual(decision.old_logp, -0.7)
        self.assertAlmostEqual(decision.advantage, 0.3)
        self.assertEqual(decision.sft_target, [0.4, 0.6])

    def test_rejects_chosen_out_of_range(self):
        record = {"prompt": "a", "candidates": ["b", "c"], "chosen": 5, "old_logp": 0.0, "advantage": 0.0}
        with self.assertRaisesRegex(ValueError, "chosen"):
            encode_decision(record, _Tokenizer())

    def test_rejects_non_distribution_sft_target(self):
        record = {
            "prompt": "a",
            "candidates": ["b", "c"],
            "chosen": 0,
            "old_logp": 0.0,
            "advantage": 0.0,
            "sft_target": [0.6, 0.6],
        }
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            encode_decision(record, _Tokenizer())


class StepLayoutTest(unittest.TestCase):
    def _decisions(self):
        return [
            EncodedRLDecision(
                leaves=[[1] * (5 + q), [2] * (3 + q), [3] * 4][: 2 + q % 2],
                chosen=q % 2,
                old_logp=-0.5,
                advantage=0.1 * (q + 1),
                sft_target=None,
            )
            for q in range(6)
        ]

    def test_decisions_stay_whole_per_rank_and_microbatch(self):
        decisions = self._decisions()
        dp_size = 2
        rows, mini_bs = build_rl_step_rows(
            decisions,
            dp_size=dp_size,
            microbatch_tokens=20,
            pad_token_id=0,
            include_sft_target=False,
        )
        self.assertEqual(len(rows) % mini_bs, 0)
        self.assertEqual(mini_bs % dp_size, 0)
        num_micro = len(rows) // mini_bs
        owner = {}
        chosen_total = 0
        for micro in range(num_micro):
            chunk = rows[micro * mini_bs : (micro + 1) * mini_bs]
            for rank in range(dp_size):
                for row in chunk[rank::dp_size]:
                    labels = row.sequence_labels
                    self.assertEqual(set(labels), {"group", "chosen", "old_logp", "advantage", "weight"})
                    gid = int(labels["group"])
                    if gid < 0:
                        self.assertEqual(labels["chosen"], 0.0)
                        self.assertEqual(len(row.tokens), 2)
                        continue
                    chosen_total += int(labels["chosen"])
                    self.assertEqual(owner.setdefault(gid, (micro, rank)), (micro, rank))
                    self.assertAlmostEqual(labels["weight"], dp_size * num_micro / len(decisions))
        self.assertEqual(set(owner), set(range(len(decisions))))
        self.assertEqual(chosen_total, len(decisions))  # one chosen row per decision

    def test_sft_target_included_only_when_requested(self):
        decisions = self._decisions()
        for d in decisions:
            d.sft_target = [1.0 / len(d.leaves)] * len(d.leaves)
        rows_with, _ = build_rl_step_rows(
            decisions, dp_size=1, microbatch_tokens=1000, pad_token_id=0, include_sft_target=True
        )
        rows_without, _ = build_rl_step_rows(
            decisions, dp_size=1, microbatch_tokens=1000, pad_token_id=0, include_sft_target=False
        )
        self.assertIn("sft_target", rows_with[0].sequence_labels)
        self.assertNotIn("sft_target", rows_without[0].sequence_labels)

    def test_pack_labels_follow_dp_split(self):
        decisions = self._decisions()
        rows, mini_bs = build_rl_step_rows(
            decisions, dp_size=2, microbatch_tokens=1000, pad_token_id=0, include_sft_target=False
        )
        pack = make_train_pack(rows[:mini_bs])
        labels = pack["sequence_labels"]
        self.assertEqual(set(labels), {"group", "chosen", "old_logp", "advantage", "weight"})
        self.assertEqual(labels["group"].shape, (mini_bs,))
        for rank, shard in enumerate(split_data_pack_by_dp(pack, 2)):
            expected = [row.sequence_labels["group"] for row in rows[:mini_bs][rank::2]]
            self.assertEqual(shard["sequence_labels"]["group"].tolist(), expected)
            self.assertEqual(shard["input_ids"].shape[0], len(expected))


class ClassifyRLConfigTest(unittest.TestCase):
    def test_backend_config_enables_score_head(self):
        config = ClassifyRLTrainerConfig(
            algo="classify_rl",
            ckpt="unused",
            dataset_path="unused",
            backend="cuda",
            score_head_lr=1e-3,
            score_head_warmup_steps=4,
            clip_range=0.3,
            entropy_weight=0.05,
            sft_weight=0.1,
        )
        cuda = config.cuda_config()
        self.assertTrue(RuntimeConfig(**cuda.runtime).score_head)
        optimizer = OptimizerConfig(**cuda.optimizer)
        self.assertEqual(optimizer.score_head_lr, 1e-3)
        self.assertEqual(optimizer.score_head_warmup_steps, 4)

    def test_registered_loss_binds_rl_hyperparams(self):
        from areno.api.algorithms import get_algorithm

        config = ClassifyRLTrainerConfig(
            algo="classify_rl",
            ckpt="unused",
            dataset_path="unused",
            backend="cuda",
            clip_range=0.15,
            entropy_weight=0.02,
            sft_weight=0.4,
        )
        spec = get_algorithm("classify_rl")
        self.assertFalse(spec.requires_rollout)
        self.assertTrue(spec.experimental)
        loss_fn = spec.make_loss_fn(config)
        self.assertIs(loss_fn.func, classify_rl_loss_fn)
        self.assertEqual(loss_fn.keywords, {"clip_range": 0.15, "entropy_weight": 0.02, "sft_weight": 0.4})

    def test_rejects_bad_hyperparams(self):
        with self.assertRaisesRegex(ValueError, "clip_range"):
            ClassifyRLTrainerConfig(
                algo="classify_rl", ckpt="unused", dataset_path="unused", backend="cuda", clip_range=0.0
            )
        with self.assertRaisesRegex(ValueError, "entropy_weight"):
            ClassifyRLTrainerConfig(
                algo="classify_rl", ckpt="unused", dataset_path="unused", backend="cuda", entropy_weight=-1.0
            )

    def test_freeze_backbone_defaults_to_false(self):
        config = ClassifyRLTrainerConfig(
            algo="classify_rl", ckpt="unused", dataset_path="unused", backend="cuda",
        )
        self.assertFalse(config.freeze_backbone)
        optimizer = OptimizerConfig(**config.optimizer_config())
        self.assertFalse(optimizer.freeze_backbone)

    def test_freeze_backbone_forces_optimizer_lr_zero(self):
        config = ClassifyRLTrainerConfig(
            algo="classify_rl", ckpt="unused", dataset_path="unused", backend="cuda",
            optimizer_lr=1e-4, freeze_backbone=True,
        )
        self.assertEqual(config.optimizer_lr, 0.0)

    def test_freeze_backbone_propagates_to_optimizer_config(self):
        config = ClassifyRLTrainerConfig(
            algo="classify_rl", ckpt="unused", dataset_path="unused", backend="cuda",
            freeze_backbone=True,
        )
        optimizer = OptimizerConfig(**config.optimizer_config())
        self.assertTrue(optimizer.freeze_backbone)
        self.assertEqual(optimizer.lr, 0.0)
        self.assertEqual(optimizer.score_head_lr, config.score_head_lr)

    def test_freeze_backbone_still_requires_positive_head_lr(self):
        with self.assertRaisesRegex(ValueError, "score_head_lr"):
            ClassifyRLTrainerConfig(
                algo="classify_rl", ckpt="unused", dataset_path="unused", backend="cuda",
                freeze_backbone=True, score_head_lr=0.0,
            )


if __name__ == "__main__":
    unittest.main()
