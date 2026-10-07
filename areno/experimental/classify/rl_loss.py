"""PPO-clip objective for sequence-scoring (classification) RL training.

Each training row is still one candidate of one decision, and the actor's
score head turns every row into a scalar logit. The candidates of a decision
share one softmax; its log-prob at the chosen candidate is the policy term.

Rows carry these `sequence_labels`:

- `group`: decision id inside the optimizer step; `-1` marks padding rows.
- `chosen`: `1.0` on the row the behavior policy actually selected, `0.0`
  everywhere else (including every row in a padding bin).
- `old_logp`: log probability of the chosen candidate under the behavior
  policy; replicated across the decision's rows and read from the chosen row.
- `advantage`: scalar advantage for the decision; replicated across the
  decision's rows and read from the chosen row.
- `weight`: per-decision loss weight chosen by the trainer so the DP- and
  microbatch-averaged gradient equals the mean over all decisions in a step.
- `sft_target` (optional): per-row probability that anchors the policy to a
  teacher distribution with a weighted cross-entropy term; omit to disable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_REQUIRED_KEYS = ("group", "chosen", "old_logp", "advantage", "weight")


def classify_rl_loss_fn(
    data_pack,
    scores,
    *,
    clip_range: float = 0.2,
    entropy_weight: float = 0.0,
    sft_weight: float = 0.0,
):
    """Worker-side entry: `scores` holds one logit per packed sequence."""

    labels = data_pack.get("sequence_labels")
    if not isinstance(labels, dict) or any(key not in labels for key in _REQUIRED_KEYS):
        raise ValueError(
            "classify_rl loss requires sequence_labels with group, chosen, old_logp, advantage, weight"
        )
    device = scores.device
    sft_target = labels["sft_target"].to(device=device) if "sft_target" in labels else None
    return grouped_ppo_loss(
        scores,
        group=labels["group"].to(device=device),
        chosen=labels["chosen"].to(device=device),
        old_logp=labels["old_logp"].to(device=device),
        advantage=labels["advantage"].to(device=device),
        weight=labels["weight"].to(device=device),
        sft_target=sft_target,
        clip_range=clip_range,
        entropy_weight=entropy_weight,
        sft_weight=sft_weight,
    )


def grouped_ppo_loss(
    scores: torch.Tensor,
    *,
    group: torch.Tensor,
    chosen: torch.Tensor,
    old_logp: torch.Tensor,
    advantage: torch.Tensor,
    weight: torch.Tensor,
    sft_target: torch.Tensor | None,
    clip_range: float,
    entropy_weight: float,
    sft_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum over decisions of the PPO-clip objective (+ optional SFT CE and entropy)."""

    import torch

    scores = scores.float()
    anchor = scores.sum() * 0.0

    def _zero_metrics() -> dict[str, torch.Tensor]:
        zero = anchor.detach()
        return {
            "rl_policy_loss": zero,
            "rl_entropy": zero,
            "rl_clip_frac": zero,
            "rl_approx_kl": zero,
            "rl_ratio_mean": zero,
            "rl_sft_ce": zero,
            "rl_questions_per_rank": zero,
        }

    valid = group >= 0
    if not bool(valid.any()):
        return anchor, _zero_metrics()

    logits = scores[valid]
    chosen_v = chosen[valid].float()
    old_logp_v = old_logp[valid].float()
    advantage_v = advantage[valid].float()
    weight_v = weight[valid].float()
    _, index = torch.unique(group[valid].long(), return_inverse=True)
    num_groups = int(index.max()) + 1

    def segment_max(values: torch.Tensor) -> torch.Tensor:
        out = values.new_full((num_groups,), float("-inf"))
        return out.scatter_reduce(0, index, values, reduce="amax", include_self=True)

    def segment_sum(values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(num_groups).index_add(0, index, values)

    shifted = logits - segment_max(logits.detach())[index]
    log_probs = shifted - torch.log(segment_sum(torch.exp(shifted)))[index]
    probs = torch.exp(log_probs)

    # Per-group reads from the chosen row; chosen is one-hot inside a group.
    group_logp_new = segment_sum(chosen_v * log_probs)
    group_old_logp = segment_sum(chosen_v * old_logp_v)
    group_advantage = segment_sum(chosen_v * advantage_v)
    group_weight = weight_v.new_zeros(num_groups).scatter_reduce(
        0, index, weight_v, reduce="amax", include_self=False
    )

    diff = group_logp_new - group_old_logp
    ratio = torch.exp(diff)
    unclipped = ratio * group_advantage
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * group_advantage
    pg_loss = -torch.minimum(unclipped, clipped)

    with torch.no_grad():
        clip_mask = (ratio > 1.0 + clip_range) | (ratio < 1.0 - clip_range)
        approx_kl = 0.5 * (diff * diff).mean()

    entropy_per_group = segment_sum(-(probs * log_probs))
    loss_group = pg_loss - entropy_weight * entropy_per_group

    sft_ce_metric = anchor.detach()
    if sft_target is not None and sft_weight > 0.0:
        sft_target_v = sft_target[valid].float()
        sft_ce = segment_sum(-(sft_target_v * log_probs))
        loss_group = loss_group + sft_weight * sft_ce
        sft_ce_metric = sft_ce.detach().mean()

    loss = (group_weight * loss_group).sum() + anchor

    metrics = {
        "rl_policy_loss": pg_loss.detach().mean(),
        "rl_entropy": entropy_per_group.detach().mean(),
        "rl_clip_frac": clip_mask.float().mean(),
        "rl_approx_kl": approx_kl,
        "rl_ratio_mean": ratio.detach().mean(),
        "rl_sft_ce": sft_ce_metric,
        "rl_questions_per_rank": torch.tensor(float(num_groups), device=scores.device),
    }
    return loss, metrics


__all__ = ["classify_rl_loss_fn", "grouped_ppo_loss"]
