PPO
===

.. admonition:: Goal

   Understand AReno's most complete RL algorithm — PPO. From the critic's value
   estimation to GAE advantage computation, from the multi-role architecture to
   the full training command.

.. admonition:: Prerequisites

   Online RL Loop (online RL core loop), GSPO, GRPO.

.. admonition:: Outcomes

   Deep-reading ppo_loss_fn's dual clip → PPOTrainer's 8-step flow → policy_sync
   principle → hands-on command.

1 Why Do You Need a Critic?
------------------------------

GSPO/GRPO compute advantage with an in-group Z-score, but they have a fundamental
limitation:

   **They can only compare relatively; they cannot know whether a given completion
   is objectively good or bad.**

If all 8 completions are wrong (reward is all 0), the advantages are 0 across the
board → no learning signal.

**The critic solves this**: the critic is a value function :math:`V(s)` that
estimates, for any context :math:`s`, "how much reward you can ultimately expect
starting from here." The critic gives an **absolute** judgement: even when every
completion is wrong, the critic can still tell which one, although wrong, is
closer to the correct line of reasoning.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - In-group normalization (GSPO/GRPO)
     - Critic (PPO)
   * - Models needed
     - Only the policy
     - policy + critic (two models)
   * - Advantage computation
     - In-group Z-score
     - GAE (uses the critic's value estimates)
   * - On sparse rewards
     - Weak: no signal for intermediate tokens
     - Strong: GAE back-propagates from the end
   * - Extra overhead
     - 0
     - critic forward + critic training

2 GAE Advantage Estimation
-----------------------------

Problem: Sparse Rewards
~~~~~~~~~~~~~~~~~~~~~~~

In RL, the reward is usually given only at the end of a completion. But the model
needs to know whether each intermediate token is good or bad.

**GAE = Generalized Advantage Estimation** — it uses the critic's value estimates
to back-propagate the terminal reward to every step.

TD Error and Accumulation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)

.. math::

   A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \cdot \delta_{t+l}

- :math:`\delta_t > 0`: this step did better than expected → advantage > 0
- :math:`\gamma` (discount factor, default 1.0) and :math:`\lambda` (GAE parameter, default 1.0)

**The critic's training target**: ``returns = advantages + values`` — the critic
regresses onto the GAE returns (MSE loss).

3 PPO's Multi-Role Architecture
----------------------------------

PPO needs 3-4 model instances working at the same time:

.. list-table::
   :header-rows: 1
   :widths: 13 18 10 59

   * - Role
     - Full name
     - Trained?
     - Purpose
   * - **Actor**
     - Policy network
     - ✓
     - rollout + loss computation + update
   * - **Ref**
     - Reference policy
     - ✗ (frozen)
     - KL divergence reference; keeps the policy from drifting too far
   * - **Critic**
     - Value network
     - ✓
     - Estimates V(s) → GAE → its own MSE loss
   * - **Reward**
     - Reward model
     - ✗ (frozen)
     - Optional; replaces the Python reward_fn

Source (``areno/api/roles.py``):

.. code-block:: python

   @dataclass(slots=True)
   class ModelRole:
       name: str              # "actor" / "ref" / "critic" / "reward"
       path: str              # checkpoint path
       trainable: bool        # whether it participates in training
       optimizer_lr: float    # only effective when trainable=True

The role configuration in PPOTrainer (``areno/api/trainers/ppo.py``):

.. code-block:: python

   self.roles = {
       "actor":  ModelRole("actor",  actor_ckpt,  trainable=True),
       "ref":    ModelRole("ref",    ref_ckpt,    trainable=False),
       "critic": ModelRole("critic", critic_ckpt, trainable=True,
                           optimizer_lr=float(config.critic_lr)),
   }

4 Reading ppo_loss_fn in Depth
---------------------------------

Source: ``areno/api/backend/cuda/losses.py::ppo_loss_fn``

.. code-block:: python

   def ppo_loss_fn(data_pack, logprobs, *, clip_eps=0.2, clip_ratio_c=3.0,
                   use_kl_loss=False, kl_loss_coef=0.001, kl_loss_type="low_var_kl"):
       import torch

       layout = response_layout(data_pack, logprobs,
                                need_old_logprobs=True, need_advantages=True, need_ref_logprobs=True)

       # 1. Decide the KL reference baseline (prefer ref; otherwise old_logprobs)
       reference = layout.old_logprobs if layout.ref_logprobs is None else layout.ref_logprobs

       # 2. The true importance ratio — not a surrogate!
       log_ratio = torch.clamp(logprobs - layout.old_logprobs, min=-20.0, max=20.0)
       ratio = torch.exp(log_ratio)

       # 3. Dual Clip
       clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
       losses1 = -ratio * layout.advantages
       losses2 = -clipped_ratio * layout.advantages
       clipped_losses = torch.maximum(losses1, losses2)      # standard PPO clip
       dual_losses = -layout.advantages * clip_ratio_c        # lower-bound protection (c=3.0)
       losses = torch.where(
           layout.advantages < 0.0,
           torch.minimum(dual_losses, clipped_losses),        # negative advantages: dual protection
           clipped_losses,                                     # positive advantages: standard clip
       )

       # 4. Policy loss + KL penalty
       policy_loss = masked_mean(losses, layout)
       kl = masked_mean(_kl_penalty(logprobs, reference, kl_loss_type), layout)
       total_loss = policy_loss + (kl_loss_coef * kl if use_kl_loss else 0.0)

       return total_loss, {"policy_loss": ..., "kl_loss": ..., "pg_clipfrac": ..., ...}

Key Differences (vs GSPO/GRPO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Real old_logprobs, not a surrogate**

.. code-block:: python

   # GSPO/GRPO: surrogate — same value (ratio=1), gradient path only
   ratio = torch.exp(logprobs - logprobs.detach())

   # PPO: real old_logprobs — from an independent forward of the actor before the step started
   log_ratio = logprobs - layout.old_logprobs  # ← note: this is real! ratio ≠ 1
   ratio = torch.exp(log_ratio)

Because the optimizer has already updated the weights once, the ``old_logprobs``
from the start of the step and the ``logprobs`` computed inside the step are
different.

**2. Dual Clip**

.. list-table::
   :header-rows: 1
   :widths: 22 45 33

   * - Case
     - Loss form
     - Meaning
   * - Positive advantage
     - ``min(-r*A, -clip(r)*A)``
     - ← standard PPO clip
   * - Negative advantage
     - ``min(lower-bound protection, standard clip)``
     - ← extra protection
   * - Here, "lower-bound protection"
     - ``-A * 3.0``
     - the lower-bound protection used for negative advantages

With negative advantages, even if the ratio is extremely large (the policy really
dislikes this token), the loss is bounded by ``-A × 3.0``, preventing overly large
gradients.

**3. KL penalty**

``_kl_penalty`` supports four estimators (``kl_loss_type``):

.. list-table::
   :header-rows: 1
   :widths: 10 32 18 40

   * - Type
     - Formula
     - Alias
     - Characteristics
   * - ``k1``
     - :math:`\log p - \log q`
     - ``kl``
     - Unbiased, no high variance
   * - ``k2``
     - :math:`\frac{1}{2}(\log p - \log q)^2`
     - ``mse``
     - Gaussian approximation
   * - ``k3``
     - :math:`\exp(r) - r - 1`
     - ``low_var_kl``
     - **Default**, low variance
   * - ``abs``
     - :math:`\|\log p - \log q\|`
     - Absolute value
     - Most conservative

5 PPOTrainer's Complete 8-Step Flow
--------------------------------------

.. list-table::
   :widths: 22 78

   * - ``for each prompt_batch:``
     -
   * - 1. Rollout
     - The actor generates ``n_samples`` completions (records rollout_logprobs)
   * - 2. Reward scoring
     - reward_fn(records) → or use the reward model's forward
   * - 3. Forward passes of the three models
     - ``ref_logprobs = ref.forward(tokens)`` # frozen reference

       ``old_logprobs = actor.forward(tokens)`` # the actor at the start of the step

       ``values = critic.forward(tokens)`` # value estimates
   * - 4. GAE
     - ``advantages, returns = compute_gae(rewards, values, gamma, lam)``
   * - 5. Build TrainSequence
     - ``TrainSequence(logprobs=old_logprobs, advantages=GAE advantages, returns=GAE returns, values=critic values, ref_logprobs=ref_logprobs)``
   * - 6. Batch-level advantage normalization
     - ``_normalize_response_advantages(train_batch)`` # mean=0, std=1
   * - 7. Train the critic first
     - ``trainer.train_values("critic", batch)`` # critic regresses values → returns
   * - 8. Then train the actor
     - ``if step >= critic_warmup_steps:``

       ``    trainer.train(batch, ppo_loss_fn)`` # PPO clipped loss + KL penalty

**Key design choices**:

- **Critic before actor**: the critic is updated with the current batch first, making value estimates more accurate
- **Critic warmup**: for the first ``critic_warmup_steps`` steps only the critic is trained; the actor stays frozen
- **Batch-level normalization**: after GAE, the advantages get one more cross-sequence Z-score

6 Hands-On: GSM8K PPO Training Command
-----------------------------------------

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo ppo \
     --n-samples 8 \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 4 \
     --critic-lr 1e-5 \
     --critic-warmup-steps 20 \
     --use-kl-loss \
     --kl-loss-coef 0.001 \
     --kl-loss-type low_var_kl

Differences from the GSPO/GRPO Command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 32 43

   * - Parameter
     - GSPO/GRPO
     - PPO
   * - ``--algo``
     - ``gspo`` / ``grpo``
     - ``ppo``
   * - ``--critic-lr``
     - None
     - ``1e-5`` (critic learning rate, usually 10× the actor lr)
   * - ``--critic-warmup-steps``
     - None
     - ``20`` (warm up the critic first)
   * - ``--use-kl-loss``
     - None
     - Recommended
   * - ``--kl-loss-coef``
     - None
     - ``0.001``
   * - ``--kl-loss-type``
     - None
     - ``low_var_kl`` (default)
   * - ``--ref-ckpt``
     - None
     - Optional (defaults to the actor)
   * - ``--reward-ckpt``
     - None
     - Optional (replaces the Python reward_fn)

PPO's Extra Memory Overhead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PPO loads 2-3 additional model instances compared with GSPO/GRPO, so it puts more
pressure on GPU memory. To run PPO with Ling-3.0-tiny on a DGX Spark (128 GB
unified memory), we recommend:

.. code-block:: bash

   areno train \
     --algo ppo \
     --ckpt ./models/Ling-3.0-tiny \
     --adam-8bit \                    # required: compress optimizer state
     --batch-size 1 \
     --max-running-prompts 1 \
     --max-prompt-tokens 64 \
     --max-new-tokens 16 \
     --drop-rollout-state \           # release KV cache after rollout
     --tp-size 1 --world-size 1

Expected Output (PPO-Specific Metrics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   role=critic stage=train_start
   role=critic metric=value_loss value=0.523   # critic's MSE loss
   role=critic stage=train_end
   role=actor stage=train_start
   train_stats={'policy_loss': 0.015, 'kl_loss': 0.003, 'pg_clipfrac': 0.12, ...}

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Metric
     - Meaning
     - Healthy range
   * - ``critic_value_loss``
     - The critic's MSE loss
     - Should decrease over time
   * - ``policy_loss``
     - The PPO actor loss
     - Non-zero
   * - ``kl_loss``
     - KL divergence
     - <0.01 normal; >0.1 means the policy has drifted too far
   * - ``pg_clipfrac``
     - Proportion of tokens that got clipped
     - 0.05-0.2 normal
   * - ``pg_clipfrac_lower``
     - Proportion of tokens protected by the dual-clip lower bound
     - <0.05 normal

7 policy_sync: Multi-Role Weight Synchronization
---------------------------------------------------

PPO has multiple model roles, which requires weight synchronization (source
``areno/engine/policy_sync.py``).

**Mechanism**:

.. list-table::
   :widths: 62 38

   * - Flow
     - Description
   * - ``train rank (tensor chunk owner)``
     - Start: the tensor chunk owner
   * - ``→ dist.reduce(chunk, dst=source_rank)``
     - # reduce within the training DP
   * - ``→ dist.broadcast(chunk, src=source_rank)``
     - # broadcast to rollout partitions
   * - ``→ bridge rank → rollout TP broadcast``
     - forwarded via the bridge rank to rollout TP
   * - ``→ rollout DP broadcast``
     - broadcast to rollout DP

- Weights are greedily assigned to different training DP ranks by byte size (load balancing)
- Transferred efficiently through the NCCL broadcast group
- ``--policy-sync-bucket-mb`` controls the size of each chunk (default 64 MiB)

8 Chapter Summary
--------------------

1. **The critic replaces in-group relative comparison with absolute value estimates**: GAE back-propagates the terminal reward to every token
2. **PPO needs 3-4 roles**: actor (trainable) + ref (frozen) + critic (trainable) + reward (optional)
3. **Dual clip**: on negative advantages, ``min(dual_losses, clipped_losses)`` provides extra protection
4. **Training order**: critic first, then actor; critic warmup ensures the initial estimates are reasonable
5. **PPO is the most memory-hungry algorithm**: multiple model instances + GAE + KL — always use ``--adam-8bit``
6. **PPO reuses the same dataset_loader and reward_fn as GSPO/GRPO**: just switch to ``--algo ppo`` and add the PPO-specific parameters

The next chapter, DPO, covers the final algorithm: an offline
preference-learning method that involves neither rollout nor a critic.
