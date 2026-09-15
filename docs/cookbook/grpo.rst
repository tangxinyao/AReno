GRPO
====

.. admonition:: Goal

   Read through the GRPO loss function, then switch ``--algo grpo`` on the same
   GSM8K data and compare the actual differences between GSPO and GRPO.

.. admonition:: Prerequisites

   GSPO + GSM8K.

.. admonition:: Outcomes

   Close-read the grpo_loss_fn source → the three key differences from GSPO →
   hands-on comparison command → interpret the training-curve differences.

1 Core Idea
--------------

**GRPO = Group Relative Policy Optimization** (introduced by DeepSeek).

GRPO and GSPO share exactly the same Trainer (``PolicyOnlyTrainer``), the same
rollout flow, and the same in-group advantage computation. **The only difference
is the loss function** — the clipping drops from sequence level to token level.

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Dimension
     - GSPO
     - GRPO
   * - Trainer
     - ``PolicyOnlyTrainer``
     - ``PolicyOnlyTrainer`` (the **same class**)
   * - Rollout
     - ``rollout_token_batch(n_samples)``
     - Exactly the same
   * - Advantage
     - ``compute_group_advantages``
     - Exactly the same
   * - **Clipping granularity**
     - **Sequence-level** (one scalar ratio per completion)
     - **Token-level** (an independent ratio for each token)
   * - ``clip_eps``
     - ``3e-4``
     - ``0.2``
   * - **Loss averaging**
     - Mean over the number of sequences
     - Mean over all response tokens

2 Mathematical Form
----------------------

Token-Level Importance Ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   r_t = \exp(\log\pi_\theta(a_t \mid s_{<t}) - \log\pi_{\text{old}}(a_t \mid s_{<t}))

The comparison with GSPO:

.. list-table::
   :header-rows: 1
   :widths: 14 54 32

   * - Algorithm
     - Formula
     - Explanation
   * - GSPO
     - ``r_seq = exp( mean( logπ_new - logπ_old ) )``
     - ← average first, then exp; a single scalar
   * - GRPO
     - ``r_t = exp( logπ_new - logπ_old )``
     - ← each token independent; N scalars

Token-Level PPO Clip
~~~~~~~~~~~~~~~~~~~~

.. math::

   L_t = \min(r_t \cdot A_t,\ \text{clamp}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t)

.. math::

   \mathcal{L} = \frac{1}{\sum \text{mask}_t} \sum_{t \in \text{response}} L_t

3 A Close Read of the grpo_loss_fn Source
--------------------------------------------

Source: ``areno/api/backend/cuda/losses.py::grpo_loss_fn``

.. code-block:: python

   def grpo_loss_fn(data_pack, logprobs, *, clip_eps: float = 0.2):
       import torch

       # 1. Get a unified response view
       layout = response_layout(data_pack, logprobs,
                                need_old_logprobs=True, need_advantages=True, need_sequences=True)

       # 2. Token-level importance ratio — note: no sequence_sum aggregation!
       ratio = torch.exp(logprobs - logprobs.detach())
       # → shape [B, L] or [total_L], each token independent

       # 3. Token-level PPO clip
       clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))

       # 4. Token-level loss + average over valid_count
       loss = (
           -torch.min(ratio * layout.advantages, clipped * layout.advantages)
           * layout.response_mask          # mask prompt positions
       ).sum() / layout.valid_count        # divide by the total number of response tokens

       return loss, {
           "policy_loss": loss.detach(),
           "ratio_mean": ratio[layout.response_mask.bool()].mean().detach(),
           "ratio_std": ...,
           "advantage_mean": masked_mean(layout.advantages, layout).detach(),
           "logp_diff_mean": masked_mean(layout.old_logprobs - logprobs.detach(), layout).detach(),
           ...
       }

The Three Key Differences from GSPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Difference 1: ratio computation**

.. code-block:: python

   # GSPO: sequence_sum aggregation → one scalar per sequence
   seq_ratio = torch.exp(sequence_sum(logprobs - logprobs.detach(), layout) / layout.response_len)

   # GRPO: direct exp, no aggregation → each token independent
   ratio = torch.exp(logprobs - logprobs.detach())

**Difference 2: loss aggregation**

.. code-block:: python

   # GSPO: seq_ratio shape [B]; .mean() averages over the number of sequences
   loss = -torch.min(seq_ratio * seq_advantage, clipped * seq_advantage).mean()

   # GRPO: ratio shape [B, L] or [total_L]; filter with response_mask, then divide by valid_count
   loss = (-torch.min(ratio * advantages, clipped * advantages) * response_mask).sum() / valid_count

**Difference 3: clip_eps**

.. code-block:: python

   # GSPO: clip_eps=3e-4  (naturally stable after sequence averaging)
   # GRPO: clip_eps=0.2   (each token fluctuates independently)

The reason for the 667× gap: GSPO's ratio is the average over the tokens within a
sequence → variance divided by N; GRPO's ratio is per-token → it needs a much
larger clip to constrain.

4 Hands-On: Switch GRPO on the Same GSM8K Data
-------------------------------------------------

Only one parameter needs to change — replace ``--algo gspo`` with
``--algo grpo``; everything else stays identical:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo grpo \
     --n-samples 8 \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 4

**The dataset_loader and reward_fn are fully reused** — no changes needed. This is
the core value of AReno's one-key ``--algo`` switch.

Training-Curve Comparison: GSPO vs GRPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a set of typical observations (same model, same data, same
hyperparameters; only the algorithm changes):

.. list-table::
   :header-rows: 1
   :widths: 30 24 24 22

   * - Metric
     - GSPO typical
     - GRPO typical
     - Interpretation
   * - ``policy_loss``
     - 0.001-0.01
     - 0.01-0.1
     - GRPO loss is numerically larger because the token-level clip is looser
   * - ``ratio_mean``
     - ~1.000
     - ~1.000
     - Both stay near 1.0 (the result of the surrogate trick)
   * - ``ratio_std``
     - <0.001
     - 0.01-0.05
     - The GRPO ratio fluctuates noticeably more
   * - ``reward_mean`` rise speed
     - Slower but stable
     - May be faster
     - GRPO's token-level control is more sensitive to sparse signals
   * - ``grad_norm``
     - 1-10
     - 5-50
     - GRPO has larger gradients

When to Choose GSPO, When to Choose GRPO?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Scenario
     - Recommendation
     - Reason
   * - Getting started / quick experiments
     - **GSPO**
     - Fewer hyperparameters, more stable, hard to break training
   * - Long sequences (code generation, long text)
     - **GRPO**
     - Token-level control is finer at every position
   * - Paper reproduction (DeepSeek-R1 style)
     - **GRPO**
     - The original paper uses GRPO
   * - Not sure which to pick
     - **GSPO**
     - Get the pipeline running first, then try GRPO

.. admonition:: Conclusion

   Start with GSPO and first make sure the pipeline runs. If the reward rises too
   slowly or is unstable, then try GRPO.

5 New Architecture Component: ``response_layout()`` Full Field Listing
-------------------------------------------------------------------------

GSPO/GRPO/PPO share this key abstraction (source
``areno/api/backend/cuda/losses.py::ResponseLayout``):

.. code-block:: python

   @dataclass(slots=True)
   class ResponseLayout:
       packed: bool              # True=packed (variable-length concatenation), False=padded (rectangular padding)
       response_mask: Tensor     # marks which positions are response tokens
       valid_count: Tensor       # total number of response tokens (scalar)
       response_len: Tensor      # per-sequence response length [B] or scalar
       old_logprobs: Tensor      # old-policy logprobs (from rollout)
       advantages: Tensor        # broadcast per-token advantages
       ref_logprobs: Tensor      # PPO reference-policy logprobs (not used by GSPO/GRPO)
       seq_ids: Tensor           # [total_L] which sequence each token belongs to (packed only)
       num_sequences: int        # total number of sequences (packed only)

**Why is this abstraction needed?** AReno training batches can come in two
formats:

- **Padded**: all sequences are padded to the same length. Simple but wastes
  computation.
- **Packed**: multiple variable-length sequences are concatenated into one long
  sequence. No waste, but ``seq_ids`` is needed to recover the boundaries.

``response_layout()`` unifies both formats behind the same interface. The loss
function does not need to know whether the underlying format is packed or padded.

6 Chapter Summary
--------------------

1. **The only difference between GRPO and GSPO is the loss function**: the
   clipping drops from sequence level to token level
2. **Three key differences**: ratio computation (with/without sequence_sum), loss
   aggregation (.mean() vs /valid_count), clip_eps (3e-4 vs 0.2)
3. **Hands-on needs only one parameter change**: ``--algo grpo``, with
   dataset_loader and reward_fn fully reused
4. **``response_layout()`` is the shared core abstraction of GSPO/GRPO/PPO**: it
   unifies the packed/padded formats
5. **Recommendation**: start with GSPO → advance to GRPO

The next chapter, PPO, introduces the complete Actor-Critic: adding multiple roles such as
critic, ref, and reward model, and replacing in-group standardization with GAE.
