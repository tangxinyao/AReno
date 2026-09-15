GSPO
====

.. admonition:: Goal

   Read through the GSPO loss function, then run a complete GSPO training on
   GSM8K with a single command, and understand the full chain from data to loss.

.. admonition:: Prerequisites

   RL in One Snippet (an overview of RL) and Online RL Loop (the online RL core loop).

.. admonition:: Outcomes

   Close-read the gspo_loss_fn source → close-read math_verify_reward →
   close-read dataset_loader → complete training command → interpret the output.

1 Core Idea
--------------

**GSPO = Group-Sampled Policy Optimization**, AReno's simplified variant of PPO.

Three key innovations:

1. **Sequence-level importance ratio**: geometrically average the logprob
   differences of all tokens in one completion to get a single scalar
2. **In-group relative advantage**: in-group Z-score standardization, no critic
   needed
3. **Pure policy optimization**: only one policy role, no ref/critic/reward
   model needed

2 Importance Ratio
---------------------

Sequence-Level vs Token-Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Token-level** (GRPO): each token computes independently
:math:`r_t = \exp(\log\pi_\theta - \log\pi_{\text{old}})`

**Sequence-level** (GSPO): first average the logprob differences over all tokens
of the entire completion, then exp:

.. math::

   r_{\text{seq}} = \exp\left(\frac{1}{N}\sum_{t} (\log\pi_\theta(a_t) - \log\pi_{\text{old}}(a_t))\right)

After sequence averaging the ratio is naturally more stable, which is why GSPO can
use a tiny ``clip_eps=3e-4``.

PPO Clip
~~~~~~~~

.. code-block:: python

   r_seq = exp(mean(logπ_new - logπ_old))      # [B] one scalar per sequence
   clipped = clamp(r_seq, 1-ε, 1+ε)             # clamp to [1-ε, 1+ε]
   L = -min(r_seq * A, clipped * A)             # conservative side

.. list-table::
   :header-rows: 1
   :widths: 18 30 26 26

   * - Value of ratio
     - Meaning
     - If A > 0
     - If A < 0
   * - r > 1
     - The new policy prefers this completion
     - Good (encourage)
     - Bad (overconfident wrong answer)
   * - r < 1
     - The new policy dislikes it
     - Bad (forgot the correct answer)
     - Good (suppresses the wrong answer)
   * - r ≈ 1
     - The policy did not change
     - No signal
     - No signal

3 A Close Read of the gspo_loss_fn Source
--------------------------------------------

Source: ``areno/api/backend/cuda/losses.py::gspo_loss_fn``

.. code-block:: python

   def gspo_loss_fn(data_pack, logprobs, *, clip_eps: float = 3e-4):
       import torch

       # 1. Get a unified response view (supports both packed and padded formats)
       layout = response_layout(data_pack, logprobs,
                                need_old_logprobs=True, need_advantages=True, need_sequences=True)

       # 2. Sequence-level importance ratio
       seq_ratio = torch.exp(
           sequence_sum(logprobs - logprobs.detach(), layout)  # ∑(logπ_new - logπ_old)
           / layout.response_len                               # divide by sequence length → geometric mean
       )

       # 3. PPO clip + sequence-level advantage
       clipped = torch.clamp(seq_ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
       seq_advantage = sequence_sum(layout.advantages, layout) / layout.response_len

       # 4. Final loss
       loss = -torch.min(seq_ratio * seq_advantage, clipped * seq_advantage).mean()

       return loss, {
           "policy_loss": loss.detach(),
           "ratio_mean": seq_ratio.mean().detach(),
           "ratio_std": seq_ratio.std(unbiased=False).detach(),
           "advantage_mean": seq_advantage.mean().detach(),
           "rollout_logprobs_mean": masked_mean(layout.old_logprobs, layout).detach(),
           "train_logprobs_mean": masked_mean(logprobs.detach(), layout).detach(),
           ...
       }

Key Code Walkthrough
~~~~~~~~~~~~~~~~~~~~~

**The ``logprobs - logprobs.detach()`` surrogate trick**:

- ``.detach()`` breaks the gradient while keeping the same values → ratio is
  always 1 (when the weights in the batch have not changed)
- Gradients flow only through the non-detached ``logprobs``
- **This is the key pattern shared by GSPO/GRPO**: use the logprobs of the
  current forward pass to play both "new policy" and "old policy", with the
  gradient path running only through the "new policy" side

**``sequence_sum(values, layout)``**: in packed mode, uses ``scatter_add_`` to
sum by group according to ``seq_ids``; in padded mode, sums along the sequence
dimension.

4 Hands-On: GSM8K Math Reasoning with GSPO
---------------------------------------------

The Dataset: GSM8K
~~~~~~~~~~~~~~~~~~

GSM8K is a dataset of grade-school math word problems. Each record contains a
question and a step-by-step solution, ending with a ``#### answer`` marker:

.. code-block:: text

   question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast..."
   answer: "Janet sells 16 - 3 = 13 eggs per day. She makes 13 * 2 = 26 dollars per day. #### 18"

A Close Read of the Dataset Loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/math/dataset_loader.py``:

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
       dataset = default_loader(dataset_path)
       first = dataset[0]

       # If already normalized → return directly
       if "prompt" in first:
           return dataset

       # GSM8K format → convert
       if "question" in first and "answer" in first:
           return dataset.map(_format_gsm8k_record)
       ...

   def _format_gsm8k_record(record: dict) -> dict:
       answer = str(record["answer"])
       final = answer.rsplit("####", 1)[-1].strip()   # extract "#### 18" → "18"
       return {
           "prompt": (
               "Solve the following grade-school math problem. Show your reasoning "
               "and put the final answer in \\boxed{}.\n\n"
               f"Problem: {record['question']}\nSolution:"
           ),
           "solutions": [final],   # ← this field is auto-mapped to reward_fn's record.answer
       }

**Key point**: the ``solutions`` field (a list) is auto-mapped by the trainer to
``RewardRecord.answer``.

A Close Read of the Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/math/math_verify_reward.py``:

.. code-block:: python

   from math_verify import parse, verify

   def reward_fn(record) -> float:
       """Compare the \boxed{...} in the model output with the ground-truth answer; match → 1.0, no match → 0.0"""
       ground_truth = record.answer[0] if isinstance(record.answer, list) else record.answer
       gt_parsed = parse(ground_truth)          # parse the ground-truth answer ("42", "1/2", etc.)
       pred_parsed = parse(record.completion)   # extract \boxed{...} from the model output
       try:
           return 1.0 if verify(gt_parsed, pred_parsed) else 0.0
       except Exception:
           return 0.0   # tolerate parse errors → treat as a wrong answer

``math_verify`` is a symbolic math verification library that recognizes
``\boxed{42}`` and ``\boxed{42.0}`` as the same answer.

The Complete Training Command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --n-samples 8 \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 4

Parameter notes:

- ``--ckpt Qwen/Qwen3-0.6B``: a small model for quick experiments; switch to
  ``Ling-3.0-tiny`` or a larger model in production
- ``--dataset-path gsm8k:main``: a Hugging Face dataset, format ``repo:config``
- ``--algo gspo``: select the GSPO algorithm
- ``--n-samples 8``: sample 8 completions per prompt, used for in-group
  comparison
- If you use a larger model (7B+), be sure to add ``--adam-8bit`` (see the OOM
  lessons in First Train)

> **Recipe tip**: treat the command above as a recipe. To adapt it to your own
> task, first replace two pieces — ``--dataset-loader-fn`` (the data loader) and
> ``--reward-fn-path`` (the reward function) — and leave the rest of the flags
> untouched. Keep the batch size small until the environment and reward signal
> are stable, then scale up gradually. Full semantics for all training flags are
> in ``reference/cli/training.rst``.

The Full Data Flow Diagram
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Data-flow stage
     - Content / example
   * - Raw data → dataset_loader.py → normalized
     - ``GSM8K raw: {"question": "...", "answer": "...#### 18"}``

       ↓

       ``{"prompt": "Solve the problem... \nProblem: ...\nSolution:", "solutions": ["18"]}``
   * - tokenize + Trainer → rollout
     - ``rollout (n_samples=8)`` → 8 completions

       Example: ``"Let me think... \boxed{18}"``, ``"The answer is \boxed{20}"``, ...
   * - reward_fn(record)
     - ``parse("\boxed{18}")`` vs ``parse("18")`` → ``verify`` → 1.0

       ``parse("\boxed{20}")`` vs ``parse("18")`` → ``verify`` → 0.0

       → rewards = ``[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]``
   * - compute_group_advantages(rewards)
     - → advantages = ``[1.53, -0.65, -0.65, 1.53, -0.65, -0.65, -0.65, 1.53]``
   * - TrainSequence
     - ``TrainSequence(tokens, logprobs=rollout_logprobs, scalar_advantage=..., reward=...)``
   * - gspo_loss_fn
     - ``seq_ratio = exp(mean(logπ_new - logπ_old))``    ← one scalar per completion

       ``L = -min(r * A, clip(r) * A)``                  ← PPO clip

       → backward → optimizer.step()

Reading the Expected Output and Training Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   epoch=0 step=0 stage=rollout_start
   epoch=0 step=0 metric=reward_mean value=0.125
   epoch=0 step=0 stage=train_start
   epoch=0 step=0 train_stats={'policy_loss': 0.02, 'ratio_mean': 1.0, 'advantage_mean': 0.0, ...}

**Healthy signs in the training curves** (watch with TensorBoard or the logs):

.. list-table::
   :header-rows: 1
   :widths: 30 20 26 24

   * - Metric
     - Early training
     - Normal training
     - Anomaly signals
   * - ``reward_mean``
     - ~0.0-0.2
     - Rises gradually
     - Always 0 (data too easy or too hard)
   * - ``policy_loss``
     - Near 0
     - Fluctuates
     - Always 0 (no learning signal)
   * - ``ratio_mean``
     - ~1.0
     - ~1.0
     - >>1.0 or <<0.5 (policy drift too large)
   * - ``rollout_logprobs_mean``
     - -2~-8
     - May change
     - Suddenly a very large negative number (model collapse)
   * - ``grad_norm``
     - >0
     - 1-100
     - Stays 0 for a long time (all rewards 0, no gradients)

.. note:: Hands-on experience (from DGX Spark)

   If the training data is too small (e.g., only 2 GSM8K samples), the model may
   answer everything incorrectly → all rewards 0 → all advantages 0 →
   grad_norm = 0. **This is not a bug — it is normal RL behavior: without
   positive signals there is no learning.** You need to enlarge the dataset or
   raise the temperature so the model has some probability of answering
   correctly.

5 GSPO Hyperparameter Quick Reference
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 28 30

   * - Parameter
     - Default
     - Purpose
     - Tuning advice
   * - ``--algo gspo``
     - —
     - Select GSPO
     - —
   * - ``--gspo-clip-eps``
     - ``3e-4``
     - Sequence-level clip threshold
     - Usually no need to change
   * - ``--n-samples``
     - ``8``
     - Samples per group
     - Larger is more reliable, but rollout time × N
   * - ``--temperature``
     - ``1.0``
     - Sampling temperature
     - Use 1.0 during exploration; you can lower it as it converges
   * - ``--batch-size``
     - ``32``
     - Prompts per step
     - Larger gives more stable gradients
   * - ``--lr``
     - ``1e-6``
     - Learning rate
     - Standard starting point for 7B+ models

6 Chapter Summary
--------------------

1. **GSPO = sequence-level PPO clip + in-group advantage**: three innovations
   keep it minimal while retaining RL effectiveness
2. **The core of gspo_loss_fn**:
   ``sequence_sum(logprobs - logprobs.detach()) / response_len`` → a
   sequence-level geometric-mean ratio
3. **The GSM8K hands-on chain**: dataset_loader (question/answer → prompt/
   solutions) → reward_fn (``\boxed{}`` extraction + math_verify) → GSPO loss →
   optimizer.step()
4. **GSPO is the friendliest choice for getting started with online RL**: few
   hyperparameters, stable, and hard to break training

The next chapter, GRPO, keeps the same Trainer and GSM8K data
— just switch ``--algo grpo``, and the loss drops from sequence-level to
token-level.
