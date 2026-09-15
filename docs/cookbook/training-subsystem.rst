Training Subsystem
===================

.. admonition:: Goal

   Understand every stage of the AReno training subsystem — how Optimizer Step executes, how Activation Checkpointing saves memory, the layout difference between Packing and Padding, and how Checkpoint I/O and Policy Sync manage model weights.

.. admonition:: Prerequisites

   Online RL Loop (Online RL core loop), Inference Subsystem.

.. admonition:: Outcome

   Understand everything that happens behind a ``backend.train(batch)`` call.

The training subsystem is the stage of the RL loop that "sends the gradients back into the model." AReno's ``TrainingManager`` (``areno/engine/training.py``, 295 lines) coordinates the optimizer step, activation checkpointing, and gradient synchronization, and works together with checkpoint I/O and policy sync to manage the lifecycle of model weights.

1 Optimizer Step: The Full Picture of a Training Iteration
-------------------------------------------------------------

TrainingManager Core Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/engine/training.py:28-63
   class TrainingManager:
       def train(self, payload: TrainPayload) -> list[dict | None]:

One ``train()`` call corresponds to one optimizer step and includes:

1. **Gradient accumulation support**: ``payload.gradient_accumulation_steps`` controls how many microbatches are accumulated before one step is taken
2. **zero_grad**: zero the gradients at the start of every accumulation group
3. **Run** ``_train_step()`` **per microbatch**
4. **Trigger** optimizer.step() **on the final step**

_train_step in Detail
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/engine/training.py:64-176
   def _train_step(self, data_pack_shards, *, allow_step, grad_scale):
       # 1. Ensure the training weights are in place
       worker._prepare_for_train()

       # 2. Model forward
       out = train_model(
           input_ids=tokens,
           position_ids=position_ids,
           train_meta=_train_meta(data_pack, tokens),
       )

       # 3. Compute logprobs (distinguishing padded / packed layouts)
       if "train_cu_seqlens" in data_pack:
           logprobs = packed_next_token_logprobs(...)
       else:
           logprobs = next_token_logprobs(...)

       # 4. Compute the loss (via the algorithm's loss_fn)
       loss = worker.loss_fn(data_pack, logprobs)

       # 5. Backward pass
       (loss / grad_scale).backward()

       # 6. Gradient accumulation
       self._accumulate_main_gradients()

       # 7. When allow_step=True:
       #    a. sync DP gradients (all-reduce)
       #    b. sync TP replicated gradients
       #    c. gradient clipping
       #    d. optimizer.step()
       #    e. global_step += 1

Gradient Management: FP32 Master Weights and 8-bit Adam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AReno stores model parameters (weights) in BF16 to save memory, but the optimizer state needs a more precise representation.

**Two Optimizer implementations** (``areno/engine/optim/``):

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Optimizer
     - Storage strategy
     - Use case
   * - ``AdamWFP32Master``
     - BF16 model weights + FP32 master copy (stored in bucketed shards)
     - Standard training
   * - ``AdamW8bit``
     - BF16 model weights + uint8 quantized optimizer state (no FP32 master)
     - When GPU memory is tight

AdamWFP32Master Bucket Design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # areno/engine/optim/adamw_fp32_master.py
   # _MasterBucket: one bucket holds the FP32 master + exp_avg + exp_avg_sq of several parameters
   # default bucket size: 16M parameters (64MB FP32)
   # each step only converts this bucket's parameters from BF16 → FP32

The key benefit of bucketing: **you don't need to convert all model parameters to FP32 at once** (that would double memory) — instead you process them bucket by bucket.

AdamW8bit Quantization Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # areno/engine/optim/adamw_8bit.py
   # exp_avg: symmetric quantization (can be positive or negative)
   # exp_avg_sq: positive quantization (always non-negative)
   # each parameter's optimizer state shrinks from 2×FP32=8 bytes to 2×uint8=2 bytes

**The gradient accumulation path is the same for both optimizers**:
1. autograd computes ``.grad`` (BF16)
2. ``_accumulate_main_gradients()`` accumulates into the FP32 master copy or the uint8 quantized state
3. the optimizer step is computed in FP32 or quantized precision

Learning-Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/engine/training.py:178-199
   @staticmethod
   def _scheduled_lr(step, *, base_lr, min_lr, decay_steps, decay_style, warmup_steps=0):

It supports three decay strategies:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - Behavior
   * - ``constant``
     - Constant ``base_lr``
   * - ``linear``
     - Linear decay down to ``min_lr``
   * - ``cosine``
     - Cosine decay down to ``min_lr`` (the common transformers strategy)

All strategies support warmup: for the first ``warmup_steps`` steps the learning rate grows linearly from 0 to ``base_lr``.

2 Recompute / Activation Checkpointing: Trading Time for Memory
-------------------------------------------------------------------

Problem
~~~~~~~~~~

During training the forward pass needs to keep intermediate activations (each layer's hidden states). Take Ling-3.0-tiny as an example:
- 28 decoder layers, hidden size 2048 per layer
- the larger batch × seqlen, the more activations you keep
- activations can take more memory than the model weights themselves

Solution: Don't Store Everything — Recompute in Backward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AReno implements activation checkpointing in ``areno/engine/runtime/recompute.py`` (84 lines).

.. code-block:: python

   # areno/engine/runtime/recompute.py:34-51
   @_disable_dynamo_frame
   def checkpoint_layer(layer_fn, hidden_states, *args, train_meta=None, infer_meta=None):
       if not should_checkpoint_layer(train_meta, infer_meta):
           return layer_fn(hidden_states, *args)       # no checkpoint: compute directly
       return checkpoint(
           lambda states: layer_fn(states, *args),     # checkpoint: recompute in backward
           hidden_states,
           use_reentrant=False,
           preserve_rng_state=True,
       )

Two Checkpoint Granularities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard checkpoint**: only stores each layer's **input** hidden states and recomputes the layer in backward → activation memory is halved (none of the per-layer intermediate results are stored)

**MoE checkpoint** (``checkpoint_routed_moe_layer``):

.. code-block:: python

   # areno/engine/runtime/recompute.py:54-84
   def checkpoint_routed_moe_layer(
       attention_fn, post_attention_norm, route_fn, expert_fn, hidden_states, ...
   ):
       # 1. Checkpoint the attention sublayer
       attended = checkpoint_layer(attention_fn, hidden_states, ...)
       # 2. Routing is not checkpointed (topk_idx/topk_weight must be kept)
       topk_idx, topk_weight = route_fn(normalized)
       # 3. Checkpoint the expert sublayer
       expert_output = checkpoint_layer(expert_fn, normalized, topk_idx, topk_weight, ...)
       return attended + expert_output

**Why routing is not checkpointed**: ``topk_idx`` is an integer index (structural information); in backward you need to know which tokens were routed to which expert. If routing were checkpointed, the backward pass would have to re-run the routing logic and produce a ``topk_idx`` exactly identical to the forward pass — a precision risk under floating-point arithmetic.

Cost and Benefit
~~~~~~~~~~~~~~~~~~~~~

- **Memory saved**: roughly 50% (store inputs only, not intermediate activations)
- **Speed cost**: roughly 20-30% (an extra forward pass during backward)
- **Configuration**: ``--activation-checkpointing`` (enabled by default)

3 Packing vs Padding: Two Batch Layouts
--------------------------------------------

Problem
~~~~~~~~~~

Completions from different prompts have different lengths. How do you organize them into one batch (tensor) for the model forward pass?

Padding (Pad-Aligned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   sequence 1: [A, B, C, D, <eos>, PAD, PAD, PAD]
   sequence 2: [X, Y, <eos>, PAD, PAD, PAD, PAD, PAD]
   sequence 3: [M, N, O, P, Q, R, <eos>, PAD]

- **Pros**: simple to implement; regular tensor shape (batch, max_len)
- **Cons**: PAD tokens waste compute (roughly 30-40% of tokens are wasted computation)

Packing (Variable-Length Concatenation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   packed: [A, B, C, D, <eos>, X, Y, <eos>, M, N, O, P, Q, R, <eos>]
   boundaries:   cu_seqlens = [0, 5, 8, 15]

- **Pros**: zero waste — every token is useful computation
- **Cons**: needs ``cu_seqlens`` to mark boundaries, and the attention mask must be block-diagonal rather than full causal

AReno's Unified Abstraction: response_layout()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both layouts are transparent to the loss function, via the unified ``response_layout()`` interface:

.. code-block:: text

   # whether padded or packed, the loss function gets everything via response_layout:
   response_layout(data_pack)
   → {
       response_mask,    # which positions are response tokens
       old_logprobs,     # logprobs from rollout
       advantages,       # advantage of each token
       response_len,     # number of tokens in the response part
       valid_count,      # total number of valid response tokens
   }

When to Use Packing?
~~~~~~~~~~~~~~~~~~~~~~~~~~

- When sequence lengths differ a lot (packing pays off most)
- Enabled by default; can be disabled with ``--no-packing``

4 Checkpoint I/O: Saving and Loading Model Weights
-------------------------------------------------------

AReno's checkpoint system is split into two layers (``areno/engine/checkpoints/``):

File Responsibilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - File
     - Lines
     - Responsibility
   * - ``common.py``
     - 1300 lines
     - Checkpoint structure declarations (which tensors each layer has, how they are sharded)
   * - ``io.py``
     - 1092 lines
     - Low-level I/O primitives (safetensors read/write, TP gather, NCCL broadcast)

What Is Saved
~~~~~~~~~~~~~~~~~~~

One ``save_checkpoint`` includes:
- **Model weights**: written sharded in the HF-compatible safetensors format
- **Optimizer state**: FP32 master weights / uint8 quantized moments
- **Training metadata**: ``global_step`` (the current training step) → used for resume
- **Configuration snapshot**: model config, engine config → compatibility is verified on reload

Loading Strategy: Load Only What You Need
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RL's multi-role architecture means different roles need different parameter versions:

.. list-table::
   :widths: 25 75

   * - ``Actor``
     - ← actor weights from the checkpoint (the version being trained)
   * - ``Ref``
     - ← ref weights from the checkpoint (a frozen snapshot)
   * - ``Critic``
     - ← critic weights from the checkpoint (if training PPO)
   * - ``Reward``
     - ← reward weights from the checkpoint (if scoring with a model)

AReno's checkpoint system supports **loading only the weights the current role needs**, avoiding wasted memory and time.

Key Technical Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Safetensors format**: AReno stores weights with safetensors (rather than pickle-based PyTorch), because:
- Safe: contains no executable code
- Efficient: supports lazy loading (only the tensors you need are read)
- Zero-copy: can be mmap'ed straight onto the GPU

**TP-aware**: when saving under tensor parallelism, the checkpoint system automatically gathers sharded parameters into a full tensor before writing. On load it automatically scatters to every TP rank.

Policy Sync
~~~~~~~~~~~~~~~~~

In multi-role scenarios (especially PPO's separated train+rollout architecture), weights updated in the train process need to be synchronized to the rollout process.

.. code-block:: python

   # areno/engine/policy_sync.py:69-134
   def transfer_policy_weights(worker, payload: PolicySyncPayload):

Synchronization uses a **NCCL relay architecture**:

.. list-table::
   :widths: 35 65

   * - →
     - Train DP rank (owner) → broadcast via policy_publisher_group
   * - →
     - Bridge rank (not on the same physical GPU) → forward to the TP row the Bridge belongs to
   * - →
     - each TP column broadcasts to its DP group

Key design principles:
- **weights travel over GPU-side NCCL directly** (not the CPU→disk→CPU path)
- **Greedy load balancing**: ``assign_policy_owners()`` distributes by parameter size across different train DP ranks so no single rank becomes a bottleneck
- **Chunked transfer**: large parameters are transferred in chunks (``bucket_bytes`` controls the chunk size) to prevent OOM

5 The Full Call Chain of the Training Subsystem
----------------------------------------------------

.. list-table::
   :widths: 100

   * - Call chain
   * - Trainer.train(batch)
   * - → Backend.train(batch)
   * - → Engine dispatches to Worker
   * - → TrainingManager.train(payload)
   * - for each microbatch:
   * - 1. _prepare_for_train()          # ensure the training weights are on the GPU
   * - 2. train_model(\*\*data_pack)      # forward pass (may include activation checkpointing)
   * - 3. next_token_logprobs(...)      # compute logprobs
   * - 4. worker.loss_fn(data_pack, logprobs)  # compute the RL loss
   * - 5. loss.backward()               # backpropagation
   * - 6. _accumulate_main_gradients()  # accumulate gradients into the master
   * - → gradient sync:
   * - a. _sync_data_parallel_gradients()
   * - b. _sync_tensor_parallel_replicated_gradients()
   * - → gradient clipping
   * - → optimizer.step()                # AdamW step
   * - → global_step += 1

Architecture Components Covered in This Section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Component
     - Responsibility / description
   * - Optimizer (AdamWFP32Master / AdamW8bit)
     - FP32 master bucket shards; 8-bit quantized optimizer state; gradient accumulation (_accumulate_main_gradients); learning-rate scheduling (warmup + linear/cosine decay)
   * - Activation Checkpointing
     - checkpoint_layer (standard layers); checkpoint_routed_moe_layer (MoE layers, routing is not checkpointed)
   * - Packing / Padding
     - the unified response_layout() abstraction
   * - Checkpoint I/O (common.py + io.py)
     - HF-compatible safetensors sharding; TP-aware gather/scatter; role-based on-demand loading
   * - Policy Sync (policy_sync.py)
     - NCCL relay architecture; greedy load balancing; chunked transfer
