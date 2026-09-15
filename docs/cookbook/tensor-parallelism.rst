Tensor Parallelism
===================

.. admonition:: Goal

   Understand AReno's tensor parallelism (TP) mechanism — why it is needed, how it shards, and how inference and training differ.

.. admonition:: Prerequisites

   Inference Subsystem, Training Subsystem.

.. admonition:: Outcome

   Understand TP's communication primitives and sharding strategies, and know when and how to scale up training with ``--tp-size``.

Tensor parallelism (TP) is the core mechanism AReno uses for **single-node multi-GPU training**. When a model is too large to fit on a single GPU, TP shards the model's layers across GPUs by dimension, using a small amount of communication to preserve mathematical equivalence.

AReno's TP implementation lives in ``areno/engine/parallel/``:
- ``context.py`` (268 lines): TP topology management and process-group initialization
- ``collectives.py`` (240 lines): autograd-aware communication primitives

1 Why Do We Need TP?
--------------------------

The Limits of a Single GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take Ling-3.0-tiny (7.9B parameters) as an example:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Component
     - BF16 memory
     - Description
   * - Model weights
     - ~16 GB
     - 7.9B × 2 bytes
   * - Optimizer state (AdamW)
     - ~32 GB
     - FP32 master + exp_avg + exp_avg_sq
   * - Activations (batch=4, seq=2048)
     - ~8 GB
     - Depends on batch size and seqlen
   * - KV Cache (rollout)
     - ~2 GB
     - Depends on max_running_prompts and max_cache_len
   * - **Total (training)**
     - **~56 GB**
     - Exceeds a single card's 24GB (A10) or just fits in 96GB (H20)

When the model is bigger (70B+), the batch size is larger, or the context is longer, the memory requirement far exceeds a single GPU's capacity.

The Core Idea of TP
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   single GPU:  [GPU 0 owns the complete weights of every layer]
   two GPUs:    [GPU 0 owns the left half of each layer's weights] [GPU 1 owns the right half]

Shard the model layer's weight matrices **by column or by row** across GPUs; each GPU computes its own part in the forward pass and combines the results through communication when needed. TP's communication volume is O(batch × seqlen × hidden), while DP (data parallelism) is O(number of parameters). When hidden is not large, TP's communication overhead is acceptable.

TP vs DP
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - Dimension
     - Tensor parallelism (TP)
     - Data parallelism (DP)
   * - What is sharded
     - The model layers (weight matrices)
     - The data (batch)
   * - What is communicated
     - Activations (forward) + gradients (backward)
     - Gradients (all-reduce)
   * - Communication volume
     - O(B × S × H)
     - O(number of parameters)
   * - Use case
     - Single node, multiple GPUs (high-bandwidth NVLink)
     - Multiple nodes, multiple GPUs (the network need not have extremely high bandwidth)
   * - AReno parameter
     - ``--tp-size N``
     - ``--dp-size N``

AReno supports **a TP + DP mix**: first use TP to place the model on the N GPUs of a single node, then use DP to scale out to multiple nodes.

2 TP Sharding Strategy
---------------------------

What AReno implements is **Megatron-LM-style TP**, whose core is sharding inside the Transformer layer. A Transformer decoder layer has two main parts:

TP Sharding of the Attention Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Original Attention
     - TP sharding (along the head dimension)
   * - Q = X · W_Q (hidden → num_heads × head_dim)
     - W_Q column-sharded → each rank holds the Q projection of a subset of the heads
   * - K = X · W_K (hidden → num_kv_heads × head_dim)
     - W_K column-sharded → each rank holds the K projection of a subset of the heads
   * - V = X · W_V (hidden → num_kv_heads × head_dim)
     - W_V column-sharded → each rank holds the V projection of a subset of the heads
   * - O = Attention(Q, K, V) · W_O (num_heads × head_dim → hidden)
     - W_O row-sharded → each rank's O projection outputs part of hidden; finally all-reduce combines the W_O outputs

TP Sharding of the FFN Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Original FFN
     - TP sharding (along the intermediate dimension)
   * - H = activation(X · W_up) * (X · W_gate)   # or SiLU
     - W_up column-sharded → each rank's intermediate dimension is 1/tp_size of the original; W_gate column-sharded
   * - O = H · W_down
     - W_down row-sharded → each rank outputs part of hidden; finally all-reduce combines

Communication Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AReno implements all the required communication operations in ``areno/engine/parallel/collectives.py``; each one is autograd-aware (with its own forward and backward):

.. list-table::
   :header-rows: 1
   :widths: 35 25 25 15

   * - Primitive
     - Forward
     - Backward
     - Use
   * - ``all_reduce``
     - sum across TP ranks
     - identity (the gradient was already summed by the previous layer)
     - combines row-parallel outputs
   * - ``copy_to_tensor_parallel_region``
     - identity
     - all-reduce
     - entering a column-parallel layer
   * - ``scatter_to_sequence_parallel_region``
     - shard along the seq dimension
     - all-gather along the seq dimension
     - sequence parallelism
   * - ``gather_from_sequence_parallel_region``
     - all-gather along the seq dimension
     - reduce-scatter along the seq dimension
     - sequence-parallel exit
   * - ``reduce_scatter_to_sequence_parallel_region``
     - reduce-scatter along the seq dimension
     - all-gather along the seq dimension
     - sequence-parallel variant
   * - ``all_gather_last_dim``
     - gather along the vocab/hidden dimension
     - —
     - vocab/gather

Source Example: The autograd Implementation of all_reduce
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/engine/parallel/collectives.py
   class _AllReduceSum(torch.autograd.Function):
       @staticmethod
       def forward(ctx, x, group):
           # forward: sum the activations across the TP ranks
           ctx.group = group
           y = x.clone()
           dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
           return y

       @staticmethod
       def backward(ctx, grad_output):
           # backward: the gradient is already the summed one, pass it through (no all-reduce needed)
           return grad_output, None

Key design: **during backward, the gradient has already gone through the forward all-reduce, so identity suffices**. This is the core efficiency guarantee of Megatron-LM-style TP.

Sequence Parallelism (SP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides sharding along the model-layer dimension, AReno also supports sequence parallelism: the LayerNorm / Dropout inputs are sharded along the sequence dimension, and the full sequence is restored between the attention and FFN blocks via gather/scatter.

.. list-table::
   :widths: 100

   * - SP data flow:
   * - [LayerNorm input: TP-sharded along the seq dimension]
   * - → gather_from_sequence_parallel_region → [full sequence]
   * - → Attention → [full sequence]
   * - → reduce_scatter_to_sequence_parallel_region → [TP-sharded along the seq dimension]
   * - → [FFN input: TP-sharded along the seq dimension]
   * - → gather_from_sequence_parallel_region → [full sequence]
   * - → FFN → [full sequence]
   * - → reduce_scatter_to_sequence_parallel_region → [TP-sharded along the seq dimension]

The benefit of SP: the LayerNorm and Dropout activations are also sharded across the TP ranks, saving about ``1/tp_size`` of the activation memory.

TPContext: TP Topology Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/engine/parallel/context.py
   @dataclass
   class TPContext:
       rank: int              # current rank within the TP group
       world_size: int        # TP group size
       device: torch.device   # device of the current rank
       group: dist.ProcessGroup  # TP's NCCL communicator
       dp_rank: int           # rank within the DP group
       dp_size: int           # DP group size
       dp_group: dist.ProcessGroup
       # Policy sync related
       role: str              # "train" | "rollout"
       policy_publisher_groups: tuple  # NCCL relay groups for weight sync
       policy_source_ranks: tuple
       policy_bridge_ranks: tuple

``get_tp_context()`` is the global access point; all code that needs to be aware of the TP environment gets rank/world_size/group through it.

3 TP Differences Between Inference and Training
----------------------------------------------------

TP During Inference
~~~~~~~~~~~~~~~~~~~~~~~~~

During inference (rollout), the main considerations are:

1. **Vocabulary sharding**: under TP each rank holds ``vocab_size/tp_size`` of the logits. Sampling needs special handling (see Section 4 of Inference Subsystem):
   - Greedy sampling: each rank finds its local max → all-gather a small amount of data
   - Stochastic sampling: all-gather the full logits

2. **Logprobs computation**: done with the 3-scalar communication (see Section 3 of Inference Subsystem)

3. **Weights only need the forward pass**: no gradients, no optimizer state

TP During Training
~~~~~~~~~~~~~~~~~~~~~~~~

Training additionally needs:

1. **Gradient synchronization**: after each microbatch's backward pass, the gradients need to be synchronized
   - ``_sync_data_parallel_gradients()``: all-reduce the gradients across DP
   - ``_sync_tensor_parallel_replicated_gradients()``: all-reduce the replicated parameters (e.g. embeddings) across TP

2. **Optimizer state sharding**: the FP32 master weights are also sharded across the TP ranks; each rank only holds the optimizer state for its own slice of parameters

3. **Checkpoint gather/scatter**:
   - When saving: each TP rank's weight shards must be gathered into a full tensor before writing
   - When loading: the full tensor is scattered to each TP rank

TP Considerations in Policy Sync
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using PPO's separated train+rollout architecture, weight synchronization must span the TP ranks across partitions:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Train partition (TP=2)
     - Rollout partition (TP=2)
   * - [GPU 0] [GPU 1] →
     - [GPU 2] [GPU 3]

AReno's ``transfer_policy_weights()`` works via a **NCCL relay**: the train rank first broadcasts to a bridge rank (making sure it is not on the same physical GPU), and the bridge rank then forwards to the rollout TP group.

Key Parameters at a Glance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Parameter
     - Description
     - When to use
   * - ``--tp-size N``
     - Tensor-parallel degree (N GPUs form one TP group)
     - When the model exceeds a single GPU's memory
   * - ``--dp-size M``
     - Data-parallel degree (M TP groups make up a DP)
     - When scaling out to multiple nodes
   * - ``--world-size N*M``
     - Total number of GPUs
     - TP × DP
   * - ``--sequence-parallel``
     - Enables sequence parallelism
     - To save activation memory

Recommended Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 55 45

   * - Ling-3.0-tiny (7.9B)
     - single A10 (24GB) → tp-size=1 (fits, no TP needed)
   * - Ling-3.0-tiny (7.9B) + large batch
     - single H20 (96GB) → tp-size=1
   * - 70B model
     - 4× A10 (24GB) → tp-size=4; 2× H20 (96GB) → tp-size=2
   * - 200B+ model
     - 8× H20 (96GB) → tp-size=8
