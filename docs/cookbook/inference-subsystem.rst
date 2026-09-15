Inference Subsystem
===================

.. admonition:: Goal

   Understand the three core technologies of the AReno inference subsystem — Continuous Batching, CUDA Graph decode acceleration, and logprobs computation with sampling strategies.

.. admonition:: Prerequisites

   Online RL Loop (Online RL core loop), GSPO.

.. admonition:: Outcome

   Understand the scheduling mechanism behind rollout inference, the performance optimization techniques, and RL training's special requirements on logprobs and sampling.

In online RL, rollout (generating completions with the current policy) is one of the most time-consuming stages of the training loop. AReno's inference subsystem is designed specifically for the RL scenario: it must deliver **high throughput** while **computing the logprob of every token precisely**, and it must support the **sampling strategies** that are unique to RL training (temperature, top-k, top-p).

This chapter dives into ``InferenceManager`` (``areno/engine/inference.py``, 1107 lines) and its key dependency modules.

1 Continuous Batching: Handling Sequences of Unequal Length
---------------------------------------------------------------

Problem
~~~~~~~~~~

During rollout, prompts differ in length and in generation speed:
- prompt_1 is short (100 tokens), generates fast (stops after 50 tokens)
- prompt_2 is long (500 tokens), generates slowly (still running)

If you wait for all prompts to finish before processing the next batch → the GPU sits idle waiting, and throughput is extremely low.

Solution: Continuous Batching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AReno's ``InferenceManager`` implements continuous batching: **a finished sequence is immediately evicted, a new prompt is immediately admitted, and the GPU stays fully loaded.**

The core data structure is the **PagedAttention KV cache** (managed in blocks), backed by a **block table** that tracks the KV-block positions of every sequence.

Key Components
~~~~~~~~~~~~~~~~~~

InferCacheSpec: Cache Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # areno/engine/inference.py:54-62
   @dataclass(slots=True)
   class InferCacheSpec:
       max_running_seqs: int     # max number of sequences running concurrently
       max_cache_len: int        # max cache length per sequence
       num_blocks: int           # total number of KV cache blocks
       block_size: int           # block size (number of tokens)
       max_blocks_per_seq: int   # max blocks per sequence

These parameters determine how many sequences can be processed at once and how many tokens each sequence can generate at most. The cache is **allocated once and reused across rollouts** — a new rollout only resets the contents, without reallocating.

_init_infer_cache: Initialize or Reuse the Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # areno/engine/inference.py:131-197
   def _init_infer_cache(self, spec: InferCacheSpec) -> None:

Key logic:
1. **Reuse path**: if every dimension of the new ``InferCacheSpec`` fits within the existing cache → just reset the KV contents + refresh the weights
2. **Reallocation path**: if the cache is not large enough → free the old CUDA Graph → reallocate the KV cache → re-prepare the infer weights → re-capture the CUDA Graph

Cache reuse is a **critical performance optimization**: every step of RL training does a rollout, and reallocating the cache and recapturing the CUDA Graph every time would be extremely expensive.

The Inference Main Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core method ``infer_rollout()`` of ``InferenceManager`` manages the whole rollout process:

.. list-table::
   :widths: 15 85

   * - 1
     - Initialize the inference cache (_init_infer_cache)
   * - 2
     - Add new prompts to the running pool (prefill → compute the first token)
   * - 3
     - Enter the decode loop:
   * - 3a
     - Evict finished sequences (based on stop_token / EOS / max_new_tokens)
   * - 3b
     - Admit new prompts (if there are still unprocessed prompts)
   * - 3c
     - Run one decode step for all active sequences (accelerated with CUDA Graph)
   * - 3d
     - Sample tokens for the current step
   * - 4
     - Return the tokens + logprobs of all completions

Prefill vs Decode
~~~~~~~~~~~~~~~~~~~~~

- **Prefill**: computes all of a prompt's tokens in one pass (in parallel), filling the KV cache. There is no sampling in this phase — it is a pure forward pass.
- **Decode**: generates tokens autoregressively, one at a time, computing only 1 new token per step. Sampling happens at every step in this phase.

AReno's design: prefill and decode **share the same KV cache** — prefill writes KV in bulk, while decode appends token by token.

2 CUDA Graph Decoding: Eliminating Kernel Launch Overhead
------------------------------------------------------------

Problem
~~~~~~~~~~

Autoregressive decoding produces only 1 token per step, yet requires a full model forward pass (dozens of decoder layers × multiple kernel launches per layer). Each kernel launch has a fixed overhead (~5-10μs), which on the GPU is the CPU→GPU scheduling latency.

For a single 1-token decode step, **the ratio of kernel launch overhead to total compute time can be as high as 30-50%**.

Solution: CUDA Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA Graph lets you **pre-record a whole sequence of kernels** and later execute all of them with a single GPU-side call (replay). The CPU no longer launches kernels one by one — the GPU executes the recorded kernel dependencies on its own.

AReno implements ``DecodeGraph`` (69 lines) in ``areno/engine/runtime/decode_graph.py``.

Bucketing
~~~~~~~~~~~~~~

Because a CUDA Graph has a fixed shape (the ``batch_size`` recorded at capture time is the same at replay time), while the number of active sequences varies under continuous batching, AReno uses **bucket tiers**:

.. code-block:: python

   # areno/engine/runtime/decode_graph.py:19-25
   def bucket_for(batch_size: int, buckets: list[int]) -> int:
       """Return the smallest configured bucket that covers `batch_size`."""
       for bucket in buckets:
           if batch_size <= bucket:
               return bucket
       return batch_size

For example, with ``buckets=[1,2,4,8,16,32,64]``:
- 3 active sequences → use the bucket=4 graph
- 15 active sequences → use the bucket=16 graph
- The shortfall is filled with a **scratch block** (a block that is never allocated to a real sequence)

DecodeGraph Static Buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/engine/runtime/decode_graph.py:72-105
   class DecodeGraph:
       def __init__(self, model, bucket, max_blocks_per_seq, scratch_block, device):
           # addresses of these tensors must not change after the graph is captured
           self.input_ids = torch.zeros((1, bucket), device=device, dtype=torch.long)
           self.position_ids = torch.zeros((1, bucket), device=device, dtype=torch.long)
           self.cache_seqlens = torch.zeros(bucket, device=device, dtype=torch.int32)
           self.block_table = torch.full((bucket, max_blocks_per_seq), scratch_block, ...)
           self.meta = InferMeta(mode="decode", ...)
           self.graph = torch.cuda.CUDAGraph()

**Key constraint**: capturing a CUDA Graph records the memory addresses of the tensors, and replaying must use the same memory. So ``DecodeGraph`` keeps these static buffers and, before each replay, **copies the current step's dynamic data into them**, then replays.

Full Flow
~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 85

   * - 1
     - warmup (3 eager decode passes): stabilize the CUDA allocator and measure memory usage
   * - 2
     - Check that every TP rank has enough memory (has_graph_capture_memory, 20% safety margin)
   * - 3
     - capture: run one model forward inside the torch.cuda.graph(self.graph) context
   * - 4
     - replay: copy the dynamic data into the static buffers → self.graph.replay()

Why It Matters for RL Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In RL training, every training step needs a rollout first. CUDA Graph accelerates decode → rollouts are faster → more training steps fit in the same time → the model converges faster. This is one important reason AReno's single-node performance beats setups that rely on an external inference engine (SGLang/vLLM).

3 Logprobs: The Mathematical Foundation of RL
-------------------------------------------------

Why Does RL Need Logprobs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall the GSPO loss formula from GSPO:

.. math::

   r_{\text{seq}} = \exp\left(\frac{1}{N}\sum_t (\log\pi_\theta(a_t) - \log\pi_{\text{old}}(a_t))\right)

Here :math:`\log\pi_\theta(a_t)` and :math:`\log\pi_{\text{old}}(a_t)` are the **log probabilities of each token**. The importance ratio computation depends entirely on logprobs. Without logprobs there is no RL training.

AReno's logprobs computation is implemented in ``areno/engine/runtime/logprobs.py`` (271 lines).

Computing It Efficiently Under TP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under tensor parallelism (TP), the vocabulary is sharded across ranks along the last dimension. The traditional approach:
1. Each rank holds a ``vocab_size/tp_size`` slice of logits
2. All-gather to assemble the full logits
3. Run log_softmax
4. Extract the logprob of the target token

**Problem**: all-gathering the full logits costs too much communication (batch × seqlen × vocab_size × 4 bytes).

**What AReno does:**

.. code-block:: python

   # areno/engine/runtime/logprobs.py:170-248
   # vocab_parallel_selected_logprobs(logits_shard, labels)

It only needs to communicate **3 scalars** per token instead of the whole vocabulary's logits:

1. **max**: each rank's local logits_shard max → all-reduce(max) → global max
2. **exp_sum**: each rank computes ``sum(exp(logits_shard - global_max))`` → all-reduce(sum) → global sum
3. **selected_logit**: whichever rank holds the target token, take the corresponding logit value from that rank → broadcast

The final logprob is ``selected_logit - global_max - log(global_exp_sum)``.

This implementation dramatically reduces TP communication.

Implementing the Two Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AReno supports two batch layouts:

**Padded (pad-aligned)**:

.. code-block:: python

   # areno/engine/runtime/logprobs.py:24-41
   def next_token_logprobs(logits_shard, tokens, chunk_size=256):
       # standard (batch, seqlen, vocab_shard) shape, processed chunk by chunk

**Packed (variable-length concatenation)**:

.. code-block:: python

   # areno/engine/runtime/logprobs.py:44-78
   def packed_next_token_logprobs(logits_shard, tokens, cu_seqlens, chunk_size=256):
       # variable-length sequences concatenated into one long sequence;
       # cu_seqlens marks each sequence boundary

Packed layout saves the memory and compute wasted on padding, but needs an extra ``cu_seqlens`` (cumulative sequence lengths) to mark the boundaries.

Gemma4 Special Case: Deferred LM Head
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/engine/runtime/logprobs.py:81-110
   def packed_next_token_logprobs_from_hidden(
       hidden_states, tokens, cu_seqlens, lm_head, *, logit_softcap=None, chunk_size=256
   ):

For multimodal models (e.g. Gemma4), the LM head's ``[tokens, vocab]`` logits can take several GB of memory. AReno uses activation checkpointing to **compute-and-release** logits per chunk: the forward pass only saves hidden_states, and the backward pass recomputes logits → peak memory drops significantly.

4 Sampling Strategies
---------------------------

The last stage of rollout: **selecting the next token** from the model's output logits.

AReno's sampling implementation lives in ``areno/engine/data/sampling.py`` (291 lines) and has to handle vocabulary sharding under TP.

Two Sampling Paths
~~~~~~~~~~~~~~~~~~~~~~~~

Greedy Sampling (No Need for the Full Logits)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # areno/engine/data/sampling.py:26-70
   def _sample_greedy_sharded(logits_shard, vocab_size, tp_size, ...):

Under TP, each rank only holds a ``vocab_size/tp_size`` slice of the vocabulary:
1. Each rank finds the **local max** within its own shard
2. All-gather the ``(local max, local argmax)`` of every rank — only a few elements are exchanged
3. Select the global max among all the local maxes
4. Reconstruct the global token ID from rank + local_id

**No need to all-gather the full logits** — communication is minimal.

Stochastic Sampling (Needs the Full Logits)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # areno/engine/data/sampling.py:73-98
   def _sample_full_vocab(logits_shard, params, vocab_size, tp_size, device, ...):

top-k / top-p / temperature sampling needs the full probability distribution:
1. All-gather every rank's vocabulary shard → full logits
2. Apply temperature scaling
3. Apply top-k truncation and top-p truncation
4. Softmax → sample from the multinomial distribution

The core stochastic-sampling function ``_sample()`` (``areno/engine/data/sampling.py:100-200``) handles:
- **nan/inf safety**: automatically handles numeric anomalies
- **EOS suppression**: forcibly masks the EOS token during ``min_new_tokens``
- **suppress_token_ids**: user-specified forbidden tokens

Recording Logprobs
~~~~~~~~~~~~~~~~~~~~~~~~

During rollout, besides sampling tokens you also need to record **the logprob of the chosen token** (for the later importance ratio computation):

.. code-block:: python

   # areno/engine/data/sampling.py:202-210
   def _policy_token_logprobs(logits_shard, tokens):
       # call vocab_parallel_selected_logprobs to get the logprob

This logprob is :math:`\log\pi_{\text{old}}(a_t)` — the probability of that token under the policy at rollout time. The logprob obtained from the re-forward pass during training is :math:`\log\pi_\theta(a_t)`; subtracting the two gives the importance ratio.

The Architecture You've Covered So Far
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Component
     - Responsibility / description
   * - KV Cache (PagedAttention block management)
     - Prefill: writes prompt KV in bulk; Decode: appends KV token by token
   * - Continuous Batching scheduler
     - Finished sequences are evicted → new prompts admitted; block_table tracks the KV-block mapping of every sequence
   * - CUDA Graph decode acceleration
     - Bucket tiers (1,2,4,8,16,32,64...); static buffer → Capture → Replay; scratch block fills the shortfall
   * - Logprobs computation (vocab_parallel_selected_logprobs)
     - 3-scalar communication (max, exp_sum, selected_logit); supports both padded / packed layouts
   * - Sampling strategies
     - Greedy: local max + all-gather a small amount of data; Stochastic: all-gather full logits → top-k/top-p/temperature
