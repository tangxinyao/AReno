CUDA Acceleration
=================

.. admonition:: Goal

   Understand AReno's CUDA acceleration layer architecture — the full chain from C++/CUDA kernel source code to the Python calling interface.

.. admonition:: Prerequisites

   Inference Subsystem, Tensor Parallelism.

.. admonition:: Outcome

   Understand what each CUDA kernel does, what the Python wrappers are responsible for, and how the operator abstraction layer keeps upper-layer code unaware of the underlying implementation.

AReno's "zero external inference-backend dependencies" comes from its self-built CUDA kernel system. The ``areno/accel/`` directory contains 11 CUDA kernel files (``.cu``), 1 C++ bridge file, 15 Python wrapper files, and 3 Triton kernel files. Upper-layer engine and model code calls these kernels through a unified Python interface, without caring whether the underlying implementation is CUDA, Triton, or a PyTorch fallback.

1 CUDA Extension Architecture
--------------------------------

Three-Layer Structure
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Layer
     - Content
   * - Upper-level code (``areno/engine/layers/``)
     - Calls: ``areno_rmsnorm(x)``, ``areno_causal_attention(...)``
       ↓
   * - Python Wrapper layer (``areno/accel/*.py``)
     - 15 files: ``activation.py``, ``attention.py``, ``linear.py``...
       Responsibilities: parameter validation → call CUDA kernel → return result
       Decorator: ``@torch._dynamo.disable`` (keeps it out of torch.compile)
       ↓ via ``torch.utils.cpp_extension.load``
   * - C++ bridge layer (``areno/accel/csrc/extension.cpp``)
     - ``PYBIND11_MODULE`` → registers every CUDA kernel as a Python callable
       ↓
   * - CUDA Kernel layer (``areno/accel/csrc/*.cu``)
     - 11 .cu files + 1 .cuh header
       Responsibilities: high-performance GPU computation

Build Flow
~~~~~~~~~~

The build logic in ``setup.py``:

.. code-block:: python

   # setup.py
   # 1. detect the CUDA toolchain (nvcc, CUDA_HOME)
   # 2. collect all .cu and .cpp files under areno/accel/csrc/
   # 3. build with torch.utils.cpp_extension.CUDAExtension
   # 4. produce _areno_accel.so (or _areno_accel.pyd on Windows)
   # 5. install the compiled artifact into areno/accel/_areno_accel.*

The Python side lazy-loads it through ``areno/accel/_extension.py``:

.. code-block:: text

   # areno/accel/_extension.py
   # imports areno.accel._areno_accel on first call
   # compilation failure → RuntimeError with build instructions

2 Kernel Capability Matrix
-----------------------------

Full Inventory
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 8 45 22 11

   * - Kernel
     - File size
     - Function
     - Python wrapper
     - Call frequency
   * - ``attention.cu``
     - 43 KB
     - Attention computation (incl. causal mask, varlen, paged decode)
     - ``areno/accel/attention.py``
     - Every layer, every forward
   * - ``activation.cu``
     - 17 KB
     - Fused activation functions (SiLU, GeLU-tanh, Sigmoid, Softplus)
     - ``areno/accel/activations.py``
     - Every layer's FFN
   * - ``linear.cu``
     - 16 KB
     - Fused linear layers (standard + grouped per-expert)
     - ``areno/accel/linear.py``
     - Every layer, every projection
   * - ``normalization.cu``
     - 16 KB
     - RMSNorm + fused variants (scale/gate/silu)
     - ``areno/accel/normalization.py``
     - Each layer's norm
   * - ``conv.cu``
     - 21 KB
     - Depthwise-separable causal Conv1d + SiLU (batch/packed/decode)
     - ``areno/accel/conv.py``
     - Convolutional models
   * - ``embedding.cu``
     - 4 KB
     - TP-aware vocabulary embedding
     - ``areno/accel/embedding.py``
     - Input layer
   * - ``moe_permute.cu``
     - 15 KB
     - MoE token permutation (permute/unpermute)
     - ``areno/accel/moe.py``
     - Every MoE layer
   * - ``moe_align_kernel.cu``
     - 10 KB
     - MoE expert alignment (computes which tokens each expert handles)
     - ``areno/accel/routing.py``
     - Every MoE layer
   * - ``router.cu``
     - 7 KB
     - DeepSeek-style grouped top-k routing
     - ``areno/accel/router.py``
     - Every MoE layer
   * - ``topk.cu``
     - 10 KB
     - Softmax top-k selection (non-grouped routing)
     - ``areno/accel/topk.py``
     - Every MoE layer
   * - ``extension.cpp``
     - 12 KB
     - C++ bridge: PYBIND11_MODULE registration
     - ``areno/accel/_extension.py``
     - Init only

Deep Dive into Key Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~

attention.cu: Three Attention Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AReno's attention kernel supports three modes:

.. code-block:: text

   # areno/accel/attention.py
   # 1. Batch attention (training forward + rollout prefill)
   areno_causal_attention(query, key, value, ...)
   → supports: causal mask, GQA (KV head broadcast), flash-attn fallback

   # 2. Varlen attention (packed sequences)
   areno_varlen_causal_attention(query, key, value, cu_seqlens, ...)
   → supports: block-diagonal mask, different lengths per sequence

   # 3. Paged attention (decode phase)
   areno_paged_causal_attention_decode(query, key_cache, value_cache, block_table, ...)
   → supports: paged KV cache, scratch block filling, CUDA graph replay

**What makes decode attention special**: each step computes attention of a single query token against all KV tokens. q has length 1; k and v are read from the paged cache indexed by block_table.

linear.cu: Fused Linear Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A standard PyTorch ``F.linear()`` is three kernel launches (matmul + bias add + possible activation). AReno's ``areno_linear()`` fuses them into a single launch, reducing kernel overhead.

.. code-block:: text

   # areno/accel/linear.py
   areno_linear(x, weight, bias=None)
   → TP-aware: input goes through copy_to_tensor_parallel_region (all-reduce on backward)

   areno_grouped_linear(x, weight_3d, ...)
   → used for MoE expert MLP: weight[expert_idx, :, :] 3D weights
   → batched gemm grouped by expert

normalization.cu: Fused RMSNorm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   # areno/accel/normalization.py
   areno_rmsnorm(x, weight, eps)
   → standard RMSNorm

   areno_rmsnorm_silu_gate(x, weight, eps)
   → RMSNorm + SiLU gate: output = x * SiLU(norm(x))
   → used for the FFN gate projection, saves one round trip

   areno_optional_scale_rmsnorm(x, weight, scale, eps)
   → optional scale parameter (needed by some model families)

Triton Kernel Supplement
~~~~~~~~~~~~~~~~~~~~~~~~

For scenarios the CUDA kernels cannot easily cover, AReno also uses Triton (Python-native GPU programming) to implement some kernels:

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - Kernel
     - File
     - Use
   * - Fused MoE Experts
     - ``areno/accel/kernels/fused_moe.py``
     - Batched expert MLP (supports several MoE configurations)
   * - Group RMSNorm + Gate
     - ``areno/accel/kernels/group_rmsnorm.py``
     - Grouped RMSNorm with sigmoid gate
   * - Segmented Linear Attention
     - ``areno/accel/kernels/seg_la.py``
     - KDA-style block linear attention
   * - KDA Flash Attention
     - ``areno/accel/kernels/kda_fla/``
     - Efficient implementation of Bailing KDA

Importing Triton is optional — if Triton is not installed, the related kernels are unavailable and the engine falls back to pure PyTorch implementations.

3 Operator Abstraction Layer
-------------------------------

ops.py: The Unified Facade
~~~~~~~~~~~~~~~~~~~~~~~~~~

``areno/accel/ops.py`` (89 lines) is the unified entry point through which upper-layer code accesses the acceleration kernels:

.. code-block:: python

   # areno/accel/ops.py
   from areno.accel.activations import areno_gelu_tanh_and_mul, areno_silu_and_mul
   from areno.accel.attention import areno_causal_attention, areno_varlen_causal_attention
   from areno.accel.kernels.fused_moe import fused_experts as areno_fused_experts
   # ...

Upper-layer code only needs ``from areno.accel.ops import areno_rmsnorm`` — it never knows whether the underlying implementation is CUDA or Triton.

Route Dispatch
~~~~~~~~~~~~~~

.. code-block:: python

   # areno/accel/ops.py:59-69
   def can_use_cuda_kernel(tensor, name, *, allow_sm121=False):
       """Non-CUDA tensors return False directly, triggering a PyTorch fallback"""
       if not tensor.is_cuda:
           return False
       return True

Upper-layer code typically uses it like this:

.. code-block:: python

   if can_use_cuda_kernel(hidden_states, "rmsnorm"):
       return areno_rmsnorm(hidden_states, weight, eps)  # CUDA fused kernel
   else:
       return torch.nn.functional.rms_norm(hidden_states, ...)  # PyTorch fallback

This pattern guarantees:
- **CUDA GPU**: goes through the fused kernel for peak performance
- **Apple Silicon (MLX)**: automatically falls back to PyTorch
- **CPU tests**: everything falls back under ``ARENO_BUILD_EXT=0``

Diagnostics Helpers
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/accel/ops.py
   log_once("attention_backend", "using flash_attention_2 backend")
   # printed only once, never spams the training log

   warn_once("moe_fallback", "fused MoE not available, using PyTorch loop")
   # warns only the first time it is hit

AReno Acceleration Layer vs External Inference Engines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Comparison axis
     - AReno (accel/)
     - vLLM / SGLang
   * - Number of kernels
     - 11 CUDA + 4 Triton
     - 100+ CUDA
   * - Coverage
     - Hot paths needed for RL training
     - Complete inference optimization
   * - Inference-specific optimizations
     - Decode CUDA Graph, Paged KV
     - Prefill/Decode split, quantized KV
   * - Training support
     - ✓ (RL-specialized)
     - ✗ (inference only)
   * - Dependencies
     - Zero external inference backends
     - Requires specific versions
   * - Startup time
     - Seconds
     - Usually needs preloading

AReno's strategy: **don't build a general-purpose inference engine; only optimize the kernels RL post-training requires.** A small, focused set of kernels covers 95%+ of each forward pass's compute time; the remaining non-critical paths are left to native PyTorch. This keeps the code maintainable and cross-platform. At the same time, the in-house kernels replace TransformerEngine (NVIDIA's transformer kernel library), breaking free from its version constraints.
