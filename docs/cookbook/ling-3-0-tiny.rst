Ling-3.0-tiny
=============

.. note:: TL;DR

   The training model in this guide is Ling-3.0-tiny (7.9B MoE, 1.3B activated). You can run it fully on a DGX Spark (local, ~$4,699 one-time) or on an Alibaba Cloud GPU instance (cloud, pay-as-you-go at ~¥12-40/hour). Consumer GPUs can't run RL training, and MacBook and Windows are not covered in this guide.

Before you get started, we need to make two things clear: "what we're running" and "where we're running it." This chapter won't teach you how to install anything (that's for DGX Spark and Aliyun Cloud)—it helps you understand the relationship between the hardware and the model: **why RL training needs this kind of hardware, and why Ling-3.0-tiny is worthy of being this book's main-line model.**

1 Why This Model?
--------------------------

The criteria for choosing a main-line model are simple: **small enough to run complete RL training on a single consumer-grade to professional GPU; strong enough to contain the key architectural designs of modern LLMs.** Ling-3.0-tiny satisfies both.

"Small and strong" isn't the whole reason, though. Architecturally, Ling-3.0-tiny is a **MoE (Mixture of Experts) model**—and MoE is one of the defining directions of frontier LLMs today (Ling-3.0 itself, DeepSeek-V3, Qwen3-MoE, and others). Plenty of small Dense models would run RL just as cheaply; a main-line model has to teach you something the frontier actually uses, and MoE is that something. Training RL on Ling-3.0-tiny means training on the real architecture, not a simplified stand-in. It also means you'll meet MoE-specific RL effects firsthand—experts specializing toward different reasoning paths under reward pressure—instead of reading about them on a Dense model.

Concretely, Ling-3.0-tiny comes from inclusionAI's Ling series and is the "tiny" variant of Ling-3.0: through MoE it uses a "7.9B-parameter shell while activating only 1.3B parameters per inference," striking a good balance between performance and efficiency.


Two Key Architectural Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before diving into the code in later chapters, let's build intuition for two of Ling-3.0-tiny's signature designs. You don't need to fully understand them now, but remember what they do in this book.

MoE: An Expert Team
^^^^^^^^^^^^^^^^^^^

A traditional Dense model activates all parameters at every layer—like a company where everyone speaks at every meeting. MoE is different:

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Dimension
     - Traditional Dense model
     - Ling-3.0-tiny MoE
   * - Activation ratio
     - Each token activates 100% of parameters
     - Each token only activates ~16% of parameters
   * - Compute flow
     - input → [all 7.9B parameters are computing] → output
     - input → Router selects 8 experts → [only these 8 experts compute] → output
   * - Expert setup
     - —
     - 128 routed experts + 1 shared expert

Specifically for Ling-3.0-tiny:

- **128 routed experts**: each is a small FFN network; the ``Router`` selects the 8 most relevant ones based on token content
- **1 shared expert**: every token passes through it, ensuring basic capability is never lost
- **Actual compute**: equivalent to a 1.3B Dense model, but with the knowledge capacity of a 7.9B model

.. admonition:: What This Means for RL Training

   MoE has a unique advantage in RL. Different experts may specialize in different reasoning paths—some experts are good at math derivation, others at code generation. RL training reinforces this differentiation: high-reward paths → relevant experts get activated more → further optimize these experts' parameters → a positive loop. The Model Adapters chapter dives into the details of MoE's implementation in AReno.

KDA + MLA: Hybrid Attention
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Across Ling-3.0-tiny's 32 Transformer layers, the attention mechanism is not uniform—it uses a **KDA (Key-Decomposed Attention) + MLA (Multi-Head Latent Attention)** hybrid:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Item
     - Content
   * - Grouping of every 4 layers
     - Layer 1: KDA → Layer 2: KDA → Layer 3: KDA → Layer 4: MLA
   * - Composition of each group
     - 3 KDA layers (faster) + 1 MLA layer (saves more KV cache)
   * - Repetition pattern
     - Layer 5: KDA → … repeated 8 groups … Layer 32: MLA

The division of labor between the two attentions:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - KDA
     - MLA
   * - **Design goal**
     - Fast inference
     - Small KV cache footprint
   * - **Core idea**
     - Decompose Key into multiple components to reduce compute
     - Compress Key/Value into a low-dimensional latent space
   * - **KV cache**
     - Larger
     - Smaller (after compression)
   * - **Layer share**
     - 24 layers (75%)
     - 8 layers (25%)

.. admonition:: What This Means for RL Training

   🔑 In RL training, rollout (the model generating completions) and training (forward + backward + optimizer update) alternate. The size of the KV cache directly affects "how many rollouts can run at once"—the low KV-cache footprint of MLA layers lets AReno fit more concurrent inference requests into limited memory. KDA layers ensure inference speed across most layers. This is a design crafted specifically for the "inference-training alternation" scenario.

Thinking Mode
~~~~~~~~~~~~~

Ling-3.0-tiny natively supports **thinking mode**—controlled via the ``enable_thinking`` switch in the tokenizer's chat template:

.. code-block:: python

   # enable_thinking=True  → 模型先输出思考过程，再输出答案
   # enable_thinking=False → 模型直接输出答案

In AReno, both training and inference can control this switch:

.. code-block:: bash

   # 训练时关闭 thinking（适合数学题——直接算就行）
   areno train --ckpt ./models/Ling-3.0-tiny --disable-thinking ...

   # 推理时开启 thinking（适合复杂 agentic 任务——需要多步推理）
   areno serve --model-path ./models/Ling-3.0-tiny ...  # 默认 enable_thinking

Agentic RL will discuss in detail the value of thinking mode in multi-turn tool invocation—the model needs to plan "what information I need, which tool to call" during the thinking stage, and only then generate a tool call.

Memory Footprint
~~~~~~~~~~~~~~~~

AReno officially did a full memory profiling of Ling-3.0-tiny on DGX Spark (FP8 precision, 8K context):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Component
     - Memory usage
   * - Model weights (FP8)
     - ~7.4 GiB
   * - KV Cache (8K × n_samples)
     - ~0.5 GiB
   * - Optimizer States (FP32 master)
     - ~0.4 GiB
   * - Other (activations, temp buffers)
     - ~0.04 GiB
   * - **Peak total**
     - **~8.34 GiB**

A few key observations:

1. **Model weights dominate**: ~7.4 GiB is the fixed "admission ticket." FP8 precision is the key—with BF16 it would double to ~14.8 GiB.
2. **KV cache and optimizer states are both small**: because Ling-3.0-tiny has only 1.3B activated parameters, the intermediate states of backpropagation are tiny.
3. **8K context is "economy class"**: if you extend to a 32K context, the KV cache grows linearly to ~2 GiB. The training examples in this book all stay within 8K.
4. **DGX Spark's 128 GB unified memory is more than enough**: ~8.34 GiB only accounts for the GPU-portion usage; in fact, DGX Spark's CPU and GPU share the 128 GB memory pool, which means you can load other models for comparison experiments while training runs.

.. warning:: A Common Misconception

   ⚠️ "A 7.9B model should take up 7.9B × 2 bytes = 15.8 GB (BF16)." But Ling-3.0-tiny is an MoE model, and AReno's FP8 training compresses the weights to ~7.4 GiB while maintaining precision. Training Subsystem will explain the memory-management strategy for optimizer states (AdamW 8-bit vs FP32 master params).

2 Which Hardware?
-----------------

Two paths run the same AReno commands; which you pick depends on **how long** you plan to train. Setup details live in their own chapters—here we only decide.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Option
     - Hardware
     - Who it's for
   * - :doc:`dgx-spark`
     - NVIDIA GB10, 128 GB unified memory, ~$4,699 one-time
     - Long-term: you'll keep doing RL experiments for months or years
   * - :doc:`aliyun-cloud`
     - A10 (24 GB, ~¥12-15/h) or H20 (96 GB, ~¥35-40/h)
     - Short-term: learning this book's examples; larger A10 vs H20 only matters later if you outgrow the default config

Buy vs Rent
~~~~~~~~~~~~~

Assume you train 20 hours a week for 6 months (about 480 hours). Roughly:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Option
     - 6-month cost
     - Takes about
   * - **DGX Spark**
     - ~$4,699 one-time
     - 480 hours
   * - **Alibaba Cloud A10**
     - ¥5,760 ≈ $790
     - Same 480 hours
   * - **Alibaba Cloud H20**
     - ¥16,800 ≈ $2,300
     - Same 480 hours

**The rule of thumb**: if you're sure you'll keep doing LLM RL experiments long-term, DGX Spark wins on economics; if you're only learning this book's examples short-term, Alibaba Cloud's pay-as-you-go is more flexible. (For how to save on the cloud, see **Aliyun Cloud → Tips for Saving Money**.)

3 Unsupported Configs
---------------------

This section isn't meant to discourage you—it's meant to save your time: if you only have the configs below, forcing RL training will most likely hit a wall.

Consumer GPUs (8-12 GB)
~~~~~~~~~~~~~~~~~~~~~~~

The most common question: "My RTX 4060 Ti has 16 GB, or my RTX 4070 has 12 GB—can it run?"

**The answer: it can't run Ling-3.0-tiny's complete RL training.** Let's do the math:

**The memory bill for RL training (FP8, 8K context):**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Item
     - Usage
     - Notes
   * - Model weights (FP8)
     - ~7.4 GiB
     - ← It's occupied no matter what you do
   * - rollout KV Cache
     - ~0.5 GiB
     - ← Needed by every concurrently generated sequence
   * - optimizer States
     - ~0.4 GiB
     - ← AdamW's first and second moments
   * - Activations (forward + backward)
     - ~0.04 GiB
     - ← Activations are tiny (1.3B activated + recompute)
   * - **Minimum requirement**
     - **~8.34 GiB**
     - —

A 12 GB RTX 4070 has about 11 GB usable after system reservations. 8.34 GiB looks like it could fit, but in reality:

1. **This isn't a case of "barely fitting is fine."** CUDA's memory allocation isn't 100% efficient—fragmentation means actual usable memory < the theoretical value
2. **PyTorch's CUDA caching allocator reserves extra space**
3. **Once n_samples is increased, the KV cache grows linearly**—8 concurrent samples × 8K context = the KV cache takes ~4 GiB
4. **Some CUDA kernels don't support FP8 on consumer cards** (this requires Transformer Engine support on Ada Lovelace or newer architectures, and since AReno doesn't depend on TE, FP8 in its self-built kernels may be unavailable on some cards)

**What can you do if you only have a consumer card?**

- Switch to a smaller model (e.g., Qwen3-0.6B) and run it with the same AReno commands
- Run only SFT (no rollout needed; much less memory pressure)
- Squeeze things to the limit with ``--batch-size 1 --mini-bs 1 --n-samples 1``

But **this book won't cover these workarounds**—our goal is for you to understand the complete RL training pipeline, not to repeatedly debug OOM at the edge of your card's memory.

MacBook / Apple Silicon (MLX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AReno supports Apple Silicon (via the MLX backend), and the README has a complete Mac installation guide:

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .

But this book chooses not to cover the MLX path, for a simple reason:

1. **MLX and CUDA are two entirely different underlying implementations.** The architecture deep dives in this book (CUDA kernels, CUDA Graph decoding, tensor parallelism) have no equivalents on MLX—MLX is Apple's own, completely different acceleration framework.
2. **This book's organizing philosophy is "learn concepts from code."** On MLX you can't learn how continuous batching is scheduled or how CUDA Graph is captured—these are the core of engineering capability.
3. **Ling-3.0-tiny has no official MLX-format weights.** You'd have to convert them yourself, which is already beyond this book's scope.

If you only have a MacBook, you can first read this book's theoretical parts (the RL concepts and algorithm chapters, from RL in One Snippet through Agentic RL), then run the code parts with Colab or an Alibaba Cloud instance.

Windows / WSL2
~~~~~~~~~~~~~~~~

AReno officially supports WSL2 (Windows Subsystem for Linux 2) as a CUDA path. But this book doesn't cover WSL2 setup, because:

1. **WSL2's CUDA support depends on a complex chain: Windows driver → WSL2 kernel → CUDA Toolkit.** A version mismatch at any link leads to hard-to-debug bugs like "CUDA available but kernel launch failed".
2. **This book's installation steps (DGX Spark and Aliyun Cloud) assume native Linux.** Path mapping, network configuration, and file-system performance under WSL2 would need separate coverage.
3. **If you're familiar with WSL2 and CUDA environments, you can absolutely adapt it yourself.** The core AReno commands are identical under WSL2 and native Linux; you just need to get the driver and CUDA Toolkit installed.

.. admonition:: The Bottom Line

   This book assumes you are running on **native Ubuntu Linux (ARM64 or x86_64) + NVIDIA GPU + CUDA 12.4+**.

Chapter Summary
~~~~~~~~~~~~~~~~~~

Before you get to the installation steps, remember this set of correspondences:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Node
     - Notes
   * - Ling-3.0-tiny (7.9B MoE, 1.3B activated)
     - —
   * - Option A: DGX Spark (~$4,699)
     - Unified memory → zero copy; good for long-term use (see DGX Spark)
   * - Option B: Alibaba Cloud
     - A10 (24 GB, ~¥12-15/h) or H20 (96 GB, ~¥35-40/h); good for short-term (see Aliyun Cloud)
   * - Minimum memory for RL training: ~8.34 GiB (FP8, 8K context)
     - —

**Next step**: choose your hardware option.

- **Option A (DGX Spark)** → jump to **:doc:`dgx-spark`**, from unboxing to a working smoke test
- **Option B (Alibaba Cloud)** → jump to **:doc:`aliyun-cloud`**, from creating an instance to a working smoke test

The two chapters are independent—just read the one that matches. Once installation is done, both paths converge at **First Train**—understanding every detail of your first training.
