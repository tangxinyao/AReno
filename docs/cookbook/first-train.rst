First Train
===========

.. admonition:: Objective

   Use one ``areno train`` command to run the complete RL training pipeline for Ling-3.0-tiny, then read each line of the output section by section to understand what it means.

.. admonition:: Prerequisites

   You've completed the environment setup in DGX Spark or Aliyun Cloud.

.. admonition:: Outcome

   One command takes you from loading the model to completing a single gradient update, confirming the whole pipeline works.

1 The Two Levels of "First Training"
-----------------------------------------

In practice, "first training" proceeds in two steps, each with a different purpose:

.. list-table::
   :header-rows: 1
   :widths: 18 34 42 12 14

   * - Stage
     - Command
     - Purpose
     - Duration
     - Updates Weights?
   * - **Smoke Test**
     - ``areno train --smoke-infer ...``
     - Verifies model loading + KV cache allocation + CUDA graph initialization without OOM
     - ~30 seconds
     - ✗
   * - **Smoke Train**
     - ``areno train --smoke-train ...``
     - On top of the previous step, runs one synthetic backward + optimizer step
     - ~60 seconds
     - ✗ (dummy weights)
   * - **Real minimal training**
     - ``areno train --algo gspo ...`` (without the smoke flag)
     - Real rollout + reward + backward + optimizer step
     - a few minutes
     - ✓

**Recommended path**: first run ``--smoke-infer`` → once it's green, run ``--smoke-train`` → once that's green, run real training.

.. warning:: Pitfall

   The rest of this chapter gives a **fully reproducible real training command** (with ``--adam-8bit``) that has been validated on DGX Spark (10 epochs / 20 steps end normally, no OOM). If you just want to get it running, you can jump straight to 5.1.4 Complete Real Training Command and copy it.

1.1 Smoke Infer: Can the Model "Stand"?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--smoke-infer`` does three things:

1. **Loads the model weights** onto GPU memory
2. **Allocates the rollout KV cache** (reserving a memory pool for later decoding)
3. **Records a CUDA Graph** (pre-recording the autoregressive decoding kernel sequence as a graph to reduce kernel launch overhead)

It does **not generate tokens, compute rewards, or do backward**. Its only purpose is to answer one question: on the current hardware, does the model + KV cache fit?

Example of a successful output (DGX Spark, Ling-3.0-tiny, single GPU):

.. code-block:: text

   AReno smoke infer ok: tp_size=1, batch_size=1, n_samples=1, mini_bs=1,
   max_running_prompts=1, adam_8bit=False, drop_rollout_state=True, peak_mem_frac=0.1996

- ``peak_mem_frac=0.1996``: peak memory usage is about 20%, so at this minimal config the 7.9B model + KV cache is very comfortable.
- ``drop_rollout_state=True``: after rollout completes, the KV cache state is released and memory is given back to the training phase.

1.2 Smoke Train: Can backward Run?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--smoke-train`` goes one step further: it uses **synthetic data** (dummy tokens, dummy advantages) to run one full forward + backward + optimizer.step(), verifying the training path has no bugs.

It does not load a real dataset and needs no ``--reward-fn-path``. A successful output looks like:

.. code-block:: text

   AReno smoke train ok: tp_size=1, ... peak_mem_frac=0.XXXX

1.3 Why Smoke Before Real Training?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In real training, backward consumes a lot of extra memory:
- **Model weights** (already loaded)
- **KV cache** (during rollout)
- **Activations** (saved during forward for backprop)
- **Optimizer state** (Adam's m and v, typically 2x the parameter count)
- **Gradients** (the same size as the model parameters)

If smoke infer's ``peak_mem_frac`` is already high (say 0.8), real training will almost certainly OOM. In that case you need to tune parameters first (``--adam-8bit``, ``--drop-rollout-state``, ``--max-running-prompts 1``, etc.), not go straight to real training.

.. admonition:: Field Lesson (from DGX Spark)

   An 8B model under FP32 Adam has an optimizer state of about 95 GB. On the GB10 with 121 GB of unified memory, the first real training run without ``--adam-8bit`` hit a direct ``torch.OutOfMemoryError`` at ``optimizer.step()`` (it tried to allocate a 64 MiB bucket after already using 110.63 GiB). Adding ``--adam-8bit`` compressed the optimizer state to ~12 GB and it passed. **For 7B+ models, ``--adam-8bit`` is not an optimization; it's a requirement.**

1.4 The Complete Real Training Command (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command has been fully validated on DGX Spark (NVIDIA GB10, 128 GB unified memory):

.. code-block:: bash

   cd ~/AReno
   export CPATH=/home/tangxinyao/pyhdrs:/home/tangxinyao/pyhdrs/aarch64-linux-gnu
   export CUDA_HOME=/usr/local/cuda
   export TORCH_CUDA_ARCH_LIST="12.1"
   export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
   export MAX_JOBS=8

   areno train \
     --ckpt /home/tangxinyao/.cache/modelscope/models/inclusionAI--Ling-3.0-tiny/snapshots/master \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --attn-backend flash \
     --adam-8bit \
     --batch-size 1 \
     --max-running-prompts 1 \
     --max-prompt-tokens 64 \
     --max-new-tokens 16 \
     --temperature 1 \
     --tp-size 1 \
     --world-size 1

.. admonition:: Important

   The command above does **not set ``--save-path``**, so no checkpoint will be saved after training. If you need to keep the fine-tuned model weights, be sure to add ``--save-path ./my_checkpoints``.

**Measured results (DGX Spark, 2026-08-21)**:
- 10 epochs / 20 steps (2 GSM8K samples x batch_size=1, 2 steps per epoch)
- ~40-60 seconds per step (rollout dominates; train ~7 seconds)
- Exits normally (``epoch=9 stage=epoch_end``), no OOM, no crash
- ``grad_norm`` is basically 0 (too little data, all rewards are 0, the model learns nothing—but the pipeline works)

2 Breaking Down the Smoke Test Command Parameter by Parameter
-----------------------------------------------------------------

Take a standard smoke-infer command as an example (parameters basically the same as the real training command in 5.1.4):

.. code-block:: bash

   areno train \
     --ckpt ./models/Ling-3.0-tiny \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --algo gspo \
     --tp-size 1 --world-size 1 \
     --batch-size 1

What each parameter means and why it's set this way:

Model and Data
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 13 30 38 45

   * - Parameter
     - Default
     - This Setting
     - Purpose
     - Why Set This Way
   * - ``--ckpt``
     - None (required)
     - ``./models/Ling-3.0-tiny``
     - Model weight directory (contains ``config.json`` + safetensors)
     - Local path, or use ``--model-hub ms`` to pull from ModelScope
   * - ``--model-hub``
     - ``modelscope``
     - Uses default
     - Remote model/dataset repository
     - ModelScope is faster in China; use ``hf`` overseas
   * - ``--dataset-path``
     - None (required)
     - ``gsm8k:main``
     - Dataset identifier of the form ``repo:config:split`` or a local path
     - ``gsm8k:main`` = the ``main`` config of the HuggingFace ``gsm8k`` dataset
   * - ``--dataset-loader-fn``
     - None
     - ``examples/math/dataset_loader.py``
     - Data preprocessing script that converts raw fields into the ``prompt`` + ``solutions`` format
     - GSM8K's ``question``/``answer`` fields need conversion; see the Dataset Loader Contract below

Algorithm and Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 13 28 25 70

   * - Parameter
     - Default
     - This Setting
     - Purpose
     - Why Set This Way
   * - ``--algo``
     - ``gspo``
     - ``gspo`` (default)
     - Selects the training algorithm
     - GSPO = Group-Sampled Policy Optimization, a sequence-level PPO variant, the friendliest for getting started. Changing only one ``--algo`` switches to ``grpo``, ``ppo``, ``sft`` or ``dpo``
   * - ``--tp-size``
     - ``4``
     - ``1``
     - Tensor parallelism (slicing the model across multiple GPUs)
     - Single GPU = 1; with multiple GPUs, usually set to the number of GPUs
   * - ``--world-size``
     - ``8``
     - ``1``
     - Total number of GPUs
     - Single GPU = 1; with multiple GPUs, set to the actual GPU count. Note ``world_size % tp_size == 0``

.. warning:: Pitfall

   ``--tp-size`` defaults to 4 and ``--world-size`` defaults to 8. If single-GPU users don't explicitly set them to 1, they'll get an error like ``world_size must be divisible by tp_size`` or insufficient CUDA devices. **For single-GPU training, be sure to use ``--tp-size 1 --world-size 1``.**

Rollout Control
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 30 26 34 60

   * - Parameter
     - Default
     - This Setting
     - Purpose
     - Why Set This Way
   * - ``--batch-size``
     - ``32``
     - ``1``
     - Number of prompts used for training each step
     - Smoke test minimizes memory; for real training you can increase it based on available memory
   * - ``--n-samples``
     - ``8``
     - Default (not set explicitly)
     - How many completions to sample per prompt
     - GSPO needs in-group comparison; larger n_samples is more reliable but slower
   * - ``--max-running-prompts``
     - Auto (= batch_size x n_samples)
     - Not set explicitly
     - Maximum number of prompts doing rollout inference concurrently
     - Default is 64 (see ``areno/api/config.py``). If memory is low, set 1-4 manually
   * - ``--max-prompt-tokens``
     - ``1024``
     - Default
     - Maximum tokens per prompt
     - GSM8K problems are short; 1024 is enough
   * - ``--max-new-tokens``
     - ``3071``
     - Default
     - Maximum generated length per completion
     - GSM8K solutions usually take 200-500 tokens; the default is enough for math problems
   * - ``--temperature``
     - ``1.0``
     - Default
     - Sampling temperature (1.0 = raw distribution, 0 = greedy)
     - Use 1.0 during exploration; use lower values for stability
   * - ``--attn-backend``
     - ``flash``
     - Default
     - Attention computation backend
     - ``flash`` uses FlashAttention (fast); ``native`` uses AReno's own CUDA kernel (better compatibility)

Training Control
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 12 80

   * - Parameter
     - Default
     - Description
   * - ``--mini-bs``
     - ``16``
     - Backend training micro-batch size. Reduce it if memory is insufficient
   * - ``--epochs``
     - ``10``
     - Number of passes over the dataset
   * - ``--lr``
     - ``1e-6``
     - Learning rate
   * - ``--adam-8bit``
     - ``False``
     - **Strongly recommended**: replace FP32 Adam with 8-bit Adam, cutting the optimizer state from 8x to 2x the parameter count. Models 7B+ on 24-48 GB cards almost always need it
   * - ``--activation-checkpointing``
     - ``True``
     - Trade compute for memory: recompute activations during backward instead of saving them
   * - ``--drop-rollout-state``
     - ``False``
     - Set ``True`` to free the KV cache after rollout completes, giving room back to the training phase

Smoke-Specific Flags
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 90

   * - Parameter
     - Purpose
   * - ``--smoke-infer``
     - Loads the model + allocates the KV cache + records the CUDA graph, then exits. **Generates no tokens, trains nothing**
   * - ``--smoke-train``
     - On top of smoke-infer, runs one more synthetic backward + optimizer step. ``--smoke-infer`` and ``--smoke-train`` are mutually exclusive
   * - ``--tune-params``
     - Automatically probes the best combination of batch_size / max_running_prompts / mini_bs for the current hardware. Very useful when memory is tight

.. admonition:: How the Three Relate

   ``--smoke-infer`` verifies the inference pipeline → ``--smoke-train`` verifies the training pipeline → ``--tune-params`` finds the best parameters → real training.

3 Reading the Expected Output Section by Section
-----------------------------------------------------

Real training output roughly divides into several phases. Below is the typical log from running the 5.1.4 command (simplified):

Phase 1: Config Echo
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   AReno training config
   Algorithm
     name               gspo
     default_loss       gspo_loss_fn
     requires_rollout   yes
   Inputs
     ckpt               /path/to/Ling-3.0-tiny
     dataset_path       gsm8k:main
     model_hub          modelscope
     dataset_loader     examples/math/dataset_loader.py
     reward_fn          examples/math/math_verify_reward.py
   Runtime
     world_size         1
     tp_size            1
     dp_size            1
     devices            0
     attn_backend       flash
   Rollout
     batch_size         1
     max_prompt_tokens  64
     max_new_tokens     16
     n_samples          8
     max_running_prompts 1
   Training
     mini_bs            16
     ...
     optimizer          lr=1e-06, ..., adam_8bit=yes
   Outputs
     save_path          none
     ...
   WARNING: no checkpoint output path configured (--save-path); checkpoints will not be saved.

All the key information is visible at a glance. Pay special attention to:
- ``requires_rollout: yes`` — GSPO is online RL and will do rollout.
- ``WARNING: ... checkpoints will not be saved.`` — If you forgot ``--save-path``, this reminds you.

Phase 2: Model Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [init] Loading checkpoint from /path/to/Ling-3.0-tiny...

This step actually does a lot (source ``areno/cli/train.py::run()`` → ``Trainer.__init__()`` → ``trainer.init()``):

1. **Load tokenizer**: read ``tokenizer_config.json``, ``tokenizer.json`` from the checkpoint directory
2. **Load model config**: read ``config.json``, identify the model family (Ling/Qwen/LLaMA/...), and create the matching ``ModelAdapter``
3. **Launch Worker processes**: ``Trainer`` → ``Backend`` → ``Engine`` → ``Worker``
4. **Load weights**: load the safetensors onto the GPU

Phase 3: Rollout (the Model Generates Completions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [rollout] Generating completions for 1 prompts × 8 samples...

This is the core of RL training. Source path:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Call chain
     - Description
   * - ``Trainer (PolicyOnlyTrainer)``
     -
   * - ``→ rollout_token_batch(prompts, n_samples=8)``
     -
   * - ``→ Backend``
     -
   * - ``→ Engine``
     -
   * - ``→ InferenceManager (Continuous Batching)``
     - continuous batching scheduling
   * - ``→ Worker (actual inference)``
     - actual inference

Key concepts:
- **Continuous Batching**: multiple prompts don't need to wait to be batched together; whoever finishes first exits first, and new prompts can join at any time. This keeps GPU utilization higher.
- **8 completions per prompt**: this is GSPO's in-group comparison mechanism — the 8 completions compare against each other; good ones get higher reward, bad ones get lower, and no extra critic network is needed.

If the dataset has only 2 samples and ``batch_size=1``, each epoch you'll see:

.. code-block:: text

   Batch 1/2: prompt_0 × 8 samples
   Batch 2/2: prompt_1 × 8 samples

Phase 4: Reward Scoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [reward] Scoring 8 completions...

Call ``reward_fn(record)`` from ``math_verify_reward.py`` for each of the 8 completions.

**How the reward function works** (source ``examples/math/math_verify_reward.py``):

.. code-block:: python

   def reward_fn(record) -> float:
       ground_truth = record.answer[0]      # 标准答案
       gt_parsed = parse(ground_truth)       # 用 math_verify 解析
       pred_parsed = parse(record.completion) # 从模型输出中提取 \boxed{...}
       return 1.0 if verify(gt_parsed, pred_parsed) else 0.0

- ``math_verify`` is a symbolic math verification library that recognizes ``\boxed{42}`` and ``\boxed{42.0}`` as the same answer
- The model needs to write ``\boxed{final answer}`` in its output to be extracted correctly
- Correct answer = 1.0; wrong or malformed = 0.0

**The dataset loader's contract** (source ``examples/math/dataset_loader.py``):

.. code-block:: python

   def load_training_dataset(dataset_path, ...):
       # 嗅探数据格式
       if "prompt" in first:       # 已规范化 → 直接返回
           return dataset
       if "question" in first:     # GSM8K 格式 → 转换为 prompt + solutions
           return dataset.map(_format_gsm8k_record)
       if "problem" in first:      # NuminaMath 格式 → 转换
           return dataset.map(_format_math_record)

The GSM8K conversion logic:

.. code-block:: python

   def _format_gsm8k_record(record):
       answer = str(record["answer"])
       final = answer.rsplit("####", 1)[-1].strip()  # 提取 "#### 42" → "42"
       return {
           "prompt": "Solve the following grade-school math problem...\n"
                     f"Problem: {record['question']}\nSolution:",
           "solutions": [final],
       }

Phase 5: Advantage Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [train] Computing group advantages...

The 8 completions of the same prompt form a "group"; within the group you do Z-score normalization:

.. math::

   A_i = \frac{r_i - \text{mean}(r_1...r_8)}{\text{std}(r_1...r_8) + \epsilon}

- advantage > 0 → this completion is better than the group average → **encourage**
- advantage < 0 → this completion is worse than the group average → **suppress**
- advantage ≈ 0 → this completion is about average → **no update**

**Key design**: this in-group comparison naturally acts as a baseline, so no extra critic network is needed.

.. admonition:: Field Observation (measured on DGX Spark)

   When all 2 GSM8K samples are answered wrong, every reward = 0, advantage = 0, gradients = 0. Then ``grad_norm`` is basically 0, ``loss=0.0``, ``ratio_mean=1.0``. **This is not a bug; it's normal RL behavior—no positive signal, no learning.** You need a larger dataset and adjusted temperature to give the model a chance to answer correctly and produce a positive signal.

Phase 6: Training Step
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [train] step=0, loss=..., lr=..., tokens=...

This step runs a full forward + backward + optimizer step. The concrete flow (detailed in GSPO):

1. **Logprob recomputation**: the current policy recomputes logprobs for the completion tokens (compared against the old logprobs from rollout)
2. **Loss computation**: GSPO sequence-level loss (see GSPO)
3. **Backward**: gradient computation
4. **Optimizer step**: Adam updates the parameters (with ``--adam-8bit``, this uses the 8-bit state)
5. **New weights take effect**: the next rollout automatically uses the new policy

**What the training statistics mean**:

.. list-table::
   :header-rows: 1
   :widths: 30 35 40

   * - Statistic
     - Meaning
     - Healthy Range
   * - ``loss``
     - GSPO sequence-level loss
     - Nonzero, decreasing over time
   * - ``grad_norm``
     - Gradient norm
     - 1-100 normal; persistently 0 means no learning signal
   * - ``reward_mean``
     - Average reward this step
     - 0-1, higher is better
   * - ``advantage_mean``
     - Average advantage this step
     - ≈0 (group normalization centers it at 0)
   * - ``ratio_mean``
     - Mean importance ratio
     - ≈1.0 normal; >>1 means the policy changed too much
   * - ``rollout_logprob_mean``
     - Mean logprob from rollout
     - starts around -2~-8 and may change over training

Phase 7: The Loop
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [train] step=0, ...
   [rollout] Generating completions for ...
   [reward] Scoring ...
   [train] step=1, ...
   ...
   [epoch 0 complete]
   [epoch 1 start]
   ...

``--epochs 10`` means iterating the entire dataset 10 times. With 2 samples + batch_size=1 = 2 steps per epoch, 20 steps total.

Phase 8: Normal Exit
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   epoch=9 stage=epoch_end

Training completes and the process exits normally. If you see ``Killed`` in the middle (exit code 137 or -9), the OOM Killer killed it — immediately reduce the parameters and retry.

4 Common Errors and Solutions
----------------------------------

This section is ordered by how often you'll hit each problem, and covers every pitfall validated in real DGX Spark practice.

4.1 CUDA Out of Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ...

Or the process is killed directly (no Python traceback, only ``exit code 137`` or ``exit code -9``).

**Diagnosis**: during training, peak memory occurs at optimizer.step(). At that moment you simultaneously have: model weights + KV cache + activations + gradients + optimizer state.

**Solutions (by priority)**:

.. list-table::
   :header-rows: 1
   :widths: 14 48 55

   * - Priority
     - Action
     - Effect
   * - 🔴 Must
     - add ``--adam-8bit``
     - optimizer state 95 GB → 12 GB (7B model)
   * - 🟠 High
     - ``--max-running-prompts 1``
     - reduces concurrent KV cache
   * - 🟠 High
     - ``--drop-rollout-state``
     - frees KV cache after rollout
   * - 🟡 Medium
     - ``--max-prompt-tokens 64 --max-new-tokens 16``
     - reduces per-sequence memory
   * - 🟡 Medium
     - ``--batch-size 1``
     - reduces the number of prompts handled at once
   * - 🟢 Low
     - ``--mini-bs 1``
     - reduces the training micro-batch size
   * - 🟢 Low
     - ``--activation-checkpointing`` (on by default)
     - trades 20-30% time for ~50% memory

.. warning:: DGX Spark-Specific Reminder

   The GB10 uses a unified memory architecture: system memory and GPU memory share one 128 GB LPDDR5x pool. If other processes run on the system (e.g., a Ray cluster, browsers), the memory they occupy comes from the same pool as the memory used for training, so the OOM risk is higher. Before training, run ``free -h`` to confirm available memory > what the model needs.

4.2 FlashAttention Not Installed or Failing to Compile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ModuleNotFoundError: No module named 'flash_attn'

**Cause**: ``--attn-backend flash`` (default) needs the ``flash-attn`` package, but it doesn't come with ``pip install areno`` automatically (it has many build dependencies and takes a long time).

**Solution (standard x86_64)**:

.. code-block:: bash

   pip install flash-attn --no-build-isolation

**Solution (DGX Spark / ARM64)**:

On ARM64, flash-attn compiles four architectures by default — sm_80, sm_90, sm_100, sm_120 — which triggers OOM on GB10 and ends in ``ninja: build stopped``. You must restrict it to compiling only the architecture you need:

.. code-block:: bash

   export FLASH_ATTN_CUDA_ARCHS="120"    # 只编 sm_120
   export MAX_JOBS=8                      # 限制编译并行度
   pip install flash-attn==2.7.3 --no-build-isolation

.. admonition:: Key

   flash-attn reads ``FLASH_ATTN_CUDA_ARCHS``, **not** ``TORCH_CUDA_ARCH_LIST``. The two environment variables must be set separately.

**If you really can't install flash-attn**: switch to ``--attn-backend native`` (AReno's own CUDA attention kernel) — better compatibility but slightly slower:

.. code-block:: bash

   areno train ... --attn-backend native

4.3 areno_accel CUDA Extension Compilation Failed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   RuntimeError: areno_accel extension not found. Please build CUDA extensions first.

**Solution**:

.. code-block:: bash

   # 确认 CUDA 编译器可用
   nvcc --version
   echo $CUDA_HOME

   # 重新安装（触发 CUDA 扩展编译）
   pip install -e . --no-build-isolation

   # 如果内存不足（ARM64 常见）
   export MAX_JOBS=8
   pip install -e . --no-build-isolation

4.4 Dataset Loading Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   KeyError: "math dataset rows must contain `prompt`, GSM8K-style `question`/`answer`, or NuminaMath-style `problem`/`solution` fields"

**Cause**: ``--dataset-loader-fn examples/math/dataset_loader.py`` can only handle three formats: normalized JSONL (with a ``prompt`` field), the GSM8K format (``question``/``answer``), and the NuminaMath format (``problem``/``solution``). Your JSON is some other format.

**Solution**: following the ``_format_gsm8k_record`` function in ``examples/math/dataset_loader.py``, write your own dataset loader that produces the two fields ``prompt`` (a string) and ``solutions`` (a list of strings):

.. code-block:: python

   # my_dataset_loader.py
   def load_training_dataset(dataset_path, *, default_loader, **kwargs):
       dataset = default_loader(dataset_path)
       def normalize(row):
           return {
               "prompt": f"Question: {row['my_question_field']}\nAnswer:",
               "solutions": [row['my_answer_field']],
           }
       return dataset.map(normalize)

.. code-block:: bash

   areno train ... --dataset-loader-fn my_dataset_loader.py

4.5 Reward Function KeyError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   KeyError: "math reward expects `record.answer`; use the math dataset loader to normalize raw rows"

**Cause**: ``reward_fn(record)`` in ``math_verify_reward.py`` reads ``record.answer``, which is mapped automatically by AReno internally after rollout (the loader produces ``solutions``, and the trainer internally converts it to ``answer``). If you didn't use the accompanying math dataset loader, or your custom loader doesn't produce the ``solutions`` field, you get this error.

**Solution**: make sure the dataset loader produces the ``solutions`` field (a list of strings), or write a custom reward function.

4.6 Slow Model Download / ModelScope Unavailable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution (for users in China, slow HuggingFace)**:

.. code-block:: bash

   # 方案 A：用 ModelScope（AReno 默认）
   pip install modelscope
   areno train --ckpt inclusionAI/Ling-3.0-tiny --model-hub modelscope ...

   # 方案 B：HuggingFace 镜像
   export HF_ENDPOINT=https://hf-mirror.com
   areno train --ckpt inclusionAI/Ling-3.0-tiny --model-hub hf ...

**Solution (download manually to local)**:

.. code-block:: bash

   # ModelScope CLI
   modelscope download --model inclusionAI/Ling-3.0-tiny --local_dir ./models/Ling-3.0-tiny

   # 然后用本地路径
   areno train --ckpt ./models/Ling-3.0-tiny ...

4.7 ``--world-size`` and ``--tp-size`` Mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   UsageError: --world-size must be divisible by --tp-size

or

.. code-block:: text

   RuntimeError: CUDA device count < world_size

**Cause**: ``--tp-size`` defaults to 4 and ``--world-size`` defaults to 8. A single-GPU user left them at the defaults.

**Solution**: single-GPU training must explicitly set ``--tp-size 1 --world-size 1``.

4.8 DGX Spark-Specific: CUDA Capability Warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
   Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)

**This is a warning, not an error**. PyTorch 2.9.1's official ceiling is compute capability 12.0 while the GB10 is 12.1. In practice it does not affect import or training — as long as ``areno check`` is all green. If some CUDA kernel actually hits a compatibility issue, it shows up as a runtime error rather than silently misbehaving.

4.9 DGX Spark-Specific: Compiling C Extensions Without Root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   fatal error: Python.h: No such file or directory

**Cause**: the ``python3-dev`` package is not preinstalled on ARM64 Ubuntu, and without sudo you can't ``apt install``.

**Solution**:

.. code-block:: bash

   # 1. 下载 deb 包但不安装
   apt-get download python3.12-dev
   # 2. 解压提取头文件
   dpkg-deb -x python3.12-dev*.deb /tmp/pyhdrs_extract/
   # 3. 整理到统一目录
   mkdir -p ~/pyhdrs
   cp -r /tmp/pyhdrs_extract/usr/include/python3.12/* ~/pyhdrs/
   # 或创建符号链接到标准路径
   # 4. 编译时注入 CPATH
   export CPATH=$HOME/pyhdrs:$HOME/pyhdrs/aarch64-linux-gnu

| You need this ``CPATH`` every time you compile flash-attn or the AReno CUDA extensions. It's recommended to add it to ``~/.bashrc``.

4.10 Quick Diagnostic Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running real training, run the following checks to avoid most problems:

.. code-block:: bash

   # 1. 环境诊断
   areno check                # 期望全部 OK
   areno env --json           # 查看完整环境信息

   # 2. 显存检查
   python -c "import torch; free, total = torch.cuda.mem_get_info(); print(f'{free/1024:.0f} MiB free / {total/1024:.0f} MiB total')"

   # 3. Smoke 冒烟（最关键的一步）
   areno train --ckpt ./models/Ling-3.0-tiny \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --algo gspo --attn-backend flash \
     --smoke-infer \
     --batch-size 1 --max-running-prompts 1 \
     --max-prompt-tokens 64 --max-new-tokens 16 \
     --tp-size 1 --world-size 1
   # 成功输出：AReno smoke infer ok, peak_mem_frac=...

If ``areno check`` has a red item, fix it before continuing. If smoke infer fails, look up the error message in the table above.

5 The Full Pipeline from Smoke Test to Real Training
---------------------------------------------------------

Here's a concrete, actionable roadmap:

.. list-table::
   :header-rows: 1
   :widths: 24 70 8

   * - Stage
     - Checks / Commands
     - Outcome
   * - 1. Environment prep
     - ``areno check`` → all green?
       ``nvidia-smi`` → GPU visible?
       ``free -h`` → enough memory?
     - ✅
   * - 2. Model in place
     - ``ls ./models/Ling-3.0-tiny/config.json`` → exists?
       or use ``--model-hub ms`` to let AReno download it automatically
     - ✅
   * - 3. Smoke Infer (can the model stand?)
     - ``areno train --smoke-infer ...``
       → ``AReno smoke infer ok``
       → ``peak_mem_frac < 0.8``?
     - ✅
   * - 4. Smoke Train (can backward run?)
     - ``areno train --smoke-train --adam-8bit ...``
       → ``AReno smoke train ok``
     - ✅
   * - 5. Real minimal training (full-pipeline validation)
     - ``areno train --algo gspo --adam-8bit \``
       ``  --save-path ./checkpoints \``
       ``  --batch-size 1 --max-running-prompts 1 \``
       ``  --tp-size 1 --world-size 1 ...``
       → ``epoch=N stage=epoch_end`` (normal exit)
     - ✅
   * - 6. Scale up training (where real results appear)
     - - Enlarge the dataset (no longer 2 samples)
       - Increase ``--batch-size`` and ``--n-samples``
       - Check that reward isn't all 0 (``grad_norm > 0``)
       - Save checkpoints with ``--save-path``
     -

6 Chapter Summary
---------------------

After reading this chapter, you should be able to:

1. **Run the complete Ling-3.0-tiny training with one command** (the command in 5.1.4 works by copy-paste)
2. **Understand each parameter's purpose and default** (the parameter tables in 5.2)
3. **Read the meaning of every line in the training log** (the section-by-section breakdown in 5.3)
4. **Troubleshoot 10 common errors on your own** (the troubleshooting checklist in 5.4)
5. **Follow the standard flow from smoke test to real training** (the roadmap in 5.5)

Once you understand this flow, the next chapter (RL in One Snippet) walks you through the same process with a Python SDK snippet, while introducing RL's core concepts — you'll see what RL idea sits behind each line of code.
