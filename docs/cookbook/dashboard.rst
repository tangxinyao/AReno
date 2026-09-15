Dashboard
=========

.. admonition:: Goal

   Master AReno's monitoring dashboard, environment diagnostics, performance tuning, and troubleshooting of common failures.

.. admonition:: Prerequisites

   DGX Spark and Aliyun Cloud (hardware/environment).

.. admonition:: Outcome

   Be able to independently diagnose the AReno environment, tune performance, and troubleshoot 90% of common issues on your own.

AReno ships a complete operations toolchain covering everything from training monitoring to troubleshooting. This chapter covers the Dashboard tech stack, the ``areno check`` / ``areno env`` diagnostics commands, performance tuning parameters, and how to troubleshoot common failures.

1 Dashboard Tech Stack
-------------------------

AReno's Web Dashboard is a single-page application (SPA) for monitoring training and serving status in real time.

Front-End Tech Stack
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 20 64

   * - Component
     - Technology
     - Notes
   * - Build tool
     - Vite 6
     - Very fast dev server and build
   * - UI library
     - React 19
     - Function components + Hooks
   * - Icons
     - Lucide React
     - Lightweight icon library
   * - Markdown
     - react-markdown + remark-gfm
     - Renders Agent chat inside the Dashboard
   * - Package manager
     - pnpm 11.9
     - Strict dependency management
   * - Build output
     - ``areno/dashboard/dist/``
     - Static assets embedded in the Python package

``dashboard/package.json`` lists only 4 runtime dependencies, keeping the final build artifact lightweight.

Five Pages
~~~~~~~~~~

The Dashboard has five functional pages, all coded in ``dashboard/src/main.jsx`` (3000+ lines):

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Page
     - Function
   * - **Overview**
     - Training metrics overview (loss curves, token throughput, learning rate)
   * - **Jobs**
     - Active training/serving task list, filter by status, task details
   * - **Runtime**
     - GPU utilization, environment checks, GPU memory usage
   * - **Launcher**
     - Training/serving launcher with a preflight tuning check
   * - **Agent**
     - Embedded AReno operations agent (natural-language interaction)

Back-End API
~~~~~~~~~~~~

The Dashboard backend is a Python HTTP server (``areno/dashboard/server.py``). Main APIs:

.. code-block:: text

   GET  /api/jobs             - list all active jobs
   GET  /api/jobs/:id         - job details (metrics, logs)
   POST /api/launcher/preflight - pre-launch check (GPU availability, TP divisibility)
   GET  /api/runtime          - environment diagnostics
   POST /api/agent/chat       - operations agent messages

How to Start It
~~~~~~~~~~~~~~~

.. code-block:: bash

   # carried automatically in train/serve processes
   areno train ... --dashboard  # starts the Dashboard automatically during training

   # or start it standalone
   areno dashboard --start --port 8765

   # stop it
   areno dashboard --stop

Once the Dashboard is running, visit ``http://localhost:8765``.

Embedded Agent
~~~~~~~~~~~~~~

The Dashboard embeds an operations agent that talks to a local model through an OpenAI-compatible API and understands natural-language instructions:

.. code-block:: text

   "Run a GSPO training on gsm8k with Ling-3.0-tiny"
   "Check the current GPU status"
   "Why did training OOM? Help me tune the parameters"

The agent's core code lives in ``areno/agent/agent_loop.py`` and ``areno/agent/tools.py``. The tools it supports are: ``list_files``, ``inspect_tree``, ``read_file``, ``rg``, ``run_command``, ``write_file``, ``replace_text``, ``apply_patch``, ``submit``.

2 Environment Diagnostics
----------------------------

AReno provides two CLI commands for environment diagnostics.

areno check
~~~~~~~~~~~

.. code-block:: bash

   areno check

Example output:

.. code-block:: text

   AReno check: ready

   OK   Python >= 3.10            (3.12.0)
   OK   Platform is Linux
   OK   PyTorch installed         (2.6.0+cu124)
   OK   CUDA available
   OK   NVIDIA GPU visible        (1 GPU: NVIDIA GeForce RTX 4090)
   OK   flash-attn installed
   OK   areno_accel extension loaded
   OK   CUDA_HOME set             (/usr/local/cuda-12.4)
   OK   nvcc found
   OK   Metrics log dir writable
   OK   HF cache writable

The checklist covers the whole chain, from the Python version to the CUDA toolchain to dependent libraries. The implementation lives in ``areno/cli/diagnostics.py`` (485 lines).

Check Details
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - Check
     - Condition
     - Common cause of FAIL
   * - Python ≥ 3.10
     - ``sys.version_info``
     - The system Python is too old
   * - Platform Linux
     - ``platform.system()``
     - Running on macOS/Windows
   * - PyTorch
     - ``import torch``
     - Not installed or installation failed
   * - CUDA available
     - ``torch.cuda.is_available()``
     - CPU-only PyTorch build, driver issue
   * - NVIDIA GPU
     - ``torch.cuda.device_count() > 0``
     - Driver not loaded, GPU not visible
   * - flash-attn
     - ``import flash_attn``
     - Not installed or failed to compile
   * - areno_accel
     - ``import areno.accel._areno_accel``
     - CUDA extension failed to compile
   * - CUDA_HOME
     - Environment variable or auto-detection
     - nvcc not on PATH
   * - nvcc
     - ``shutil.which("nvcc")``
     - CUDA Toolkit not installed

areno env
~~~~~~~~~

.. code-block:: bash

   areno env

It prints a full environment report (Python version, PyTorch version, CUDA version, GPU info, all dependency versions, environment variables, path info). ``--json`` gives a machine-readable format:

.. code-block:: bash

   areno env --json > env_report.json

3 Performance Tuning
-----------------------

Auto-Tuning: --tune-params
~~~~~~~~~~~~~~~~~~~~~~~~~~

AReno's auto-tuning (``areno/cli/auto_tune.py``, 688 lines) automatically searches for the best combination of ``batch_size``, ``mini_bs``, ``n_samples``, and ``max_running_prompts``:

.. code-block:: bash

   areno train \
     --ckpt ./models/Ling-3.0-tiny \
     --dataset-path gsm8k:main \
     --algo gspo \
     --tune-params \
     --mem-frac 0.9 \
     --tune-max-samples 256

Tuning logic: a two-phase search:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Phase
     - Content
   * - Phase 1: Rollout concurrency search
     - Test candidate values of ``max_running_prompts`` × ``n_samples`` from high to low
       Run a rollout with a dummy model → measure peak GPU-memory ratio
       Pick the first configuration ≤ ``mem_frac`` (default 0.9)
   * - Phase 2: Training micro-batch search
     - Fix ``batch_size`` from Phase 1
       Test ``mini_bs`` from high to low
       Run one train step with a dummy model → measure peak GPU memory

Candidates are powers of two plus sparse sampling (at most 16 rollout candidates, at most 32 train candidates), so the search finishes in reasonable time (usually < 2 minutes).

Key Tuning Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Parameter
     - Default
     - Tuning advice
   * - ``--mem-frac 0.9``
     - 0.9
     - Lower it to leave headroom for CUDA graphs and the allocator (0.85 recommended on H20)
   * - ``--drop-rollout-state``
     - False
     - **Strongly recommended**: free rollout state after every train step, trading GPU memory for I/O
   * - ``--activation-checkpointing``
     - True
     - Swaps 20-30% time for ~50% GPU memory (on by default; not recommended to turn off)
   * - ``--max-running-prompts``
     - auto
     - Increasing → higher throughput but more KV-cache memory
   * - ``--n-samples 8``
     - 8
     - Lowering → faster but less reliable advantage comparisons
   * - ``--batch-size``
     - auto
     - Raising → faster convergence but more memory
   * - ``--mini-bs``
     - auto
     - Smaller → saves memory but noisier gradients
   * - ``--adam-8bit``
     - False
     - Enable → optimizer state uses 75% less memory (slightly lower precision)

drop-rollout-state in Detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is one of AReno's most important GPU-memory-saving switches:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Flow
   * - Normal flow
     - init → rollout (occupies rollout cache) → train → rollout → train → ...
   * - ``--drop-rollout-state``
     - init → rollout → train → ``optimizer.offload_state()`` → ``torch.cuda.empty_cache()``
       → rollout (reloads weights) → train → ...
   * - Cost
     - one extra weight offload/onload per step (0.5-2 seconds depending on model size)
   * - Benefit
     - frees the rollout KV cache + sampling-state memory (usually 2-8 GB)

**Default recommendation**: always keep ``--drop-rollout-state`` enabled for online RL training. Turn it off only for performance stress testing.

4 Troubleshooting Common Failures
-------------------------------------

OOM (Out of Memory)
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   RuntimeError: CUDA out of memory. Tried to allocate X MiB

**Troubleshooting priority**:

1. **Enable --drop-rollout-state** (the most direct fix, usually resolves it)
2. **Reduce --max-running-prompts** (fewer rollout sequences running at once)
3. **Reduce --batch-size** or **--n-samples**
4. **Enable --adam-8bit** (optimizer state from FP32×3 → uint8×2, saving ~75%)
5. **Reduce --mini-bs** (smaller training microbatch)
6. **Reduce --max-new-tokens** (shorter KV cache per sequence)

CUDA Extension Compilation Failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ModuleNotFoundError: No module named 'areno.accel._areno_accel'

**Troubleshooting steps**:

.. code-block:: bash

   # 1. verify the CUDA toolchain
   nvcc --version
   echo $CUDA_HOME

   # 2. verify the PyTorch CUDA version matches
   python -c "import torch; print(torch.version.cuda)"

   # 3. clean and recompile
   pip uninstall areno -y
   rm -rf build/
   pip install -e . --no-build-isolation

   # 4. if it still fails, set environment variables and retry
   export MAX_JOBS=4          # limit parallel compilation (prevents OOM on ARM64)
   export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # specify the GPU architecture

FlashAttention Installation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ImportError: No module named 'flash_attn'

**Solution**:

.. code-block:: bash

   # standard install
   pip install flash-attn --no-build-isolation

   # if compilation fails (common on ARM64 / DGX Spark)
   # fallback: use PyTorch SDPA
   areno train ... --attn-backend sdpa

``--attn-backend sdpa`` uses ``torch.nn.functional.scaled_dot_product_attention`` built into PyTorch 2.0+, with performance close to flash-attn and zero extra dependencies.

CUDA / PyTorch Version Mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   NVIDIA A100-SXM4-80GB with CUDA capability sm_80 is not compatible

**Cause**: the CUDA version PyTorch was compiled against is incompatible with the system CUDA driver.

**Solution**: install a matching PyTorch:

.. code-block:: bash

   # CUDA 12.4
   pip install torch --index-url https://download.pytorch.org/whl/cu124

   # CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

Tool-Call Errors (Agentic RL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   JSONDecodeError: Expecting value: line 1 column 1 (char 0)

**Cause**: the model's tool_call JSON is malformed.

**Troubleshooting**:
1. Check whether the model supports function calling (Ling-3.0-tiny supports it natively)
2. Manually test tool calls against the chat API with ``areno serve``
3. Check whether ``--max-new-tokens`` is large enough to generate the complete JSON

Debugging Reward Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: loss does not decrease, or rewards are all 0.

**Troubleshooting**:

.. code-block:: python

   # add debug logging in reward_fn
   def reward_fn(example, completions):
       rewards = []
       for completion in completions:
           score = compute_score(example, completion)
           print(f"[DEBUG] completion={completion[:100]}... score={score}")
           rewards.append(score)
       return rewards

AReno captures the Worker process's stdout, so you can see this debug output in the Dashboard or logs.

Training Not Converging
~~~~~~~~~~~~~~~~~~~~~~~

**Common causes and solutions**:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Symptom
     - Possible cause
     - Solution
   * - Loss oscillates wildly
     - Learning rate too high
     - Lower ``--lr`` (e.g. 2e-5 → 5e-6)
   * - Loss slowly rises
     - Reward function has issues
     - Check the reward distribution (all 0s / all 1s?)
   * - Loss unchanged
     - The model is not training
     - Check ``_train_state_ready`` and whether gradients are zero
   * - Reward unchanged
     - Temperature too low
     - Raise ``--temperature`` (0.8 → 1.2) to add randomness
   * - OOM keeps happening
     - Batch too large
     - Use ``--tune-params`` to auto-search suitable parameters

Quick Verification Fix
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # quickly verify with a smoke test whether the environment is fixed
   areno train --smoke-train \
     --ckpt ./models/Ling-3.0-tiny \
     --algo gspo \
     --tp-size 1 --world-size 1

   # success output: train_stats={loss=...} → environment OK

The smoke test uses dummy data and a minimal configuration, can finish in under 30 seconds, and is the first line of defense in troubleshooting.

Operations Command Quick Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - Purpose
   * - ``areno check``
     - Environment diagnostics (continue only if everything passes)
   * - ``areno env``
     - Full environment report
   * - ``areno env --json``
     - Machine-readable report
   * - ``areno dashboard --start``
     - Start the monitoring dashboard
   * - ``areno dashboard --stop``
     - Stop the monitoring dashboard
   * - ``areno train --tune-params``
     - Auto-search tuning parameters
   * - ``nvidia-smi``
     - Real-time GPU status
   * - ``watch -n 1 nvidia-smi``
     - Continuously monitor the GPU
