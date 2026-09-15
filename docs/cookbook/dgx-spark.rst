DGX Spark
=========

.. note:: TL;DR

   Unbox the DGX Spark → verify the CUDA environment → work through the ARM64-specific build issues → install AReno → download the model → run a smoke test to validate the whole pipeline. Expect 30-90 minutes (depending on network and compilation), with the flash-attn build possibly taking half that time. **This chapter is validated on a real GB10 and covers real pitfalls such as no root access, a missing Python.h, flash-attn OOM, and unified memory contention.**

Ling-3.0-tiny covered "what to run" and "where to run it". This chapter gets hands-on — turn the DGX Spark on your desk into an AReno workstation that can run RL training.

The content in this chapter comes from hands-on operating notes on a GB10 (DGX Spark) and covers the entire journey from unboxing to a working end-to-end GSPO training run. If a step doesn't quite match your environment (for example, your PyTorch version differs, or you do have root access), adjust accordingly — the core approach and methods are universal.

1 Unboxing Check
------------------

Physical connections
~~~~~~~~~~~~~~~~~~~~

The back of the DGX Spark has three ports you need to connect:

.. list-table:: DGX Spark rear panel
   :widths: 25 25 25 25

   * - [USB-C PD]
     - [HDMI]
     - [10GbE RJ45]
     - [2× 200GbE QSFP]
   * - 240W power
     - Monitor
     - Network
     - ← not used in this chapter

1. **USB-C PD power**: Connect the included 240W power adapter. The DGX Spark has no separate power switch — plug in power and it boots.
2. **HDMI display**: You need a display for the first boot to complete Ubuntu's initial setup. After that you can work over SSH.
3. **10GbE network**: Plug in the network cable. DHCP assigns the IP automatically; no fixed IP is needed.

First boot
~~~~~~~~~~

Once power is connected the DGX Spark boots automatically. If a display is attached, you'll see the Ubuntu setup wizard:

1. Choose a language (English is recommended, to avoid garbled text in the terminal later)
2. Set a username and password (remember this password — you'll need it for ``sudo`` later)
3. Connect to Wi-Fi or confirm the wired network is connected
4. Select your time zone
5. Wait for setup to finish and enter the desktop

Verify the base environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a terminal (``Ctrl+Alt+T``) and run the commands below in sequence to confirm the environment. **This step takes only 30 seconds, but it can save you hours of debugging environment issues later.**

.. code-block:: bash

   # 1. Confirm it is an ARM64 architecture
   uname -m
   # Expected output: aarch64

   # 2. Confirm the GPU is visible
   nvidia-smi
   # Expected: shows the NVIDIA GB10 GPU, driver version >= 535

   # 3. Confirm the CUDA compiler is available
   nvcc --version
   # Expected: shows the CUDA version number (>= 12.4) and Build info

   # 4. Confirm the CUDA environment variables
   echo $CUDA_HOME
   # Expected: /usr/local/cuda or a similar path (non-empty)

The expected results of the four commands and the problems they reveal:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Command
     - Expected output
     - If you don't get...
   * - ``uname -m``
     - ``aarch64``
     - What you have isn't a DGX Spark (it may be a regular x86_64 PC). Jump to Aliyun Cloud and use the cloud plan
   * - ``nvidia-smi``
     - GPU list + driver version
     - The driver isn't installed. ``sudo apt install nvidia-driver-550`` (preinstalled at the factory; this is rare)
   * - ``nvcc --version``
     - CUDA version number
     - ``export PATH=/usr/local/cuda/bin:$PATH``
   * - ``echo $CUDA_HOME``
     - ``/usr/local/cuda``
     - ``export CUDA_HOME=/usr/local/cuda``

.. warning:: If both ``nvcc --version`` and ``echo $CUDA_HOME`` fail

   The DGX Spark ships with the CUDA Toolkit preinstalled, but the environment variables may not be written to ``.bashrc``. Add them manually:

   .. code-block:: bash

      echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
      echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
      source ~/.bashrc

2 Validating the CUDA Environment on ARM64
--------------------------------------------

Why validate this separately?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DGX Spark uses an **ARM64 CPU + NVIDIA GPU**. This differs from the x86_64 + NVIDIA GPU that most developers are used to. PyTorch and CUDA support on ARM64 is already mature, but there are two real-world issues to keep in mind:

1. **Compute capability mismatch**: The GB10 GPU has a compute capability of **12.1**, while the current ceiling for official PyTorch support is 12.0. Every ``import torch`` prints a warning:

   .. code-block:: text

      Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
      Minimum and Maximum cuda capability supported by this version of
      PyTorch is (8.0) - (12.0)

   This usually doesn't block execution (``areno check`` still goes all green), but some kernel behavior may be incomplete — you need to verify with actual runs.

2. **Incomplete prebuilt packages**: Prebuilt wheels for PyTorch extensions on ARM64 aren't as complete as on x86_64; many packages need to be compiled from source.

Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create a virtual environment (isolates the system Python; good habit)
   python3 -m venv ~/areno-env
   source ~/areno-env/bin/activate

Install PyTorch (ARM64 + CUDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu124

.. note::

   If your CUDA version is 13.x (the environment this chapter was written and tested in), the PyTorch CUDA 12.4 build is generally backward compatible. If you run into problems, try:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cu130

Verify that PyTorch can see the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

Expected output:

.. code-block:: text

   PyTorch: 2.6.x (or 2.9.x)
   CUDA available: True
   GPU: NVIDIA GB10

If ``CUDA available: False``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Possible cause
     - Fix
   * - PyTorch was installed as a CPU-only build
     - Confirm the install command used ``--index-url https://download.pytorch.org/whl/cu124`` (or ``cu130``)
   * - CUDA version mismatch
     - ``nvcc --version`` to confirm the CUDA version; PyTorch ≥ 2.6 requires CUDA ≥ 12.1
   * - Virtual environment not activated
     - ``source ~/areno-env/bin/activate``

.. note::

   Once ``CUDA available: True`` is confirmed: run ``deactivate`` to leave the virtual environment, and reactivate it when you install AReno.

3 Installing AReno
--------------------

Step 0: Prepare an environment variable template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the DGX Spark, **you must reset the following environment variables every time you open a new terminal** (they don't persist automatically). Save them as a script to source whenever you need them:

.. code-block:: bash

   # Save to ~/areno-env.sh; load it each time with source ~/areno-env.sh
   export CUDA_HOME=/usr/local/cuda
   export PATH=/usr/local/cuda/bin:$PATH
   export TORCH_CUDA_ARCH_LIST="12.1"
   export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
   export MAX_JOBS=8

Role of each variable:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Variable
     - Purpose
     - Why you must set it
   * - ``CUDA_HOME``
     - CUDA Toolkit installation path
     - Required for building CUDA extensions and at runtime
   * - ``TORCH_CUDA_ARCH_LIST``
     - Target GPU compute capability
     - Setting ``"12.1"`` compiles only the sm_120 kernels GB10 needs, avoiding wasted time building other architectures
   * - ``TRITON_PTXAS_PATH``
     - Triton's PTX assembler path
     - Triton's bundled ptxas doesn't recognize sm_121a; you must use the system one
   * - ``MAX_JOBS``
     - Number of parallel build processes
     - Limit to 8 so the build processes don't blow up unified memory

.. warning:: The CPATH variable

   If you **do not have root access** (common on a shared DGX Spark or in multi-user environments), you'll also need to handle the Python header files — see the next subsection.

Step 0-A: Special handling when you don't have root access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you don't have ``sudo`` access (``sudo apt install python3-dev`` would fail), building C/C++ extensions will hit ``fatal error: Python.h: No such file or directory``. The fix is to extract the Python development headers manually:

.. code-block:: bash

   # 1. Download the python3-dev package (don't install it, just unpack)
   mkdir -p /tmp/pyhdrs && cd /tmp/pyhdrs
   apt-get download python3-dev
   dpkg -x python3-dev_*.deb .

   # 2. Find the directory that contains Python.h
   find /tmp/pyhdrs -name "Python.h"
   # Assume the output is: /tmp/pyhdrs/usr/include/python3.12/Python.h

   # 3. Set CPATH so the compiler can find these headers
   export CPATH=/tmp/pyhdrs/usr/include/python3.12:/tmp/pyhdrs/usr/include/aarch64-linux-gnu

   # 4. Verify
   echo '#include <Python.h>' | gcc -fsyntax-only -x c - && echo "CPATH OK" || echo "CPATH FAIL"

**Add this CPATH to your ``~/areno-env.sh`` too** — you'll need it for every build from now on.

Step 1: Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno

.. note::

   If GitHub is slow, you can use a mirror:

   .. code-block:: bash

      git clone https://ghproxy.com/https://github.com/inclusionAI/AReno.git

Step 2: Activate the virtual environment and build/install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Load the environment variables
   source ~/areno-env.sh
   source ~/areno-env/bin/activate

   # Install flash-linear-attention first (the flash-attn build may need it)
   pip install flash-linear-attention --no-build-isolation

   # Build and install AReno (includes the areno_accel CUDA extensions)
   pip install -e . --no-build-isolation

This step builds the 12 CUDA kernels under ``areno/accel/csrc/``. Compilation takes about 5-15 minutes. If ``MAX_JOBS`` is too high and causes an OOM, drop it to 4:

.. code-block:: bash

   MAX_JOBS=4 pip install -e . --no-build-isolation

.. admonition:: ``scripts/install.sh`` vs installing manually

   The official install script ``scripts/install.sh`` automatically installs system dependencies. But if you don't have root access, installing manually with ``pip install -e . --no-build-isolation`` is more controllable — it skips the system-package install steps that need sudo.

4 Compiling FlashAttention (the Easiest Step to Trip On ARM64)
----------------------------------------------------------------

Why is this covered separately?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building flash-attn on a GB10 is the most error-prone part of the whole environment setup. Without understanding the pitfalls below up front, you may repeatedly hit errors such as ``Killed`` and ``ninja: build stopped``, wasting hours.

There are two core reasons:

1. **GB10 uses a unified memory architecture** — build processes, system processes, and GPU processes share the same ~121 GB memory pool. flash-attn's default build behavior spawns hundreds of heavy build processes that blow up the whole pool at once.
2. **ARM64 has almost no prebuilt flash-attn wheels** — on x86_64 you can ``pip install flash-attn`` in seconds, but on ARM64 you must compile from source.

The core problem: why does the default build OOM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default flash-attn compiles kernels for 4 GPU architectures:

.. code-block:: text

   Default TORCH_CUDA_ARCH_LIST: "8.0;9.0;10.0;12.0"
                                   sm_80  sm_90  sm_100  sm_120

flash-attn has about 85 ``.cu`` source files, and each is compiled once for each of the 4 architectures: **85 × 4 ≈ 340 heavy compilations**. On a GB10 with unified memory and a 20-core CPU, ``MAX_JOBS`` may default to 20 or even be unlimited, which means 20 nvcc processes running at once — each peaking at 2-3 GB, for 40-60 GB total. Add the system and other processes (you may be running a Ray cluster or other services at the same time) and you instantly hit the ceiling → the OOM killer kills processes → ``ninja: build stopped``.

**The fix is simple: compile only the sm_120 architecture that GB10 needs.**

Key switch: ``FLASH_ATTN_CUDA_ARCHS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is an easy-to-miss detail: flash-attn reads its own environment variable ``FLASH_ATTN_CUDA_ARCHS``, **not** PyTorch's standard ``TORCH_CUDA_ARCH_LIST``. If you only set ``TORCH_CUDA_ARCH_LIST="12.1"`` but don't set ``FLASH_ATTN_CUDA_ARCHS``, flash-attn will still build all 4 architectures.

**The correct approach**: set both.

.. code-block:: bash

   export TORCH_CUDA_ARCH_LIST="12.1"
   export FLASH_ATTN_CUDA_ARCHS="120"   # Note: this is flash-attn's own variable!

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - Variable
     - Who reads it
     - Value
     - Effect
   * - ``TORCH_CUDA_ARCH_LIST``
     - PyTorch, areno_accel
     - ``"12.1"``
     - Builds only the PyTorch extensions for sm_120
   * - ``FLASH_ATTN_CUDA_ARCHS``
     - **flash-attn only**
     - ``"120"``
     - Builds only the flash-attn kernels for sm_120

The actual build
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Make sure all the environment variables are ready
   source ~/areno-env.sh
   source ~/areno-env/bin/activate

   # 2. Key: set flash-attn's single-architecture build switch
   export FLASH_ATTN_CUDA_ARCHS="120"   # Don't miss this one!

   # 3. Build and install (expect 10-20 minutes)
   pip install flash-attn --no-build-isolation

.. warning:: Note ``--no-build-isolation``

   flash-attn depends on PyTorch's headers. Without ``--no-build-isolation``, pip creates an isolated environment and re-downloads PyTorch — which almost certainly fails on ARM64. ``--no-build-isolation`` makes the build use the PyTorch already installed in your current environment.

Verify the build result
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -c "import flash_attn; print('flash-attn version:', flash_attn.__version__)"

If there's no error and a version number prints (e.g. ``2.7.3``), the build succeeded.

Common causes of build failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Error
     - Cause
     - Fix
   * - ``Killed`` / ``ninja: build stopped``
     - Memory exhausted — multi-architecture build + high parallelism
     - 1) Set ``FLASH_ATTN_CUDA_ARCHS="120"`` 2) ``MAX_JOBS=4``
   * - ``fatal error: Python.h: No such file``
     - No root; Python dev headers missing
     - Manually extract and set ``CPATH`` per Step 0-A in Section 3
   * - ``nvcc fatal: Unsupported gpu architecture 'compute_120'``
     - CUDA version doesn't support sm_120
     - ``nvcc --version`` to confirm CUDA ≥ 12.4
   * - ``error: identifier "PRAGMA_UNROLL" is undefined``
     - CUDA and flash-attn versions are incompatible
     - Try an earlier or newer flash-attn
   * - Build succeeds but ``import flash_attn`` errors
     - Build-time and runtime PyTorch versions differ
     - Make sure the correct virtual environment is activated during ``pip install``

.. admonition:: 🔑 If you can't get flash-attn to build at all

   flash-attn is a "recommended" dependency, not a "required" one. Add ``--attn-backend sdpa`` when training to use PyTorch's built-in SDPA attention implementation instead. Inference will be slower, but functionality is completely fine.

5 Verify the Environment
--------------------------

areno check
~~~~~~~~~~~

.. code-block:: bash

   source ~/areno-env.sh
   source ~/areno-env/bin/activate
   areno check

AReno checks the following items and reports their status:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Check item
     - Importance
     - Description
   * - Python version ≥ 3.10
     - Required
     - Base runtime environment
   * - PyTorch ≥ 2.6 + CUDA
     - Required
     - Core dependency for training and inference
   * - ``nvcc`` available
     - Required
     - Needed to compile CUDA extensions
   * - ``areno_accel`` extension
     - Required
     - Python interface for the 12 CUDA kernels
   * - FlashAttention
     - Recommended
     - Speeds up attention computation (still runs without it, using ``--attn-backend sdpa``)
   * - flash-linear-attention
     - Recommended
     - Speeds up linear attention (needed for KDA/MLA)
   * - ``CUDA_HOME`` environment variable
     - Recommended
     - Needed for building and some diagnostics

**Expected result**: all ``OK``.

   A ``WARN`` item in ``areno check`` (unlike ``FAIL``) doesn't affect basic training. A WARN on FlashAttention means "inference will be slower, but functionality is fine".

Also check the environment report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno env --json

This prints full Python/PyTorch/CUDA/GPU version info — useful for pasting into an issue.

6 Download Ling-3.0-tiny
--------------------------

Download the model weights
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Option 1: ModelScope (recommended in China; fast)
   pip install modelscope
   modelscope download \
     --model inclusionAI/Ling-3.0-tiny \
     --local_dir ~/models/Ling-3.0-tiny

   # Option 2: HuggingFace (needs a proxy or a mirror)
   # export HF_ENDPOINT=https://hf-mirror.com
   # huggingface-cli download inclusionAI/Ling-3.0-tiny --local-dir ~/models/Ling-3.0-tiny

.. admonition:: Download size

   About 15 GB (FP8 format). With ModelScope in China this usually takes 5-15 minutes.

If you don't specify ``--local_dir``, ModelScope caches the model under ``~/.cache/modelscope/models/inclusionAI--Ling-3.0-tiny/snapshots/master/`` — remember this path, because you'll use it for ``--ckpt`` later.

Verify download integrity
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ls ~/models/Ling-3.0-tiny/

At minimum you should see the following files:

.. code-block:: text

   config.json            # model config (layer count, hidden dims, MoE settings, etc.)
   tokenizer.json         # tokenizer vocabulary
   tokenizer_config.json  # tokenizer config (incl. chat template)
   model*.safetensors     # model weight files (FP8 format)

If ``tokenizer_config.json`` is missing, AReno can't apply the chat template correctly (which breaks the thinking-mode toggle).

7 Smoke Test: Validate the Full Pipeline in Two Steps
-------------------------------------------------------

Before starting real training, AReno provides two levels of smoke test:

.. list-table::
   :header-rows: 1
   :widths: 18 30 40 12

   * - Level
     - Command
     - What it does
     - Duration
   * - **smoke-infer**
     - ``--smoke-infer``
     - Loads the model + allocates KV cache + initializes CUDA Graph, **without actually generating tokens**
     - ~30 sec
   * - **smoke-train**
     - ``--smoke-infer`` + real training args
     - Full rollout → reward → loss → backward → optimizer.step()
     - 1-2 min

It's recommended to run smoke-infer first (it quickly rules out loading/memory problems); once it passes, run the full training.

7.1 Step 1: smoke-infer (quick check)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd ~/AReno
   source ~/areno-env.sh
   source ~/areno-env/bin/activate

   areno train \
     --ckpt ~/models/Ling-3.0-tiny \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --algo gspo \
     --attn-backend flash \
     --smoke-infer \
     --batch-size 1 \
     --max-running-prompts 1 \
     --max-prompt-tokens 64 \
     --max-new-tokens 16 \
     --tp-size 1 \
     --world-size 1

Success marker:

.. code-block:: text

   AReno smoke infer ok: tp_size=1, batch_size=1, n_samples=1, mini_bs=1,
   max_running_prompts=1, adam_8bit=False, drop_rollout_state=True, peak_mem_frac=0.1996

``peak_mem_frac=0.1996`` means peak memory used only ~20% (about 25 GB) — very safe in the GB10's 121 GB pool.

If you fail at this stage (``Killed`` or OOM), there's a more fundamental problem — check whether other processes are eating the unified memory.

7.2 Step 2: Real training (minimal config)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once smoke-infer passes, run one real GSPO training step — including rollout (generating real tokens), reward computation, backprop, and an optimizer step:

.. code-block:: bash

   areno train \
     --ckpt ~/models/Ling-3.0-tiny \
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

.. warning:: Three key parameters introduced this chapter

   .. list-table::
      :header-rows: 1
      :widths: 35 65

      * - Parameter
        - Why you must add it
      * - ``--adam-8bit``
        - 🔑 **Without this, FP32 master optimizer states would take ~95 GB of unified memory, and the 8B model would OOM outright.** 8bit Adam compresses the optimizer state to ~12 GB, so it passes safely.
      * - ``--max-running-prompts 1``
        - 🔑 **The default is 64, which preallocates KV cache for 64 concurrent generations and blows up unified memory immediately.** Setting it to 1 is the minimum safe value.
      * - ``--max-prompt-tokens 64 --max-new-tokens 16``
        - Limits sequence lengths to further reduce KV cache memory usage

Parameter-by-parameter explanation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Parameter
     - Purpose
     - Why it's set this way
   * - ``--ckpt ~/models/Ling-3.0-tiny``
     - Model weights path
     - The model you just downloaded
   * - ``--dataset-path gsm8k:main``
     - Dataset
     - GSM8K math problems; AReno automatically downloads them from ModelScope
   * - ``--dataset-loader-fn ...``
     - Dataset loader function
     - Converts raw data into prompt format
   * - ``--reward-fn-path ...``
     - Reward function
     - Compares the model output against the standard answers
   * - ``--algo gspo``
     - Algorithm
     - GSPO — the simplest entry into online RL
   * - ``--attn-backend flash``
     - Attention backend
     - Uses flash-attn to speed things up (switch to ``sdpa`` if the build failed)
   * - ``--adam-8bit``
     - 8bit optimizer
     - 🔑 Saves ~80 GB of optimizer state memory
   * - ``--batch-size 1``
     - Batch size
     - 1 sample — minimizes memory
   * - ``--max-running-prompts 1``
     - Concurrent inference count
     - 🔑 Minimizes KV cache preallocation
   * - ``--max-prompt-tokens 64``
     - Max prompt length
     - Limits sequence length
   * - ``--max-new-tokens 16``
     - Max generation length
     - Limits sequence length
   * - ``--temperature 1``
     - Sampling temperature
     - Default value; controls generation randomness
   * - ``--tp-size 1``
     - Tensor parallelism
     - Single GPU
   * - ``--world-size 1``
     - Process count
     - Single process

Expected output (section by section)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 50 50

   * - ``[init] Loading checkpoint from ~/models/Ling-3.0-tiny...``
     - → Loads the model weights, tokenizer, and model config
   * -
     - → Initializes the Worker processes

Seeing this line means AReno recognized the checkpoint format correctly.

.. code-block:: text

   [rollout] Generating completions for 1 prompts × 1 samples...
             → sample one completion for the prompt with the current model

This is real token generation (unlike smoke-infer). The model will generate answers to GSM8K math problems.

.. code-block:: text

   [reward] Scoring 1 completions...
            → call math_verify_reward to score each completion

The reward function extracts the ``\boxed{...}`` content from the model output and compares it with the standard answer. **A score of 0 during the smoke test is completely normal** — an untrained model failing math problems is expected behavior.

.. list-table::
   :widths: 50 50

   * - ``[train] step=0, loss=..., lr=..., tokens=...``
     - → Computes the GSPO loss and performs one gradient update
   * -
     - → Includes backward + optimizer.step()

Seeing ``train_stats={loss=..., ...}`` means **a full training step executed successfully**. This is the main success marker. If this step triggers ``torch.OutOfMemoryError`` (especially at ``optimizer.step()``), it means you forgot ``--adam-8bit``.

.. warning:: loss=0.0 and grad_norm=0.0 in the smoke test are both normal

   If the model gets everything wrong (reward=0), GSPO's advantage is also 0, so gradients are naturally 0 and the model weights don't update. This doesn't mean training is broken — it just means "the model hasn't learned to solve these problems yet". Real training needs more data and samples to produce a learning signal.

Success marker
~~~~~~~~~~~~~~

When the whole command finishes, the last line will look something like this (no traceback, no ``CUDA error``, no ``Killed``):

.. code-block:: text

   [INFO] Training complete.

.. warning:: Checkpoint saving warning

   If you don't add ``--save-path``, the log will tell you ``checkpoints will not be saved``, and the trained weights won't be persisted. That's fine during validation, but for real training make sure to add ``--save-path ./checkpoints``.

8 Quick Reference for Common Problems
---------------------------------------

Build-related
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Problem
     - Fix
   * - ``areno_accel`` build fails
     - Rebuild with ``ARENO_BUILD_EXT=1 MAX_JOBS=4 pip install -e . --no-build-isolation``
   * - ``Python.h: No such file``
     - In a no-root environment, manually extract python3-dev and set ``CPATH`` per Step 0-A in Section 3
   * - flash-attn build ``Killed``
     - You must set ``FLASH_ATTN_CUDA_ARCHS="120"`` (it doesn't read ``TORCH_CUDA_ARCH_LIST``)
   * - flash-attn pip can't find an ARM64 wheel
     - Build from source: ``pip install flash-attn --no-build-isolation`` (10-20 minutes)
   * - Want to skip the CUDA build (CLI check only)
     - ``ARENO_BUILD_EXT=0 pip install -e .`` — but training needs a GPU, so it can't be skipped

Runtime-related
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Problem
     - Fix
   * - ``torch.OutOfMemoryError`` (during training)
     - 🔑 Add ``--adam-8bit``. FP32 master optimizer states are ~95 GB; 8bit compresses them to ~12 GB
   * - ``Killed`` (killed mid-run by the system)
     - Unified memory is saturated by other processes. Check memory with ``free -h``, find the culprit with ``ps aux --sort=-%mem``. Add ``--max-running-prompts 1`` to reduce AReno's memory usage
   * - PyTorch warning ``compute capability 12.1 not supported``
     - Harmless; ``areno check`` still goes all green and training works in practice
   * - ``ModuleNotFoundError: No module named 'areno'``
     - Virtual environment not activated: ``source ~/areno-env/bin/activate``
   * - Environment variables not taking effect
     - New terminals on the DGX Spark don't inherit previously exported variables. Source ``~/areno-env.sh`` in every new terminal
   * - Training runs but is slow
     - Check whether FlashAttention is installed: ``areno check``
   * - Model download is slow
     - Use ModelScope instead of HuggingFace

Why unified memory: zero copy for RL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a normal PC, CPU memory and GPU VRAM are two separate pools connected by PCIe (~32 GB/s). Every step of the RL loop crosses that bus: loading weights (SSD → CPU → VRAM), shipping rollout results to the CPU for reward computation, and writing checkpoints back out. Those copies are pure overhead.

The DGX Spark packages the CPU and GPU on one substrate with a shared 128 GB LPDDR5x pool (NVLink-C2C ~450 GB/s). Weights load once from SSD and stay visible to both CPU and GPU; rollout results are written by the GPU and read by the CPU with no copy in between. Since the RL loop is essentially alternation between GPU inference/training and CPU reward computation, unified memory turns every alternation from a bus transfer into a pointer read.

.. admonition:: Memory is shared, so watch the pool

   The 128 GB is one pool—not "64 GB CPU + 64 GB GPU". When other processes (Jupyter, Docker, Ray) hold memory, AReno has less headroom. The scenarios below are the practical cases you'll actually hit.

Special notes on unified memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Scenario
     - Note
   * - **Running other services at the same time** (Ray, Jupyter, Docker)
     - They share the 121 GB unified memory pool with AReno. Before training, run ``free -h`` to confirm at least 30 GB is free
   * - **Large memory footprint with default parameters**
     - Many AReno defaults (e.g. ``max_running_prompts=64``) are tuned for standalone GPUs; on a unified memory architecture you need to lower them manually
   * - **Multi-user shared machine**
     - Other users' processes are in the same memory pool too. If OOMs are frequent, coordinate resource usage or pick a dedicated time slot

9 Chapter Summary
-------------------

Your DGX Spark is now a usable AReno workstation. Recap of the key steps:

.. list-table::
   :widths: 100

   * - Unbox → confirm aarch64 + GPU + CUDA
   * - → create a virtual environment + install PyTorch
   * - → set up the environment variable template (~/areno-env.sh)
   * - → [no root] manually extract the Python headers + set CPATH
   * - → pip install -e . --no-build-isolation (builds areno_accel)
   * - → FLASH_ATTN_CUDA_ARCHS="120" pip install flash-attn --no-build-isolation
   * - → areno check (all OK)
   * - → download Ling-3.0-tiny (ModelScope)
   * - → --smoke-infer (validates loading/KV cache/CUDA Graph)
   * - → real training (add --adam-8bit --max-running-prompts 1)

**Run these every time you open a new terminal**:

.. code-block:: bash

   source ~/areno-env.sh
   source ~/areno-env/bin/activate

**Next steps**:

- If you want to **understand what each smoke test parameter means and what each line of output represents** → jump to **First Train**
- If you want to **run on an Alibaba Cloud GPU instance** → read **Aliyun Cloud** (the two chapters are independent; you can skip it)
- If you want to **start writing RL training code directly** → jump to **RL in One Snippet**, and understand the whole picture of RL from one block of Python code
- If you want to **start real training right away** → don't use the smoke test parameters — see the full parameter list in First Train, and don't forget to add ``--save-path``
