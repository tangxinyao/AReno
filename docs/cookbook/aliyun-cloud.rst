Aliyun Cloud
============

.. note:: TL;DR

   Create a GPU instance in the Alibaba Cloud ECS console → SSH in → install CUDA + PyTorch → install AReno → download the model → run a smoke test. On x86_64 the build is much simpler (flash-attn has a prebuilt wheel), and the whole process should take 20-40 minutes. Pay-as-you-go — release the instance once training is done, so you don't waste money.

Ling-3.0-tiny presented two hardware options. If you don't have a DGX Spark, or you're just doing short-term learning and don't want a one-time investment, an Alibaba Cloud GPU instance is the better choice.

Compared with the DGX Spark path in DGX Spark, the cloud approach has two clear advantages:

1. **x86_64 architecture**: almost every Python package has a prebuilt wheel, so you skip the ARM64 build misery (no missing ``Python.h``, no 330 flash-attn compilations)
2. **root access**: it's your own instance; you can install anything — no manual header extraction

There's one difference to keep in mind: **the A10 has only 24 GB of VRAM** (not the DGX Spark's 128 GB unified memory), so you need to be more careful with parameter configuration.

1 Create a GPU Instance
-------------------------

Enter the ECS console
~~~~~~~~~~~~~~~~~~~~~

1. Log in to the `Alibaba Cloud console <https://ecs.console.aliyun.com/>`_
2. Left menu → **Instances & Images** → **Instances**
3. Click **Create Instance**

Key configuration items
~~~~~~~~~~~~~~~~~~~~~~~

Every configuration step during creation. **Bold items are the easy-to-trip-on ones**:

Basic configuration
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Configuration item
     - Choice
     - Notes
   * - **Billing method**
     - **Pay-as-you-go**
     - Release it once training is done to save money. Don't choose subscription
   * - **Region**
     - Hangzhou/Shanghai/Beijing/Shenzhen
     - Prefer regions with GPU stock. GPU instances are often out of stock; if one region doesn't have one, switch to another
   * - **Instance type**
     - See the two options below
     - —
   * - **Image**
     - Ubuntu 22.04
     - Usually ships with the NVIDIA driver

Two GPU options
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Budget (recommended to start)
     - Advanced
   * - **Spec**
     - ``ecs.gn7i-c16g1.4xlarge``
     - ``ecs.gn8is`` (single GPU)
   * - **GPU**
     - 1× NVIDIA A10
     - 1× NVIDIA H20
   * - **VRAM**
     - 24 GB GDDR6
     - 96 GB HBM3
   * - **vCPU**
     - 16
     - varies by spec
   * - **Reference price**
     - ~¥12-15/hour
     - ~¥35-40/hour
   * - **Best for**
     - Ling-3.0-tiny FP8 standard training
     - Larger batch / BF16 training / larger models

.. warning:: GPU instances often run out of stock

   If your preferred region has no stock, try Hangzhou → Shanghai → Beijing → Shenzhen in turn. If none of the four regions has an A10, you may need to switch to an H20 or wait for stock to refresh (usually around the top of the hour).

Storage
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Configuration item
     - Choice
     - Notes
   * - **System disk**
     - ESSD cloud disk, 100 GB
     - Model ~15 GB + dependencies ~10 GB + checkpoints + system = 100 GB is enough
   * - Data disk
     - Not needed
     - Training data isn't large; no extra disk needed

Network and security group
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Configuration item
     - Choice
     - Notes
   * - **Public IP**
     - **Assign a public IPv4 address**
     - Without a public IP you can't SSH in
   * - **Security group**
     - Create a new one or pick an existing one
     - Open the following ports inbound

**Security group inbound rules** (configured when creating the instance, or modified after creation):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Port
     - Protocol
     - Purpose
   * - **22**
     - TCP
     - SSH connection (required)
   * - **8000**
     - TCP
     - API port for ``areno serve``
   * - **8888**
     - TCP
     - JupyterLab (optional)

.. note::

   If the security group doesn't open port 22, you'll find SSH connections failing once the instance is created — this is the most common reason for "created successfully but can't use it".

Other
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Configuration item
     - Choice
   * - Login credentials
     - **Key pair** (recommended; more secure than a password) or **password**
   * - Instance name
     - Anything, e.g. ``areno-train``

Confirm creation
~~~~~~~~~~~~~~~~

Click **Confirm Order** and wait for the instance to be created (usually 1-2 minutes). Once it's ready, note down the **public IP address**.

2 SSH Connection and Base Environment
---------------------------------------

SSH connection
~~~~~~~~~~~~~~

.. code-block:: bash

   # If logging in with a password
   ssh root@<public-IP>

   # If using a key pair
   ssh -i ~/.ssh/your-key.pem root@<public-IP>

The first connection will prompt you to confirm the host fingerprint — type ``yes``.

Confirm the GPU
~~~~~~~~~~~~~~~

.. code-block:: bash

   nvidia-smi

You should see an NVIDIA A10 or H20, with driver version ≥ 535. If you get ``nvidia-smi: command not found``, the image didn't preinstall the driver — switch to an image that ships one, or install the NVIDIA driver manually.

Install the CUDA Toolkit (if the image didn't preinstall it)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most Ubuntu 22.04 GPU images already include CUDA. Check first:

.. code-block:: bash

   nvcc --version

If it says ``command not found``, install it manually:

.. code-block:: bash

   wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
   sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent --override

   # Add to the environment variables
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
   source ~/.bashrc

Install system dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo apt update && sudo apt install -y build-essential git cmake

Create a virtual environment and install PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create a virtual environment
   python3 -m venv ~/areno-env
   source ~/areno-env/bin/activate

   # On x86_64, install the prebuilt package directly (much faster than ARM64)
   pip install torch --index-url https://download.pytorch.org/whl/cu124

Verify PyTorch
~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

Expected:

.. code-block:: text

   PyTorch: 2.6.x
   CUDA available: True
   GPU: NVIDIA A10  (or NVIDIA H20)

3 Install AReno
-----------------

Installation on x86_64 is much simpler than the ARM64 flow in DGX Spark — most packages have prebuilt wheels, so you don't need to compile from source.

Set environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~

Simpler than on the DGX Spark: on x86_64 you usually don't need ``CPATH`` (with root you can run ``apt install python3-dev`` directly) and you don't need ``TRITON_PTXAS_PATH`` (Triton's bundled ptxas supports the common x86_64 architectures):

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda
   export PATH=/usr/local/cuda/bin:$PATH
   export TORCH_CUDA_ARCH_LIST="8.6"   # A10 = sm_86. If using H20, change to "9.0"
   export MAX_JOBS=8

.. admonition:: GPU architecture reference

   - A10 = Ampere, compute capability **8.6**
   - H20 = Hopper, compute capability **9.0**
   - Not sure about your GPU architecture? ``nvidia-smi --query-gpu=compute_cap --format=csv``

Clone the repository and install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno

   # Official install script (use it directly if you have root)
   bash scripts/install.sh

This script will:

1. Install system dependencies (``build-essential``, ``cmake``, etc.)
2. Install Python dependencies (including flash-attn — on x86_64 there's a prebuilt wheel, so it finishes in seconds)
3. Compile the ``areno_accel`` CUDA extensions
4. ``pip install -e .``

.. note::

   Key difference from DGX Spark: on x86_64 **flash-attn has a prebuilt wheel**, and ``pip install flash-attn`` finishes in seconds. No need to set ``FLASH_ATTN_CUDA_ARCHS``, no source compilation, no OOM worries.

Verify the environment
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   source ~/areno-env/bin/activate
   areno check

Expect all ``OK``. If a recommended item (e.g. flash-linear-attention) shows ``WARN``, it doesn't affect basic training.

4 Download Ling-3.0-tiny
--------------------------

.. code-block:: bash

   # ModelScope (recommended in China)
   pip install modelscope
   modelscope download \
     --model inclusionAI/Ling-3.0-tiny \
     --local_dir ~/models/Ling-3.0-tiny

   # Or a HuggingFace mirror
   # export HF_ENDPOINT=https://hf-mirror.com
   # huggingface-cli download inclusionAI/Ling-3.0-tiny --local-dir ~/models/Ling-3.0-tiny

Alibaba Cloud ECS is in China, so ModelScope downloads are usually fast (lots of bandwidth between cloud hosts). About 15 GB; expect 3-8 minutes.

Verify the download:

.. code-block:: bash

   ls ~/models/Ling-3.0-tiny/
   # Should contain: config.json tokenizer.json tokenizer_config.json model*.safetensors

5 JupyterLab Remote Development (Optional)
--------------------------------------------

If you prefer writing Python in a browser rather than using vim/nano in a terminal, you can start a JupyterLab on the GPU instance:

On the GPU instance
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install jupyterlab
   jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

The terminal prints a ``http://127.0.0.1:8888/lab?token=...`` link — note down that token.

On your local machine
~~~~~~~~~~~~~~~~~~~~~

Open a new terminal and set up an SSH tunnel:

.. code-block:: bash

   ssh -L 8888:localhost:8888 root@<public-IP>

Then open ``http://localhost:8888`` in a browser and enter the token.

.. warning:: Security reminder

   When ``--allow-root`` is combined with ``--ip 0.0.0.0``, make sure the security group opens port 8888 only to your own IP (or access it through an SSH tunnel instead of exposing the port directly).

6 Smoke Test Validation
-------------------------

A10 (24 GB) notes
~~~~~~~~~~~~~~~~~

The A10 has only 24 GB of VRAM. Ling-3.0-tiny FP8 peaks at only ~8.34 GiB (see Ling-3.0-tiny), but some of AReno's default parameters are designed for larger VRAM. **You must lower the following parameters manually**:

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 30

   * - Parameter
     - Default
     - Recommended on A10
     - Reason
   * - ``--adam-8bit``
     - False
     - **Must add**
     - FP32 master optimizer states ~15 GB, plus the model ~8 GB = 23 GB — the A10 OOMs outright. 8bit compresses it to ~3 GB
   * - ``--max-running-prompts``
     - 64
     - **1**
     - 64 concurrent KV cache allocations put heavy pressure on 24 GB. Set it to 1 first; increase it as needed later

Step 1: smoke-infer (quick memory check)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd ~/AReno
   source ~/areno-env/bin/activate

   areno train \
     --ckpt ~/models/Ling-3.0-tiny \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --algo gspo \
     --smoke-infer \
     --batch-size 1 \
     --max-running-prompts 1 \
     --max-prompt-tokens 64 \
     --max-new-tokens 16 \
     --tp-size 1 \
     --world-size 1

Success marker:

.. code-block:: text

   AReno smoke infer ok: tp_size=1, ..., peak_mem_frac=...

``peak_mem_frac`` should be between 0.6-0.8 (higher than on the DGX Spark, because the A10's VRAM pool is much smaller). If you OOM outright, first check that no other process is using the GPU VRAM (look with ``nvidia-smi``).

Step 2: Real training (GSPO, minimal config)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno train \
     --ckpt ~/models/Ling-3.0-tiny \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --adam-8bit \
     --batch-size 1 \
     --max-running-prompts 1 \
     --max-prompt-tokens 64 \
     --max-new-tokens 16 \
     --temperature 1 \
     --tp-size 1 \
     --world-size 1

Expected output, stage by stage:

.. list-table::
   :widths: 45 55

   * - ``[init] Loading checkpoint...``
     - → model loading OK
   * - ``[rollout] Generating completions...``
     - → token generation OK
   * - ``[reward] Scoring...``
     - → reward computation OK (a 0 score is also normal)
   * - ``[train] step=0, loss=..., ...``
     - → backward + optimizer.step() OK

H20 (96 GB) users
~~~~~~~~~~~~~~~~~

The H20's 96 GB of VRAM is more than enough for Ling-3.0-tiny. You don't need ``--adam-8bit`` (the FP32 masters fit entirely), and ``--max-running-prompts`` can stay at its default. Use one of the A10 commands, picked to match:

.. code-block:: bash

   # H20: you can skip --adam-8bit; all other parameters are the same
   areno train \
     --ckpt ~/models/Ling-3.0-tiny \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --batch-size 1 \
     --max-running-prompts 1 \
     --max-prompt-tokens 64 \
     --max-new-tokens 16 \
     --temperature 1 \
     --tp-size 1 \
     --world-size 1

7 Tips for Saving Money
-------------------------

Pay-as-you-go GPU instances aren't cheap. A few money-saving habits:

Release it when you're done
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most important one. Training finished → back to the ECS console → **release the instance**. Even if you're just "pausing training to grab a meal", 2 hours is ¥24-80. If you experiment often, that adds up to a significant cost.

Stop ≠ release
~~~~~~~~~~~~~~

Alibaba Cloud ECS's "stop instance" and "release instance" are two different things:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Operation
     - Effect
     - Billing
   * - **Stop**
     - Instance shuts down; data is kept
     - **GPU keeps being billed** (pay-as-you-go doesn't support stopping without charge)
   * - **Release**
     - Instance is destroyed; system disk data is lost
     - **All billing stops**

The correct cost-saving flow: training complete → download checkpoints to your local machine or OSS → **release the instance**. Recreate the instance next time you train.

Persist the model weights
~~~~~~~~~~~~~~~~~~~~~~~~~

Don't re-download the 15 GB of model weights every time you start a new instance. Two options:

**Option 1: Store in OSS (recommended)**

.. code-block:: bash

   # Upload to OSS once
   apt install ossutil
   ossutil cp -r ~/models/Ling-3.0-tiny oss://your-bucket/models/Ling-3.0-tiny

   # Download from OSS when you open the next instance
   ossutil cp -r oss://your-bucket/models/Ling-3.0-tiny ~/models/Ling-3.0-tiny

**Cost**: 15 GB × ¥0.12/GB/month ≈ ¥1.8/month + a small download traffic fee. Almost negligible.

**Option 2: Create a custom image**

If your environment configuration is fairly fixed, you can install the model + AReno and bake them into a custom image. Use that image directly the next time you create an instance — out of the box.

Download training checkpoints promptly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training checkpoints live on the system disk. Before releasing the instance:

.. code-block:: bash

   # Download to your local machine
   scp -r root@<public-IP>:~/AReno/checkpoints ./checkpoints-backup/

   # Or upload to OSS
   ossutil cp -r ~/AReno/checkpoints oss://your-bucket/checkpoints/

After you release the instance the system disk **is permanently lost** — there is no way to recover it.

Use auto-tuning to cut trial-and-error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AReno's ``--tune-params`` parameter can automatically probe the optimal batch size and concurrency before real training:

.. code-block:: bash

   areno train \
     ... (other parameters) \
     --tune-params \
     --mem-frac 0.9

This cuts the trial-and-error cost of "OOM caused by bad parameters → rerun → OOM again". On an hourly-billed cloud instance, every OOM rerun is real money.

Common Problem Quick Reference
------------------------------

Instance creation
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Problem
     - Fix
   * - The selected region has no GPU stock
     - Switch regions (Hangzhou → Shanghai → Beijing → Shenzhen). GPU stock changes dynamically; it's more likely to refresh around the top of the hour
   * - Can't SSH in
     - 1) Confirm the security group opens port 22 2) Confirm the instance has a public IP assigned 3) Confirm you used the right key/password
   * - Instance creation fails (insufficient permissions)
     - Some GPU specs require a work order to be approved. Submit the request in the ECS console; it's usually approved within one business day

Runtime
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Problem
     - Fix
   * - ``CUDA out of memory`` (A10)
     - 🔑 Add ``--adam-8bit``. FP32 master optimizer states are the primary cause of OOM
   * - ``CUDA out of memory`` (H20)
     - Basically impossible for Ling-3.0-tiny itself. Check with ``nvidia-smi`` whether another process is using the GPU
   * - ``areno_accel not found``
     - Rebuild with ``MAX_JOBS=4 pip install -e . --no-build-isolation``
   * - Training was interrupted and you want to resume
     - ECS has no checkpoint persistence — if the instance is still running, checkpoints are in ``~/AReno/checkpoints/``; if the instance has been released, they're gone

Cost
~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Problem
     - Note
   * - "How much did I spend this month?"
     - Alibaba Cloud console → Billing Center → bill details; filter by ECS product
   * - "How do I set a budget alert?"
     - Billing Center → Budget Management → set a monthly budget + 80%/100% threshold SMS reminders
   * - "Will I still be charged after the instance is released?"
     - If the system disk (cloud disk) was set to "release with instance", it isn't billed; otherwise you need to manually delete cloud disk snapshots

Chapter Summary
---------------

Your Alibaba Cloud GPU instance is now a usable AReno workstation. Recap of the key steps:

.. list-table::
   :widths: 100

   * - Create an instance in the ECS console (pay-as-you-go, A10/H20, Ubuntu 22.04)
   * - → open 22/8000/8888 in the security group
   * - → SSH in, confirm the GPU
   * - → install CUDA + PyTorch (x86_64 prebuilt, fast)
   * - → git clone + bash scripts/install.sh
   * - → areno check (all OK)
   * - → download Ling-3.0-tiny via ModelScope
   * - → smoke test (A10 must add --adam-8bit)

Compared with DGX Spark (the ARM64 path), the nice things about the cloud x86_64 plan:

- flash-attn **installs in seconds** (prebuilt wheel)
- no need for ``CPATH`` / manually unpacking Python headers
- no ``FLASH_ATTN_CUDA_ARCHS`` single-architecture build
- no worry about other processes grabbing the unified memory

The price is **hourly billing** — remember to release the instance when you're done.

**Next steps**:

- If you want to **understand what each smoke test parameter means and what each line of output represents** → jump to **First Train**
- If you want to **start writing RL training code directly** → jump to **RL in One Snippet**
- If you've already read DGX Spark and have both environments set up → verify with **First Train**, then start learning RL from **RL in One Snippet**
