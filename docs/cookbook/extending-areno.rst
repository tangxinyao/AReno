Extending AReno
===============

.. admonition:: Goal

   Master AReno's four extension mechanisms — registering a new algorithm, adapting a new model, writing reward functions, and extending CUDA kernels.

.. admonition:: Prerequisites

   Repository Tour, the algorithm fundamentals (GSPO, GRPO, PPO, DPO), Model Adapters, and CUDA Acceleration.

.. admonition:: Outcome

   Be able to add a new algorithm, support a new model family, write a custom reward function, or extend a CUDA kernel from scratch in AReno.

One of AReno's design philosophies is "easy to extend". This chapter covers four main extension scenarios, each with a complete code template and the integration workflow.

1 Registering a New Algorithm
--------------------------------

Core API for Algorithm Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/api/algorithms.py
   register_algorithm(AlgorithmSpec(
       name="my_algo",                    # algorithm name (--algo my_algo on the CLI)
       trainer_cls=MyAlgoTrainer,         # Trainer class (or lazy-loading factory function)
       default_loss_fn=my_algo_loss_fn,   # default loss function
       requires_rollout=True,             # whether rollout is needed (True for online RL)
       loss_fn_factory=bind_my_algo_loss, # optional: factory that builds the loss from config
       experimental=False,                # True → lives in experimental/
   ))

Incubation Workflow
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Step
     - Content
   * - 1.
     - Create the loss function and Trainer in ``areno/experimental/my_algo/``
   * - 2.
     - Set ``experimental=True`` → auto-discovered at startup
   * - 3.
     - Test and iterate
   * - 4.
     - Move to ``areno/api/trainers/my_algo.py`` once mature
   * - 5.
     - Set ``experimental=False`` → it becomes a formal algorithm

Guide to Writing a Loss Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A minimal loss function:

.. code-block:: python

   # areno/experimental/my_algo/loss.py
   import torch

   def my_algo_loss_fn(data_pack: dict, logprobs: torch.Tensor) -> torch.Tensor:
       """
       Args:
           data_pack: a dict containing response_mask, old_logprobs, advantages, etc.
           logprobs: the current model's next-token logprobs for the training batch
       Returns:
           a scalar loss (torch.Tensor)
       """
       # Get the unified view through response_layout()
       layout = response_layout(data_pack)

       # Compute the token-level importance ratio
       log_ratio = logprobs - layout.old_logprobs  # (current policy / old policy)
       ratio = torch.exp(log_ratio)

       # Clip
       eps = 0.2
       clipped = torch.clamp(ratio, 1 - eps, 1 + eps)

       # PPO-style surrogate loss
       loss_per_token = -torch.min(
           ratio * layout.advantages,
           clipped * layout.advantages
       )

       # Average only over response tokens
       masked = loss_per_token * layout.response_mask
       return masked.sum() / layout.valid_count

Key points:
- **Use ``response_layout()``** rather than reading the data_pack fields directly (guarantees padded/packed compatibility)
- **Where ``old_logprobs`` comes from**: for online RL, the ``logprobs - logprobs.detach()`` surrogate trick keeps the ratio at 1.0 (equivalent to no clipping) — that is what GSPO/GRPO do; for true PPO, ``old_logprobs`` comes from the forward pass at the start of the step
- **The loss must be a scalar**: the training loop expects a single ``torch.Tensor`` scalar

Guide to Writing the Trainer Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/experimental/my_algo/trainer.py
   from areno.api.trainers.policy_only import PolicyOnlyTrainer

   class MyAlgoTrainer(PolicyOnlyTrainer):
       """Trainer for a custom algorithm, reusing PolicyOnlyTrainer's rollout + reward + train loop"""

       # In most cases you only need to inherit PolicyOnlyTrainer
       # The core loop is already implemented in the parent class:
       # for batch in dataloader:
       #     completions = rollout_batch(prompts)
       #     rewards = reward_fn(completions)
       #     advantages = compute_group_advantages(rewards)
       #     train(batch, my_algo_loss_fn)

       # If you need special training logic (such as PPO's multi-role setup), see the PPOTrainer implementation

2 Adapting a New Model
-------------------------

Full Checklist
~~~~~~~~~~~~~~

Adapting a new model family requires implementing the following:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Step
     - Content
   * - 1.
     - Create the ``areno/models/my_model/`` directory
   * - 2.
     - Implement ``adapter.py`` → a ``ModelAdapter`` subclass
   * - 3.
     - Implement ``modeling.py`` → model construction logic (optional: simple models can reuse existing layers)
   * - 4.
     - Implement ``checkpoint.py`` → layout mapping for weight loading/saving
   * - 5.
     - Register in ``areno/models/__init__.py``

Minimal Adapter Template
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/models/my_model/adapter.py
   from pathlib import Path
   from typing import Any
   import torch
   from torch import nn
   from areno.engine.config import ModelConfig
   from areno.models.base import ModelAdapter

   class MyModelAdapter(ModelAdapter):
       name = "my_model"

       def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
           return hf_config.get("model_type") == "my_model"

       def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
           return ModelConfig(
               model_type=self.name,
               vocab_size=hf_config["vocab_size"],
               hidden_size=hf_config["hidden_size"],
               num_layers=hf_config["num_hidden_layers"],
               num_attention_heads=hf_config["num_attention_heads"],
               num_kv_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
               intermediate_size=hf_config["intermediate_size"],
               head_dim=hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"]),
               # ... map all required fields
           )

       def build(self, config: ModelConfig) -> nn.Module:
           # Simplest approach: reuse AReno's generic Transformer construction
           from areno.engine.layers import build_transformer
           return build_transformer(config)

       def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
           from safetensors import safe_open
           from pathlib import Path
           import json

           path = Path(model_path)
           # 1. find the weight files
           index_path = path / "model.safetensors.index.json"
           if index_path.exists():
               index = json.loads(index_path.read_text())
               weight_files = set(index["weight_map"].values())
           else:
               weight_files = ["model.safetensors"]

           # 2. load each file and map it onto the model parameters
           model_state = model.state_dict()
           for wf in weight_files:
               with safe_open(path / wf, framework="pt", device="cpu") as f:
                   for key in f.keys():
                       if key in model_state:
                           model_state[key].copy_(f.get_tensor(key))

       def save_weights(self, model: nn.Module, output_path: str | Path,
                        source_path: str | Path | None) -> str | None:
           # Simplest implementation: save with PyTorch
           # A production implementation should write safetensors shards
           torch.save(model.state_dict(), str(Path(output_path) / "pytorch_model.bin"))
           return "pytorch_model.bin"

The Key Challenge: Weight Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HuggingFace and AReno may name weights differently:

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - HuggingFace
     - AReno
     - Notes
   * - ``model.layers.0.self_attn.q_proj.weight``
     - ``layers.0.attention.wq.weight``
     - Different naming conventions
   * - ``model.layers.0.mlp.gate_proj.weight``
     - ``layers.0.mlp.wg.weight``
     - Needs mapping
   * - Fused QKV
     - Stored separately
     - Some HF models fuse QKV into one matrix

The declarative checkpoint-spec system in ``areno/engine/checkpoints/common.py`` (1300 lines) was designed exactly for this problem: declare the weight layout with ``CheckpointSpec`` and skip the endless hand-written ``for key in state_dict`` mapping.

Registration
~~~~~~~~~~~~

.. code-block:: python

   # areno/models/__init__.py
   from areno.models.my_model.adapter import MyModelAdapter
   from areno.models.registry import register_adapter

   def register_models():
       register_adapter(MyModelAdapter())
       # ... other registered adapters

3 Writing Reward Functions
-----------------------------

The Contract
~~~~~~~~~~~~

.. code-block:: python

   def reward_fn(example, completions: list[str]) -> list[float]:
       """
       Args:
           example: the raw data item produced by the dataset loader (may include ground_truth)
           completions: n_samples model generations for the same prompt
       Returns:
           a scalar reward per completion; length must == len(completions)
       """

Template: Rule-Based Rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # examples/my_task/my_reward.py
   import re

   def reward_fn(example, completions: list[str]) -> list[float]:
       """Compare the model output against the reference answer"""
       answer = example.get("answer", "")
       rewards = []
       for completion in completions:
           # extract the answer from the model output
           match = re.search(r'\\boxed{(.+?)}', completion)
           if match:
               pred = match.group(1).strip()
               score = 1.0 if pred == answer else 0.0
           else:
               score = 0.0  # no answer extracted
           rewards.append(score)
       return rewards

Template: Model-Based Rewards (LLM-as-Judge)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def reward_fn(example, completions: list[str]) -> list[float]:
       """Score with an LLM serving as judge"""
       from openai import OpenAI

       client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
       prompt = example.get("prompt", "")
       reference = example.get("reference", "")

       rewards = []
       for completion in completions:
           response = client.chat.completions.create(
               model="policy",
               messages=[{
                   "role": "system",
                   "content": "You are a judge. Rate the following response on a scale of 1-10. Output only the number."
               }, {
                   "role": "user",
                   "content": f"Prompt: {prompt}\n\nReference: {reference}\n\nResponse: {completion}\n\nRating:"
               }]
           )
           try:
               score = float(response.choices[0].message.content.strip()) / 10.0
           except ValueError:
               score = 0.5  # default for unparseable output
           rewards.append(score)
       return rewards

Debugging Tips
~~~~~~~~~~~~~~

- **Test the reward function with GSOP/GRPO first** (simpler than PPO, no critic needed)
- **Add logging**: ``print(f"prompt={...}, reward={rewards}")`` (AReno captures the Worker's stdout)
- **Check the reward distribution**: normally there should be a clear split between positive and negative; all 0s or all 1s means the reward function is ineffective

4 Extending CUDA Kernels
---------------------------

Development Workflow
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   1. Create my_kernel.cu in areno/accel/csrc/
   2. Register the Python binding in areno/accel/csrc/extension.cpp
   3. Create my_kernel.py (Python wrapper) in areno/accel/
   4. Recompile: pip install -e . or python setup.py build_ext --inplace
   5. Integrate the call in the upper-layer code

CUDA Kernel Template
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cuda

   // areno/accel/csrc/my_kernel.cu
   #include <torch/extension.h>
   #include <cuda_runtime.h>

   template <typename scalar_t>
   __global__ void my_fused_kernel(
       const scalar_t* __restrict__ input,
       scalar_t* __restrict__ output,
       const int n,
       const int d
   ) {
       const int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= n) return;

       // your core computation logic
       // e.g. an element-wise fused op
       scalar_t val = input[idx];
       output[idx] = val * val;  // example
   }

   torch::Tensor my_fused_op(
       torch::Tensor input
   ) {
       TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
       TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

       const int n = input.numel();
       auto output = torch::empty_like(input);

       const int threads = 256;
       const int blocks = (n + threads - 1) / threads;

       AT_DISPATCH_FLOATING_TYPES_AND_HALF(
           input.scalar_type(), "my_fused_op", ([&] {
               my_fused_kernel<scalar_t><<<blocks, threads>>>(
                   input.data_ptr<scalar_t>(),
                   output.data_ptr<scalar_t>(),
                   n, 1  // example parameters
               );
           })
       );

       return output;
   }

C++ Bridge
~~~~~~~~~~

.. code-block:: cpp

   // add to areno/accel/csrc/extension.cpp:
   #include "my_kernel.cu"  // or compile it separately

   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       // existing registrations ...
       m.def("my_fused_op", &my_fused_op, "My fused operation");
   }

Python Wrapper Template
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/accel/my_kernel.py
   import torch
   from areno.accel._extension import _get_extension

   @torch._dynamo.disable  # keep out of torch.compile
   def areno_my_fused_op(x: torch.Tensor) -> torch.Tensor:
       """Fused operation: one line describing what it does"""
       if not x.is_cuda:
           # Fall back to a PyTorch implementation
           return x * x  # pure-PyTorch equivalent

       ext = _get_extension()
       return ext.my_fused_op(x.contiguous())

Faster Compilation
~~~~~~~~~~~~~~~~~~

- **ccache**: after installing ``ccache``, set ``export CMAKE_CUDA_COMPILER_LAUNCHER=ccache`` for a 10x speedup on repeated builds
- **MAX_JOBS**: ``export MAX_JOBS=8`` limits parallel compilation (avoids running out of memory on ARM64)
- **Incremental compilation**: change only one kernel → ``python setup.py build_ext --inplace`` recompiles just the changed files

Simplest Integration: Use Triton Instead of CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For simple kernels, Triton is easier to develop than native CUDA:

.. code-block:: python

   # areno/accel/kernels/my_triton_kernel.py
   import torch
   import triton
   import triton.language as tl

   @triton.jit
   def my_fused_kernel(
       x_ptr, y_ptr, output_ptr,
       n_elements,
       BLOCK_SIZE: tl.constexpr,
   ):
       pid = tl.program_id(0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       output = x * tl.sigmoid(x) + y  # your computation
       tl.store(output_ptr + offsets, output, mask=mask)

   def my_triton_fused_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
       output = torch.empty_like(x)
       n_elements = x.numel()
       grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
       my_fused_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
       return output

Triton's advantages: pure Python syntax, auto-tuning, no C++ toolchain, and it can be plugged straight into ``areno/accel/ops.py``. AReno's existing Triton kernels (fused_moe, group_rmsnorm, seg_la, kda_fla) serve as reference templates.
