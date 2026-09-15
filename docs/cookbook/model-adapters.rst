Model Adapters
==============

.. admonition:: Goal

   Understand how AReno natively supports many model families through the ModelAdapter mechanism — the complete conversion chain from HuggingFace's config.json to AReno's internal representation.

.. admonition:: Prerequisites

   Repository Tour, Training Subsystem.

.. admonition:: Outcome

   Understand the design contract of the model adapter layer and know how to add support for a new model family.

AReno is not tied to any specific model family. It supports multiple models through a **model adapter layer**: each model family implements a ``ModelAdapter`` that maps HuggingFace-format checkpoints into AReno's internal representation. Eight model families are currently supported, including Ling-3.0-tiny (the main model used throughout this book).

1 Why Do You Need an Adapter Layer?
--------------------------------------

The Diversity of the HuggingFace Ecosystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although different models all follow the ``config.json`` + ``model.safetensors`` format, their internal structures differ wildly:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Difference axis
     - Examples
   * - Attention mechanism
     - MHA (LLaMA), GQA (Qwen3), MLA (Ling-3.0), KDA (Bailing), SWA (Gemma)
   * - FFN type
     - Dense MLP, MoE (128 experts), Hybrid (partially dense + partially MoE)
   * - Activation function
     - SiLU (LLaMA/Qwen), GeGLU (Gemma)
   * - Positional encoding
     - RoPE, M-RoPE (Qwen3-VL)
   * - Tokenizer
     - Standard tokenizer, multimodal processor
   * - Special features
     - thinking mode (Ling), vision encoder (MiniCPM-V), linear attention

AReno cannot cover every case with a single "universal model class" — that would lead to endless ``if/else`` branches and brittle conditionals.

Design Philosophy of the Adapter Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 25 25 25

   * - HuggingFace config.json
     - ModelAdapter
     - AReno ModelConfig
     - internal nn.Module
   * - HuggingFace weights
     - ModelAdapter
     - AReno internal format
     - GPU tensor

The adapter layer is the "electrical plug adapter" between the two worlds: outward it stays compatible with the HuggingFace ecosystem (any HF checkpoint loads), inward it presents a unified ``ModelConfig`` and ``nn.Module`` interface (the engine layer never sees model differences).

2 The ModelAdapter Contract
------------------------------

``ModelAdapter`` is an abstract base class defined in ``areno/models/base.py`` (76 lines). Every model family must implement 5 core methods:

The Full Contract
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # areno/models/base.py
   class ModelAdapter(ABC):
       name: str  # unique identifier, e.g. "llama", "qwen3"

       @abstractmethod
       def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
           """Decide whether this adapter handles the given HF config (matched via fields such as model_type)"""

       @abstractmethod
       def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
           """Translate the raw HF config dict into an internal ModelConfig (~60 fields)"""

       @abstractmethod
       def build(self, config: ModelConfig) -> nn.Module:
           """Instantiate the nn.Module from a ModelConfig (no weights yet)"""

       @abstractmethod
       def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
           """Load HF-format weights into the model (in-place, no return value)"""

       @abstractmethod
       def save_weights(self, model: nn.Module, output_path: str | Path,
                        source_path: str | Path | None) -> str | None:
           """Save the model's weights back to an HF-compatible checkpoint"""

       def build_policy_plan(self, model: nn.Module):
           """Return the canonical weight layout for policy sync (used for the train→rollout weight transfer)"""

Registration Mechanism
~~~~~~~~~~~~~~~~~~~~~~

Adapters are managed by the registry in ``areno/models/registry.py`` (149 lines):

.. code-block:: python

   # areno/models/registry.py
   register_adapter(adapter)     # add an adapter (duplicate names raise an error)
   adapter_from_hf(model_path)   # read HF config → match adapter
   config_from_hf(model_path)    # adapter match → config translation → ModelConfig
   build_model(config)           # ModelConfig → nn.Module
   load_model_weights(model, config, model_path)  # load weights
   save_model_weights(model, config, output_path, source_path)  # save weights

Key details:
- **Lazy loading**: a model family's Python package is not all imported when ``import areno`` runs; it is loaded only the first time it is needed (``load_model_plugins()``)
- **On-demand loading**: ``_load_model_plugin(model_type)`` loads only the matching model family (avoids importing everything)
- **Match order**: adapters are iterated in registration order; the first one whose ``match_hf_config()`` returns True wins

ModelConfig: One Unified Internal Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ModelConfig`` is a dataclass defined in ``areno/engine/config.py`` with about 60 fields. It uniformly describes the architecture of every model family:

.. code-block:: python

   @dataclass
   class ModelConfig:
       model_type: str              # "qwen3", "llama", ...
       vocab_size: int
       hidden_size: int
       num_layers: int
       num_attention_heads: int
       num_kv_heads: int            # GQA / MLA
       intermediate_size: int       # FFN intermediate dimension
       # MoE-related
       num_experts: int
       num_shared_experts: int
       moe_intermediate_size: int
       # Attention-related
       attn_implementation: str     # "flash_attention_2", "sdpa", "native"
       use_sliding_window: bool
       # Positional encoding
       rope_theta: float
       rope_scaling: dict | None
       # Special features
       enable_thinking: bool        # Ling thinking mode
       multimodal: bool
       # ...

Each family adapter's ``config_from_hf()`` is responsible for translating all HF-specific fields into this unified structure.

3 Adapter Highlights for Ling-3.0-tiny
-----------------------------------------

Ling-3.0-tiny is one of the most complex adaptation cases, because it uses a hybrid attention architecture (KDA + MLA) and a large MoE (128 routed experts + 1 shared expert).

Hybrid Attention: KDA + MLA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Of Ling-3.0-tiny's 28 decoder layers:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Layer range
     - Attention type
     - Characteristics
   * - 3 of every 4 layers
     - KDA (Kimi Delta Attention)
     - Linear attention, O(N) complexity, good for long sequences
   * - 1 of every 4 layers
     - MLA (Multi-head Latent Attention)
     - Standard softmax attention, captures fine-grained dependencies

The adapter must parse the attention-type description in the HF config correctly inside ``config_from_hf()`` and map it onto AReno's ``ModelConfig`` fields. The ``build()`` method picks a different attention implementation per layer based on the config.

Routing the 128 MoE Experts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ling-3.0-tiny activates 8 routed experts per token (top-8):
- Routing logic: ``areno/accel/router.py`` → ``areno_grouped_topk_router()`` (CUDA fused kernel)
- Token reordering: ``areno/accel/moe.py`` → ``areno_moe_topk_permute()`` / ``areno_moe_unpermute()``
- Expert alignment: ``areno/accel/routing.py`` → ``areno_moe_align()``

In ``build()`` the adapter constructs the network structure of the MoE layers; during weight loading, the expert weight matrices in the HF checkpoint are loaded into the MoE layers in the correct layout.

Special Tokenizer Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ling-3.0-tiny natively supports **thinking mode** (the ``enable_thinking`` switch):
- When enabled, the model "thinks" between the two special tokens ``thinking`` and ``response``
- The adapter must set the chat template and special tokens correctly when loading the tokenizer

AReno handles this uniformly through ``load_tokenizer()`` in ``areno/api/tokenizer.py``; the adapter needs to supply the special token IDs from the tokenizer configuration.

4 Overview of Other Supported Models
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 22 25 38

   * - Model family
     - Directory
     - Architecture
     - Key adaptation points
   * - **LLaMA**
     - ``areno/models/llama/``
     - Standard MHA + SiLU + RoPE
     - Simplest adaptation; the "template" for all model families
   * - **Qwen3**
     - ``areno/models/qwen3/``
     - GQA + SiLU + RoPE, Dense/MoE
     - Shared KV head on some layers
   * - **Qwen3.5**
     - ``areno/models/qwen3_5/``
     - Qwen3 + multimodal, Dense/MoE/VL/MoE+VL
     - Vision encoder + M-RoPE positional encoding
   * - **Bailing**
     - ``areno/models/bailing/``
     - MoE + Linear Attention v2
     - Linear attention weight mapping
   * - **Bailing V3**
     - ``areno/models/bailing_v3/``
     - MoE + Hybrid Attention
     - Hybrid attention layer configuration
   * - **Gemma4**
     - ``areno/models/gemma4/``
     - SWA + GeGLU, multimodal
     - defer_lm_head optimization (logits not stored in GPU memory)
   * - **MiniCPM-V-4.6**
     - ``areno/models/minicpmv46/``
     - Vision-language model
     - Vision encoder + multimodal token fusion
   * - **OLMo2**
     - ``areno/models/olmo2/``
     - MHA + SwiGLU
     - Non-standard weight tying configuration

Difficulty Levels for Adapting a Model Family
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - Complexity
     - Example
     - What needs to be done
   * - Low
     - LLaMA variants
     - Adjust ``match_hf_config`` + fine-tune the ``config_from_hf`` field mapping
   * - Medium
     - A new attention mechanism
     - Implement an AReno version of the attention layer + weight mapping
   * - High
     - MoE + multimodal
     - Implement the routing layer + vision encoder + multimodal token fusion

Core Call Chain
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Step
     - Description
   * - ``areno train --ckpt <path-to-model>``
     - CLI entry point ↓
   * - ArenoEngine construction
     - ↓
   * - ``modeling.py: build_model_on_device()``
     - ↓
   * - ``registry.py: adapter_from_hf(model_path)``
     - ``read_hf_config(model_path)`` → dict
       iterate over adapters → ``match_hf_config()``
       return the matching ModelAdapter
       ↓
   * - ``config_from_hf(hf_config)`` → ``ModelConfig``
     - ↓
   * - ``build_model(ModelConfig)`` → ``nn.Module``
     - ↓
   * - ``load_model_weights(model, config, model_path)``
     - End of the chain
