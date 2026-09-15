SFT
===

.. admonition:: Goal

   Understand the math and source implementation of SFT, then run a complete training on the Alpaca dataset with a single command.

.. admonition:: Prerequisite

   RL in One Snippet (the RL picture) and Repository Tour.

.. admonition:: Outcome

   ``sft_loss_fn`` source deep read → SFT Trainer call chain → Alpaca hands-on command + ``dataset_loader`` pattern.

1 What Is SFT?
----------------

**SFT = Supervised Fine-Tuning**. The most direct post-training method: show the model a prompt + response, and maximize the next-token probability of the response part.

.. list-table::
   :header-rows: 1
   :widths: 26 34 46

   * - Feature
     - SFT
     - Online RL (GSPO/GRPO/PPO)
   * - Needs rollout?
     - ✗
     - ✓
   * - Needs a reward function?
     - ✗
     - ✓
   * - Needs advantage computation?
     - ✗
     - ✓
   * - Training speed
     - Fast (no inference overhead)
     - Slow (rollout takes >80%)

**Why learn SFT first?** The SFT training skeleton (forward → loss → backward → step) is shared by all RL algorithms. RL simply adds rollout, reward, and advantage on top of this skeleton.

2 Mathematical Form
---------------------

A language model is a conditional probability distribution:

.. math::

   p_\theta(a_t \mid s_{<t})

- :math:`\theta`: model parameters, :math:`s_{<t}`: the already-generated context, :math:`a_t`: the token you want to predict next

SFT's loss: only the mean negative log-likelihood over the response-part tokens.

.. math::

   \mathcal{L}_{\text{SFT}} = -\frac{1}{|\text{response}|} \sum_{t \in \text{response}} \log p_\theta(a_t \mid s_{<t})

The more "certain" the model is about the correct answer (probability → 1), the closer the loss gets to 0.

3 The TrainSequence Fill Logic
--------------------------------

SFT and RL share the ``TrainSequence`` data structure. SFT zero-fills the fields it does not need:

.. code-block:: python

   areno.api.TrainSequence(
       prompt_mask=prompt_mask,   # [True]*len(prompt) + [False]*len(response)
       tokens=tokens,            # prompt_tokens + response_tokens
       logprobs=zeros,           # 全 0.0 — SFT 不需要旧策略 logprobs
       advantages=zeros,         # 全 0.0 — SFT 不区分好坏 token
       eos_token_id=eos_token_id,
   )

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Example value
   * - ``tokens``
     - ``[P1, P2, P3, R1, R2, R3, EOS]``
   * - ``prompt_mask``
     - ``[T, T, T, F, F, F, F]``
   * - Annotation
     - ``↑── prompt does not contribute to loss ──↑  ↑── response contributes to loss ──↑``

4 Deep Read of the sft_loss_fn Source
---------------------------------------

Source: ``areno/api/backend/cuda/losses.py::sft_loss_fn``

.. code-block:: python

   def sft_loss_fn(data_pack, logprobs):
       # packed 路径：变长序列拼接成一个大序列，无填充浪费
       if "packed_response_mask" in data_pack:
           response_mask = data_pack["packed_response_mask"].to(device=logprobs.device).bool()
           valid_count = response_mask.sum().clamp_min(1)
           logprob_sum = logprobs[response_mask].sum()

       # padded 路径：补齐到相同长度，简单但有计算浪费
       else:
           response_mask = (~data_pack["prompt_mask"][:, 1:]).to(device=logprobs.device, dtype=logprobs.dtype)
           if "loss_mask" in data_pack:      # agentic RL 可进一步过滤 tool result token
               response_mask = response_mask * data_pack["loss_mask"][:, 1:].to(...)
           valid_count = response_mask.sum().clamp_min(1.0)
           logprob_sum = (logprobs * response_mask).sum()

       loss = -(logprob_sum / valid_count.to(dtype=logprobs.dtype))
       return loss, {"sft_loss": loss.detach(), "sft_target_tokens": valid_count.detach(),
                     "sft_logprob_mean": (-loss).detach()}

**Two key details**:

**The ``prompt_mask[:, 1:]`` one-position shift**: the i-th position of ``logprobs`` predicts the (i+1)-th token. Shifting by one aligns the mask with the "predicted token" positions.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Example value
   * - ``tokens``
     - ``[P1, P2, P3, R1, R2, EOS]``
   * - ``logprobs``
     - ``[lp(P2|P1), lp(P3|P1,P2), lp(R1|...), lp(R2|...), lp(EOS|...)]``
   * - ``p_mask[:, 1:]``
     - ``[T, T, F, F, F]``
   * - Annotation
     - ``↑ predicts P2,P3 (not in loss)  ↑ predicts R1,R2,EOS (in loss)``

**``clamp_min(1.0)`` defensive programming**: prevents division by zero if the response is empty.

5 The Complete SFT Trainer Call Chain
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - Step / Layer
     - Description
   * - CLI: ``areno train --algo sft --ckpt Qwen/Qwen3-0.6B --dataset ...``
     - Command-line entry point
   * - ``SFTTrainer.fit()``
     - Main training entry
   * - ``Trainer.init()``
     - Load tokenizer → create ``Context`` → launch Worker
   * - ``_iter_train_batches()``
     - dataset row → ``TrainSequence`` (``prompt_mask``, ``tokens``, ``zeros``, ``zeros``)
   * - ``Trainer.train(batch, sft_loss_fn)``
     - Enter training
   * - ``Backend → Engine → Worker``
     - Lower-layer execution
   * - ``forward``
     - Compute per-token ``logprobs``
   * - ``sft_loss_fn``
     - Use ``prompt_mask`` to select response tokens → mean ``-logp``
   * - ``backward → optimizer.step()``
     - Backpropagation and parameter update

The SFT main loop is extremely clean — no ``rollout_session``, no ``reward_fn``, no ``compute_group_advantages``.

6 Hands-On: SFT Training on the Alpaca Dataset
------------------------------------------------

6.1 Dataset Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Alpaca <https://huggingface.co/datasets/yahma/alpaca-cleaned>`_ is Stanford's instruction-tuning dataset. Each record contains three fields: ``instruction``, ``input`` (optional), and ``output``:

.. code-block:: json

   {
     "instruction": "Give three tips for staying healthy.",
     "input": "",
     "output": "1. Eat a balanced diet and avoid processed foods...\n2. Exercise regularly..."
   }

6.2 The Dataset Loader Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete implementation at ``examples/sft/alpaca/dataset_loader.py``:

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
       """把 Alpaca 的 instruction/input/output 转成 SFT trainer 的 prompt/response 格式"""
       records = []
       for row in default_loader(dataset_path):
           record = dict(row)
           instruction = str(record["instruction"]).strip()
           input_text = str(record.get("input") or "").strip()
           prompt = f"Instruction: {instruction}\n"
           if input_text:
               prompt += f"Input: {input_text}\n"
           prompt += "Response:"
           records.append({"prompt": prompt, "response": str(record["output"])})
       return records

**Core contract of the SFT dataset loader**: it must produce ``prompt`` (a string) and ``response`` (a string). The dataset's original field names are not constrained — the loader is responsible for the conversion.

6.3 Complete Training Command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno train \
     --algo sft \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path yahma/alpaca-cleaned \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 2 \
     --mini-bs 1 \
     --max-prompt-tokens 128 \
     --max-new-tokens 64

**Parameter notes**:

- ``--algo sft``: select the SFT algorithm — no ``--reward-fn-path`` needed
- ``--ckpt Qwen/Qwen3-0.6B``: a small 0.6B model for fast experiments; you can also switch to Ling-3.0-tiny
- ``--dataset-path yahma/alpaca-cleaned``: the HuggingFace dataset ID
- ``--batch-size 2 --mini-bs 1``: small batches to reduce memory pressure
- ``--max-prompt-tokens 128 --max-new-tokens 64``: Alpaca instructions are usually short

6.4 Expected Output
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   epoch=0 stage=epoch_start
   epoch=0 step=0 role=policy stage=train_start rows=2
   epoch=0 step=0 train_stats={'sft_loss': 2.34, 'sft_target_tokens': 45, 'sft_logprob_mean': -2.34}
   epoch=0 step=0 role=policy stage=train_end
   epoch=0 step=1 role=policy stage=train_start rows=2
   ...
   epoch=9 stage=epoch_end

**Key metrics**:

- ``sft_loss``: goes lower as training progresses — the model gets better and better at "imitating" the response
- ``sft_logprob_mean = -sft_loss``: goes higher as training progresses — the model becomes more "confident" about correct answers

6.5 SFT vs Online RL Speed Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 42 50

   * -
     - SFT (Alpaca, 0.6B)
     - GSPO (GSM8K, 7.9B, n_samples=8)
   * - Per-step time
     - ~0.5s
     - ~40-60s
   * - Bottleneck
     - forward + backward
     - **rollout** takes >80% (generating 8 completions is the slowest part)
   * - Why fast?
     - The data is ready-made; no inference needed
     - The model first has to generate its 8 answers

This explains why the common strategy in real projects is **SFT first → RL to improve** — quickly inject domain knowledge, then fine-tune behavior.

7 Chapter Summary
-------------------

1. **SFT = maximize the next-token likelihood of the response**: :math:`-\frac{1}{N}\sum \log p_\theta`
2. **The keys of ``sft_loss_fn``**: ``prompt_mask[:, 1:]`` offset alignment + ``clamp_min(1.0)`` as defense against empty responses
3. **The SFT call chain**: CLI → SFTTrainer → Engine → Worker → forward → sft_loss_fn → backward → step
4. **Hands-on**: the Alpaca dataset loader does just one thing — converting any format into ``prompt`` + ``response``
5. **SFT is the fastest training method**: no rollout overhead, great for rapid iteration
