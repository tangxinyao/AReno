Algorithms
=================

.. admonition:: Objective

   Use a 50-line Python SDK snippet to run GSPO training, then map each line to the core RL concepts — without learning RL first, you grasp it naturally while reading the code.

.. admonition:: Prerequisites

   You've completed First Train (you know how to use the ``areno train`` CLI).

.. admonition:: Outcome

   Understand the five core concepts — rollout, reward, advantage, TrainSequence, loss_fn — and how they chain together into the RL training loop.

1 Run the Code First, Then Read the Theory
-----------------------------------------------

AReno's README has a complete Quick Start snippet that uses ~50 lines of Python to run one GSPO training step. Read it through first (you don't need to understand every line; just get a feel for the overall flow):

.. code-block:: python

   import asyncio

   from datasets import load_dataset

   from areno import Trainer
   from areno.api import RewardRecord, SamplingParams, TrainSequence, gspo_loss_fn
   from areno.api.rewards import compute_group_advantages
   from examples.math.math_verify_reward import reward_fn

   async def main():
       # 1. Create the trainer
       trainer = Trainer(world_size=1, model_path="Qwen/Qwen3-0.6B")
       trainer.init()
       tokenizer = trainer.get_tokenizer()

       row = load_dataset("gsm8k", "main", split="train[0:1]")[0]
       target = str(row["answer"]).rsplit("####", 1)[-1].strip()
       prompt = f"Solve the problem and put the final answer in \\boxed{{}}.\n\nProblem: {row['question']}\nSolution:"
       prompt_tokens = tokenizer.encode(prompt)
       sampling = SamplingParams(max_new_tokens=512)

       # 2. Rollout on-policy completions
       async with trainer.rollout_session(sampling_params=sampling, proxy=False):
           sequences = trainer.rollout_token_batch([prompt_tokens], n_samples=8, sampling_params=sampling)[0].sequences

       # 3. Score and normalize rewards within the sample group
       rewards = [
           reward_fn(RewardRecord(prompt=prompt, completion=tokenizer.decode(seq.resp_tokens), answer=[target]))
           for seq in sequences
       ]
       advantages = compute_group_advantages(rewards)

       # 4. Train one step
       batch = [
           TrainSequence(
               tokens=prompt_tokens + seq.resp_tokens,
               logprobs=[0.0] * len(prompt_tokens) + seq.resp_logprobs,
               prompt_len=len(prompt_tokens),
               scalar_advantage=advantage,
               reward=reward,
               eos_token_id=tokenizer.eos_token_id,
           )
           for seq, reward, advantage in zip(sequences, rewards, advantages, strict=True)
       ]
       trainer.train(batch, gspo_loss_fn)

       # 5. Repeat for more prompts, then close
       trainer.close()

   asyncio.run(main())

This code does exactly what the ``areno train`` CLI command in First Train does — but through the SDK, every line exposed in front of you. Now let's break it down section by section.

2 Code → RL Concept Mapping
--------------------------------

First Section: Creating the Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   trainer = Trainer(world_size=1, model_path="Qwen/Qwen3-0.6B")
   trainer.init()
   tokenizer = trainer.get_tokenizer()

What the Code Does
^^^^^^^^^^^^^^^^^^^^

1. ``Trainer(world_size=1, model_path="Qwen/Qwen3-0.6B")`` — creates a trainer that uses 1 GPU, with the model Qwen3-0.6B (you can also swap in Ling-3.0-tiny)
2. ``trainer.init()`` — initializes the backend: loads the tokenizer, launches Worker processes, and loads model weights onto the GPU
3. ``trainer.get_tokenizer()`` — gets the tokenizer, used later to convert text into tokens

Concept Introduced: What Is a "Policy"?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the RL context, a language model is a **policy function**:

.. math::

   \pi_\theta(a_t \mid s_{<t})

Read it as: "given the context :math:`s_{<t}` (the first t-1 tokens already generated), the model with parameters :math:`\theta` outputs the probability distribution of the next token :math:`a_t`."

- **State**: :math:`s_{<t}` — the context generated so far (prompt + generated tokens). In the code, that's ``prompt_tokens``.
- **Action**: :math:`a_t` — the next token to generate. In the code, that's the integer token ID the model decodes on each step.
- **Parameters**: :math:`\theta` — the model weights. Training is fundamentally adjusting :math:`\theta` so that "good actions" get higher probability.
- **Policy**: :math:`\pi_\theta` — the function that maps states to action probabilities. That is the language model itself.

Behind the line ``Trainer(world_size=1, model_path="...")`` is exactly the loading of a policy function with parameters :math:`\theta`.

Architecture Concept: What Does ``init()`` Do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``init()`` is AReno's "master switch". Its call chain is (source ``areno/api/trainer.py::Trainer.init()``):

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Call / Step
     - Description
   * - ``Trainer.init()``
     -
   * - ``→ load_tokenizer(model_path)``
     - loads the tokenizer (text ↔ token conversion)
   * - ``→ load_processor(model_path)``
     - loads the multimodal processor (needed for image/audio models)
   * - ``→ Context(world_size, path, ...)``
     - creates the context object (holds all configuration)
   * - ``→ Backend.initialize(ctx)``
     - starts the backend engine
   * - ``→ Engine launches Worker processes``
     - Worker is the process that actually does inference and training

Simplified view:

- ``Trainer`` — your code
   - ``Backend`` — the backend abstraction layer
      - ``Engine`` — the core engine
         - ``Worker`` — does the actual work

This architecture is developed in detail in Repository Tour. For now, just know that after ``init()`` the model is in place and ready to work.

Second Section: Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   row = load_dataset("gsm8k", "main", split="train[0:1]")[0]
   target = str(row["answer"]).rsplit("####", 1)[-1].strip()
   prompt = f"Solve the problem and put the final answer in \\boxed{{}}.\n\nProblem: {row['question']}\nSolution:"
   prompt_tokens = tokenizer.encode(prompt)
   sampling = SamplingParams(max_new_tokens=512)

What the Code Does
^^^^^^^^^^^^^^^^^^^^

1. Take the first sample from the GSM8K dataset
2. Extract the final answer from the ``answer`` field's ``#### 42`` format (``"42"``)
3. Build the prompt: instruct the model to solve a math problem, with the answer wrapped in ``\boxed{}``
4. Tokenize the prompt → ``prompt_tokens`` (a list of integers)
5. Set the sampling parameters: generate at most 512 tokens

The point of this section is the **data format conversion**. In First Train, the CLI did this automatically via ``--dataset-loader-fn``; with the SDK, you handle it manually.

The complete fields of ``SamplingParams`` (source ``areno/api/models.py``):

.. list-table::
   :header-rows: 1
   :widths: 30 12 70

   * - Field
     - Default
     - Meaning
   * - ``max_new_tokens``
     - ``16``
     - Maximum generation length
   * - ``temperature``
     - ``1.0``
     - Sampling temperature
   * - ``top_p``
     - ``1.0``
     - Nucleus sampling threshold
   * - ``top_k``
     - ``-1``
     - Top-k sampling (-1 disables it)
   * - ``greedy``
     - ``False``
     - Whether to use greedy decoding

Third Section: Rollout (the Model Generates Completions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async with trainer.rollout_session(sampling_params=sampling, proxy=False):
       sequences = trainer.rollout_token_batch([prompt_tokens], n_samples=8, sampling_params=sampling)[0].sequences

What the Code Does
^^^^^^^^^^^^^^^^^^^^

1. ``rollout_session(...)`` — opens a rollout session (prepares the inference engine)
2. ``rollout_token_batch([prompt_tokens], n_samples=8, ...)`` — samples **8 completions** for the same prompt
3. ``[0].sequences`` — takes all 8 results of the first (and only) prompt

Each ``sequence`` contains two fields:

.. list-table::
   :header-rows: 1
   :widths: 25 30 70

   * - Field
     - Type
     - Meaning
   * - ``resp_tokens``
     - ``list[int]``
     - The token sequence generated by the model
   * - ``resp_logprobs``
     - ``list[float]``
     - The log probability of each response token (used later for the importance ratio)

Concept Introduced: What Is Rollout?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Rollout = using the current policy (current weights) to generate completions.**

In RL terminology, rollout is "letting the model work a problem on its own":

- Give the model a prompt ("1+1=?")
- The model generates tokens autoregressively ("2")
- Record the log probability of each step (how probable is generating "2")

**Why sample 8 instead of 1?**

Because RL needs **comparison** — for the same problem, the model may give 8 different answers. Only through comparison can you know "which kind of answer is better".

- ``n_samples=1``: only one completion, nothing to compare → no learning signal
- ``n_samples=8``: the 8 completions compare against each other; good ones get higher reward, bad ones get lower → learning signal
- ``n_samples=64``: comparison is more reliable, but rollout time x64

**Why is it called "on-policy"?**

on-policy = generating data with the **current, latest** policy. This is the fundamental difference between RL and SFT:

- SFT data is "correct answers written by others" (a fixed dataset)
- RL data is "answers generated by the model itself" (changes as the policy updates)

What's the benefit? The model can **explore solutions absent from the training data** — trial and error is itself the learning process.

Architecture Concept: What Happens Behind the Scenes in rollout?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 65 45

   * - Call / Step
     - Description
   * - ``trainer.rollout_token_batch(prompts, n_samples=8)``
     -
   * - ``→ Trainer``
     -
   * - ``→ Backend.rollout_batch()``
     -
   * - ``→ Engine``
     -
   * - ``→ InferenceManager (Continuous Batching)``
     - continuous batching scheduling
   * - ``→ Worker (actual inference)``
     - actual inference
   * - ``→ forward → sample per token → record logprobs``
     -

**Key components:**

- **Continuous Batching**: multiple prompts don't need to wait for each other; whoever finishes first exits first, so GPU utilization is higher
- **Logprob recording**: during rollout, the log probability of each response token — :math:`\log p_\theta(a_t \mid s_{<t})` — is recorded; this is the basis for the later loss computation

Fourth Section: Reward Scoring + Advantage Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   rewards = [
       reward_fn(RewardRecord(prompt=prompt, completion=tokenizer.decode(seq.resp_tokens), answer=[target]))
       for seq in sequences
   ]
   advantages = compute_group_advantages(rewards)

What the Code Does
^^^^^^^^^^^^^^^^^^^^

1. Decode each completion back into text
2. Call ``reward_fn`` to score each one (correct = 1.0, wrong = 0.0)
3. Z-score normalize the 8 rewards within the group → 8 advantages

Concept Introduced: What Is a Reward Function?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A reward function = the function that judges how good a model's output is.**

.. code-block:: text

   Input: (prompt, completion, standard answer)
   Output: a float score (usually 0.0 or 1.0)

In the GSM8K example:

.. code-block:: python

   def reward_fn(record) -> float:
       ground_truth = record.answer[0]      # 标准答案 "42"
       gt_parsed = parse(ground_truth)       # 解析为数学符号
       pred_parsed = parse(record.completion) # 从 \boxed{42} 中提取答案
       return 1.0 if verify(gt_parsed, pred_parsed) else 0.0

- Model outputs ``\boxed{42}`` → extracts "42" → matches the standard answer "42" → 1.0
- Model outputs ``\boxed{43}`` → extracts "43" → no match → 0.0
- Model output doesn't write ``\boxed{}`` → extraction fails → 0.0

**The reward function is RL's "baton"** — it tells the model what is good and what is bad. The design of the reward function directly shapes the behavior the model learns.

Concept Introduced: What Is Advantage?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Advantage = within a "group", how much better (or worse) a completion is than the average.**

``compute_group_advantages`` does within-group Z-score normalization (source ``areno/api/rewards.py``):

.. math::

   A_i = \frac{r_i - \text{mean}(r_1...r_8)}{\text{std}(r_1...r_8) + \epsilon}

A concrete example (8 completions, 3 correct = 1.0 and 5 wrong = 0.0):

.. code-block:: text

   rewards    = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
   mean       = 0.375
   std        = 0.5175

   advantages = [1.208, -0.725, -0.725, 1.208, -0.725, -0.725, 1.208, -0.725]

Interpretation:
- **advantage > 0** (+1.208): this completion is better than the group average → **encourage, make this kind of answer more probable**
- **advantage < 0** (-0.725): this completion is worse than the group average → **suppress, make this kind of answer less probable**

**Why use "in-group comparison" instead of "absolute values"?**

Because different prompts differ in difficulty: a perfect score on "1+1=?" and on "solve a differential equation" should not be compared directly. Within-group normalization naturally solves this — each completion is only compared with other completions of **the same problem**. This design means **no extra critic network is needed**.

Fifth Section: Packing TrainSequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   batch = [
       TrainSequence(
           tokens=prompt_tokens + seq.resp_tokens,
           logprobs=[0.0] * len(prompt_tokens) + seq.resp_logprobs,
           prompt_len=len(prompt_tokens),
           scalar_advantage=advantage,
           reward=reward,
           eos_token_id=tokenizer.eos_token_id,
       )
       for seq, reward, advantage in zip(sequences, rewards, advantages, strict=True)
   ]

What the Code Does
^^^^^^^^^^^^^^^^^^^^

Pack each of the 8 completions into its own ``TrainSequence`` object.

Concept Introduced: Why Is TrainSequence Shaped This Way?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TrainSequence`` is the "unified data format" of AReno's training pipeline. Every field is designed with an RL reason behind it (source ``areno/api/models.py``):

.. list-table::
   :header-rows: 1
   :widths: 22 16 32 78

   * - Field
     - Type
     - Fill Logic
     - RL Meaning
   * - ``tokens``
     - ``list[int]``
     - ``prompt_tokens + resp_tokens``
     - The complete prompt+completion sequence
   * - ``logprobs``
     - ``list[float]``
     - prompt part filled with 0, response part filled with rollout logprobs
     - Used for the importance ratio computation: :math:`\frac{\pi_\theta}{\pi_{\text{old}}}`. The prompt is the condition and doesn't participate in the loss, so it's filled with 0
   * - ``prompt_len``
     - ``int``
     - prompt token count
     - Marks where the prompt ends; the loss function uses it to distinguish the "condition" from the "part to optimize"
   * - ``scalar_advantage``
     - ``float``
     - the within-group normalized A
     - The "good/bad signal" of this completion; the loss function uses it to decide whether to encourage or suppress
   * - ``reward``
     - ``float``
     - the raw score from the reward function
     - kept for logging/analysis
   * - ``eos_token_id``
     - ``int``
     - the tokenizer's EOS token ID
     - marks where the sequence ends

**Key design**: ``prompt_len`` lets the loss function know which tokens are prompt (the condition, not part of the loss) and which are response (the model's output, part of the loss). The prompt part's ``logprobs`` and ``advantage`` are both filled with 0 because they're not needed.

A diagram showing the structure of ``TrainSequence`` at a glance:

.. list-table::
   :header-rows: 1
   :widths: 25 78

   * - Field
     - Value / Annotation
   * - ``tokens:``
     - ``[P1, P2, P3, R1, R2, R3, R4, EOS]``
   * - ``logprobs:``
     - ``[0.0, 0.0, 0.0, -1.2, -0.8, -0.5, -2.1, 0.0]``
   * - ``advantage:``
     - ``(via scalar_advantage=1.208, broadcast to all R tokens)``
   * - Position annotations
     - ``↑── prompt_len=3 ──↑── response (participates in loss) ──↑``
       ``↑── not part of loss ──↑``

Sixth Section: Train (Gradient Update)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   trainer.train(batch, gspo_loss_fn)

What the Code Does
^^^^^^^^^^^^^^^^^^^^

Hand the 8 ``TrainSequence`` objects to the backend and run one forward + backward + optimizer.step(). ``gspo_loss_fn`` is the loss function of the GSPO algorithm.

Concept Introduced: How Does the Loss Function Use Advantages to Update the Policy?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GSPO's loss function does three things (source ``areno/api/algorithms.py::gspo_loss_fn`` → ``areno/api/backend/cuda/losses.py``):

1. **Recompute logprobs**: with the **current** policy, compute :math:`\log \pi_\theta(a_t \mid s_{<t})` for each response token
2. **Compute the importance ratio**: :math:`r_{\text{seq}} = \exp(\frac{1}{N}\sum (\log\pi_\theta - \log\pi_{\text{old}}))` (sequence-level geometric mean)
3. **PPO clip**: :math:`\mathcal{L} = -\min(r_{\text{seq}} \cdot A, \text{clip}(r_{\text{seq}}) \cdot A)`

Intuition:
- If advantage > 0 (this completion is good) → the loss pushes the ratio up → the model "prefers" this kind of answer
- If advantage < 0 (this completion is bad) → the loss pushes the ratio down → the model "dislikes" this kind of answer
- The clip mechanism prevents updating too much in one shot (a policy that changes too fast may "forget" what it learned before)

A more detailed look at the loss function is in the GSPO and GRPO chapters.

Architecture Concept: What Happens Inside ``train()``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 65 45

   * - Call / Step
     - Description
   * - ``trainer.train(batch, gspo_loss_fn)``
     -
   * - ``→ Trainer``
     -
   * - ``→ Backend.train()``
     -
   * - ``→ Engine``
     -
   * - ``→ TrainingManager``
     -
   * - ``→ Worker``
     -
   * - ``→ forward (compute logprobs)``
     - compute logprobs
   * - ``→ loss_fn (compute GSPO loss)``
     - compute GSPO loss
   * - ``→ backward (backprop)``
     - backprop
   * - ``→ optimizer.step() (gradient update)``
     - gradient update
   * - ``→ new weights take effect automatically → next rollout uses the new policy``
     -

**Why do the weights take effect automatically?** Inside AReno's Engine, inference (rollout) and training (optimizer step) share the same model weights. After training updates the weights, the next rollout automatically uses the new policy — no manual synchronization is needed.

3 Complete Data Flow Diagram
---------------------------------

Chain the six sections above together and you get a complete RL training data flow:

.. list-table::
   :header-rows: 1
   :widths: 18 72 18

   * - Stage
     - Data flow / Operations (verbatim)
     - Annotation
   * - Goal
     - one RL Training Step
     -
   * - 1. Input
     - ``prompt = "1+1=?"``
       ▼ ``tokenize``
       ``prompt_tokens = [P1, P2, P3, ...]``
     -
   * - 2. Rollout
     - rollout (n_samples=8): the 8 completions are sampled independently by the current policy
       ``sequences = [``
       ``  {resp_tokens: [R1, R2, EOS], resp_logprobs: [-1.2, -0.8, 0.0]},``
       ``  {resp_tokens: [R1, EOS],     resp_logprobs: [-0.3, 0.0]},``
       ``  ...8 in total``
       ``]``
     - ← more "confident"
   * - 3. Reward scoring
     - ``reward_fn(record)`` scores each completion
       ``rewards = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]``
     - (3 correct, 5 wrong)
   * - 4. Advantage computation
     - ``compute_group_advantages(rewards)``: A_i = (r_i - mean) / (std + ε)
       ``advantages = [1.208, -0.725, -0.725, 1.208, -0.725, -0.725, 1.208, -0.725]``
     - ↑ encourage ↑ suppress ↑ suppress ↑ encourage (at the corresponding positions)
   * - 5. TrainSequence packing
     - each completion → one ``TrainSequence``, forming:
       ``batch = [``
       ``  TrainSequence(``
       ``    tokens           = [P1,P2,...,R1,R2,EOS],``
       ``    logprobs         = [0,0,...,-1.2,-0.8,0.0],``
       ``    prompt_len       = 3,``
       ``    scalar_advantage = 1.208,``   ← positive signal
       ``    reward           = 1.0,``
       ``  ),``
       ``  TrainSequence(``
       ``    tokens           = [P1,P2,...,R1,EOS],``
       ``    logprobs         = [0,0,...,-0.3,0.0],``
       ``    prompt_len       = 3,``
       ``    scalar_advantage = -0.725,``  ← negative signal
       ``    reward           = 0.0,``
       ``  ),``
       ``  ...8 in total``
       ``]``
     -
   * - 6. Train
     - ``trainer.train(batch, gspo_loss_fn)``, inside ``loss_fn``:
       ``1. re-run forward → get the current policy's logprobs``
       ``2. r_seq = exp(mean(logπ_new - logπ_old))``  ← sequence-level importance ratio
       ``3. L = -min(r_seq * A, clip(r_seq) * A)``     ← PPO clip
       ``4. L.backward()``                              ← backprop
       ``5. optimizer.step()``                          ← update parameters θ
     -
   * - 7. The loop
     - new weights :math:`\theta'` → back to the start → the next rollout uses the new policy
     -

**The loop continues**: new weights produce new rollouts → new rewards → new advantages → new loss → new weights → ...

4 Memorizing the Five Core Concepts
-----------------------------------------

After reading this chapter, remember these five concepts:

.. list-table::
   :header-rows: 1
   :widths: 22 52 58

   * - Concept
     - One sentence
     - Where in the code
   * - **Policy**
     - language model = :math:`\pi_\theta(a_t \mid s_{<t})`
     - ``Trainer(model_path="...")``
   * - **Rollout**
     - generate completions with the current policy
     - ``trainer.rollout_token_batch(prompts, n_samples=8)``
   * - **Reward**
     - the function that grades completions
     - ``reward_fn(RewardRecord(...))``
   * - **Advantage**
     - within-group normalized relative goodness
     - ``compute_group_advantages(rewards)``
   * - **TrainSequence**
     - the unified format packing prompt+completion+signal
     - ``TrainSequence(tokens=..., logprobs=..., ...)``

These five concepts run through all of AReno's online RL algorithms (GSPO/GRPO/PPO). When later chapters introduce new algorithms, only the concrete implementation of these concepts differs (e.g., PPO computes advantage with a critic rather than in-group normalization), but the structure stays the same.

5 Chapter Summary
---------------------

After reading this chapter, you should be able to:

1. **Read AReno's 50-line Quick Start snippet** and understand what each line is doing
2. **Understand the five core RL concepts**: policy, rollout, reward, advantage, TrainSequence
3. **Understand RL training from a data-flow perspective**: prompt → rollout → reward → advantage → TrainSequence → loss → optimizer.step → loop
4. **Distinguish on-policy from off-policy**: RL uses the data generated by the model itself; SFT uses fixed data
5. **Know the relationship between CLI and SDK**: ``areno train --algo gspo`` and ``trainer.train(batch, gspo_loss_fn)`` do the same thing

The next chapter (Repository Tour) takes you on a tour of AReno's repository structure and builds a "programmer's map" for reading code — so when you need to go deep into the source, you know where to look for what.
