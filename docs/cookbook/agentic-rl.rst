Agentic RL
==========

.. admonition:: Goal

   Understand the core concept of Agentic RL — the model does not directly produce
   the final answer; it completes the task through multi-turn tool calls. Read
   AReno's trajectory modeling, tool-call parsing, and local OpenAI proxy endpoint
   in depth, then run two complete hands-on examples: TicTacToe and Coding Agent.

.. admonition:: Prerequisites

   Online RL Loop (online RL core loop), GSPO.

.. admonition:: Outcomes

   Understand the full pipeline of Agentic RL → be able to write an agent function /
   reward function / dataset loader → run TicTacToe and Coding Agent end to end.

1 The Concept of Agentic RL
------------------------------

Differences from Traditional RL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Traditional RL (GSPO, GRPO, PPO):

.. list-table::
   :widths: 35 65

   * - Step chain
     - Description
   * - ``prompt``
     - Task input
   * - ``→ the model generates one completion``
     - A single complete generation
   * - ``→ reward_fn scores``
     - Scores the result
   * - ``→ train``
     - Updates the model

Agentic RL:

.. list-table::
   :widths: 40 60

   * - Step chain
     - Description
   * - ``prompt``
     - Task input
   * - ``→ the model generates a tool_call``
     - The model decides to call a tool
   * - ``→ the environment executes the tool``
     - Execution in the environment
   * - ``→ returns the tool_result``
     - Returns the execution result
   * - ``→ the model keeps reasoning``
     - The model interprets the result
   * - ``→ may call more tools``
     - The multi-turn loop
   * - ``→ ...``
     - Until it is done
   * - ``→ finally gives the answer``
     - Produces the final answer
   * - ``→ reward_fn scores based on the final result``
     - Scores by the final outcome
   * - ``→ train``
     - Updates the model

The essential difference: **the model does not directly generate the answer; it
completes the task through multi-turn tool interactions.**

Why Do You Need Agentic RL?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **The limitation of traditional RL**: the model can only optimize within the
  framework of "single-turn text generation", so it cannot handle tasks that need
  external tools (code execution, file operations, API calls)
- **What Agentic RL enables**: the model learns **when to call which tool**, **how to
  interpret tool results**, and **how to correct mistakes across multi-turn
  interactions**

The Core Loop
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Component / stage
     - Content
   * - Agentic RL core loop
     - Two-way traffic between the model and the environment
   * - ``Model (policy) → Environment (tool execution)``
     - ``tool_call``: the model requests tool execution
   * - ``Environment (tool execution) → Model (policy)``
     - ``tool_result``: the environment returns the tool result
   * - The model generates
     - "I need to call the search_file tool"
   * - The environment returns
     - "Found 3 matching files: a.py, b.py, c.py"
   * - The model reasons
     - "Let me read a.py..."
   * - The model generates
     - "Call read_file('a.py')"
   * - ...
     - ...multi-turn loop...
   * - The model finally
     - "The problem is solved"
   * - Wrap-up
     - → reward_fn scores based on the final result → train

2 Trajectory Modeling: AgentTrajectoryTurn
---------------------------------------------

The Core Data Structure
~~~~~~~~~~~~~~~~~~~~~~~

Agentic RL's training data is no longer a simple ``(prompt, completion)`` pair, but
**multi-turn trajectories**. AReno models each turn with ``AgentTrajectoryTurn``:

.. code-block:: python

   @dataclass(slots=True)
   class AgentTrajectoryTurn:
       item: AgentItem                          # current task (prompt + metadata)
       messages: list[dict[str, Any]]           # full conversation history up to this turn
       response: Any | None                     # the model's OpenAI response object
       input_tokens: list[int]                  # input tokens of the current turn
       response_tokens: list[int]               # output tokens of the current turn
       response_logprobs: list[float]           # output logprobs of the current turn
       parsed_tool_calls: list[dict[str, Any]]  # the parsed tool calls
       tools: list[dict[str, Any]]              # the available tool definitions
       tool_choice: Any                         # the tool-selection strategy

Key Design: loss_mask
~~~~~~~~~~~~~~~~~~~~~

**Not every token participates in the loss computation.** ``LossMaskPolicy`` controls
which content participates in training:

.. code-block:: python

   @dataclass(slots=True)
   class LossMaskPolicy:
       assistant_text: bool = True          # ✅ the model's text replies → participate in the loss
       assistant_tool_calls: bool = True    # ✅ the model's tool calls → participate in the loss
       tool_results: bool = False           # ❌ tool results → do not participate in the loss
       final_assistant_text: bool = True    # ✅ the final reply → participates in the loss
       system_prompt: bool = False          # ❌ system prompt → does not participate in the loss
       user_prompt: bool = False            # ❌ user input → does not participate in the loss

This design is crucial:

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - Message role
     - Content
     - loss_mask
   * - ``system``
     - ``"You are a helpful assistant"``
     - ``False`` (not generated by the model)
   * - ``user``
     - ``"What files are in /src?"``
     - ``False`` (not generated by the model)
   * - ``assistant``
     - ``"Let me check..."``
     - ``True`` (text generated by the model)
   * - ``assistant: tool_call``
     - ``search_file(...)``
     - ``True`` (the model decides to call a tool)
   * - ``tool``
     - ``"Found: a.py, b.py"``
     - ``False`` (returned by the environment, not generated by the model)
   * - ``assistant``
     - ``"Found 2 files: a.py, b.py"``
     - ``True`` (the model interprets the result)

What the model needs to learn:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Learn?
     - Content
     - Reason
   * - ✅ Yes
     - When to call a tool
     - → tool_call tokens participate in the loss
   * - ✅ Yes
     - Which tool to call and with what arguments
     - → tool_call tokens participate in the loss
   * - ✅ Yes
     - How to interpret tool results
     - → assistant_text tokens participate in the loss
   * - ❌ No
     - "What the tools returned"
     - → that is decided by the environment

**The ``--train-tool-results`` flag**: if set to True, tool results also participate
in the loss (an advanced usage that lets the model "learn" the output distribution of
tools as well).

3 New Architecture Components
--------------------------------

Agentic RL adds three key components to the existing architecture:

A Local OpenAI-Compatible Proxy Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``areno/api/openai_chat.py`` is **not** the HTTP service of ``areno serve``; it is an
**in-process OpenAI-compatible proxy**. It starts a lightweight HTTP server inside the
Trainer process, so an agent function can call the model with the standard OpenAI SDK:

.. code-block:: python

   from openai import AsyncOpenAI

   client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key)
   response = await client.chat.completions.create(
       model="policy",
       messages=[...],
       tools=[...],        # pass the tool definitions
       tool_choice={...},  # force a specific tool
       stream=False,
   )

Core capabilities:

- Supports ``tools`` and ``tool_choice`` (the standard OpenAI function-calling format)
- Supports multi-turn conversations (full message histories with tool_call and tool_result)
- Automatically converts OpenAI-format messages to the model's native chat-template format

The Tool-Call Parser
~~~~~~~~~~~~~~~~~~~~~

``areno/api/tool_call_parser.py`` parses the tool-call text produced by the model.
Different models use different tool-call formats:

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - Model family
     - Tool-call format
     - Parser
   * - Qwen / Ling
     - ``<tool_call>`` XML tag
     - ``qwen``
   * - Gemma4
     - ``<tool_call>`` tag
     - ``gemma4``
   * - MiniCPM
     - Custom format
     - ``minicpm``
   * - LLaMA / general
     - JSON
     - ``json`` (default)

.. code-block:: python

   def infer_tool_call_parser_name(trainer) -> str:
       """Automatically pick the right parser based on the model's tokenizer template"""
       template = str(getattr(tokenizer, "chat_template", "") or "").lower()
       if "qwen" in haystack or "ling" in haystack:
           return "qwen"
       if "gemma4" in haystack:
           return "gemma4"
       return "json"  # default: general JSON parser

RolloutSession: Managing Multi-Turn Reasoning Sessions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``areno/api/agentic.py::RolloutSession`` is the core orchestrator of Agentic RL. It
manages:

- The local OpenAI-compatible HTTP endpoint (auto-started when ``proxy=True``)
- Continuous-batch inference scheduling (reuses the engine's InferenceManager)
- Message normalization and tokenization
- Trajectory construction and loss-mask computation

.. admonition:: The Boundary of the Environment Layer

   The "environments" in this chapter's examples (TicTacToe, Coding Agent) are simple
   built-in executors for demonstration. In real production agentic RL, the environment
   and evaluation usually come from frameworks like Harbor — they own docker sandboxes,
   benchmark-set configuration, and result scoring. AReno's interface to them is the
   in-process OpenAI-compatible proxy + explicit trajectories described in this
   chapter: how the agent calls the model and how trajectories come back are exactly
   the same on both sides. In other words, the trajectory modeling, loss mask, and
   proxy endpoint you learn here **do not change when the environment layer is
   swapped** (see **14.10** for the full division of labor and the three integration
   approaches).

4 The Complete Data Flow of Agentic Training
-----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Step
     - Content
   * - One Agentic Training Step
     - The overall flow of a single training step
   * - 1. Training data
     - ``{"board": [X,_,_,_,O,_,_,_,_], "best_moves": [3,7], ...}``
   * - 2. dataset_loader
     - ``board → prompt`` ("Choose X's next move on this board:\n...")
   * - 3. The agent function is called
     - Create the OpenAI client → point it at the local proxy endpoint

       Send the messages + tools definitions

       The model returns: ``tool_call: choose_square(square=3)``

       (for multi-turn) the environment executes the tool → returns the tool_result

       Finally return the list of AgentTrajectoryTurn
   * - 4. RolloutSession turns the trajectory into a TrainSequence
     - assistant turns → tokens + logprobs → ``loss_mask=True``

       tool turns → tokens → ``loss_mask=False`` (default)

       the prompt of the whole trajectory → ``prompt_mask=True``
   * - 5. reward_fn(record)
     - Extract the tool_call from the trajectory

       Use the environment to judge whether this tool_call is correct

       Return a float score
   * - 6. compute_group_advantages
     - ``compute_group_advantages(rewards)`` → TrainSequence packing
   * - 7. Train
     - ``trainer.train(batch, gspo_loss_fn)`` → backward → ``optimizer.step()``

**Key differences from standard RL**:

.. list-table::
   :header-rows: 1
   :widths: 20 36 44

   * -
     - Standard RL (GSPO)
     - Agentic RL
   * - Agent function
     - None
     - ``run_agent(ctx, batch)``
   * - How the model is invoked
     - ``rollout_token_batch()``
     - OpenAI SDK → local proxy endpoint
   * - Data format
     - ``prompt`` → ``completion``
     - ``messages`` → ``tool_calls`` → ``tool_results`` → ...
   * - loss_mask
     - Entire response = True
     - assistant=True, tool_result=False
   * - Reward function input
     - ``record.completion`` (text)
     - ``record.tool_calls`` (structured tool calls)

5 Hands-On 1: TicTacToe — The Simplest Agentic RL Starter
-------------------------------------------------------------

TicTacToe is the simplest agentic example: one board, one ``choose_square`` tool, and
a single-move decision. It is ideal for understanding the basic flow of agentic RL.

File Structure
~~~~~~~~~~~~~~

.. list-table::
   :widths: 55 45

   * - Path/node
     - Responsibility
   * - ``examples/agentic/tictactoe/``
     -
   * - ``game.py``
     - Board logic: legal moves, minimax best moves, win/loss detection
   * - ``dataset_generator.py``
     - Randomly generate board states → boards.jsonl
   * - ``dataset_loader.py``
     - board → prompt format conversion
   * - ``run_agent.py``
     - Agent function: call the OpenAI SDK → tool_call
   * - ``reward.py``
     - Reward function: extract the tool_call → score with game.score_move
   * - ``web_ui.py``
     - Browser play UI

Reading the Agent Function in Depth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core of ``run_agent.py``:

.. code-block:: python

   SYSTEM_PROMPT = (
       "You are a careful Tic-Tac-Toe player. You play X. "
       "Choose exactly one legal empty square by calling the choose_square tool. "
       "Digits on the board are empty square labels, not marks. "
       "Win immediately if possible; otherwise block any immediate O win."
   )

   CHOOSE_SQUARE_TOOL = {
       "type": "function",
       "function": {
           "name": "choose_square",
           "description": "Choose the next Tic-Tac-Toe square for X.",
           "parameters": {
               "type": "object",
               "properties": {"square": {"type": "integer", "minimum": 1, "maximum": 9}},
               "required": ["square"],
           },
       },
   }

   async def run_agent(ctx, batch):
       """Have the model pick one move for each board state"""
       client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, ...)
       items = list(batch.iter_samples())

       async def run_one(item):
           messages = [
               {"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": item.prompt},
           ]
           response = await client.chat.completions.create(
               model="policy", messages=messages,
               tools=[CHOOSE_SQUARE_TOOL],
               tool_choice={"type": "function", "function": {"name": "choose_square"}},
               stream=False,
           )
           return AgentTrajectoryTurn(item=item, messages=messages, response=response,
                                       tools=[CHOOSE_SQUARE_TOOL], ...)

       return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))

**Key points**:

- ``ctx.get_base_url()`` returns the URL of the local proxy endpoint (``http://127.0.0.1:PORT/v1``)
- ``tool_choice`` forces the model to call the ``choose_square`` tool
- Returns ``AgentTrajectoryTurn`` instead of bare text — AReno uses its ``response_tokens``
  and ``response_logprobs`` to build the TrainSequence

Reading the Reward Function in Depth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def reward_fn(record):
       """Extract the choose_square call from the trajectory and judge whether it is legal/optimal"""
       source = record.source_record
       board = game.normalize_board(source["board"])
       return game.score_move(board, _tool_square(record))

   def _tool_square(record):
       for call in record.tool_calls:          # record.tool_calls is the parsed structured data
           if call.get("name") == "choose_square":
               return int(call["arguments"]["square"])
       return None

The scoring logic of ``game.score_move``:

- Win → 1.0
- Block the opponent's win → 0.8
- Legal move (draw) → 0.5
- Illegal move → 0.0

**Difference from a standard RL reward_fn**: it does not read ``record.completion``
(text); it reads ``record.tool_calls`` (a list of structured tool calls).

The Complete Training Command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Generate the board data
   python examples/agentic/tictactoe/dataset_generator.py \
     --output /tmp/areno-tictactoe-boards.jsonl \
     --count 2048 --seed 2026

   # 2. Start training
   areno train \
     --ckpt Qwen/Qwen3-1.7B \
     --dataset-path /tmp/areno-tictactoe-boards.jsonl \
     --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --algo gspo \
     --batch-size 2 \
     --n-samples 4 \
     --max-new-tokens 32

New Agentic-specific parameters:

- ``--agent-fn``: specify the agent function file (defines ``async run_agent(ctx, batch)``)
- ``--agent-timeout-s``: the timeout for each agent call (default 300 seconds)

> **Recipe tip**: TicTacToe is the smallest agentic recipe, useful for learning
> the shape of an agentic task before moving to a larger environment. When
> adapting it, focus on four points: replace the rule loop with your
> environment; keep the agent function responsible for external actions; return
> trajectory turns that AReno can tokenize and score; and add reward diagnostics
> before increasing concurrency. The agentic rollout API contract lives in
> ``reference/agentic-rollout-api.rst``.

Playing against the Trained Model with the Web UI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Terminal 1: start the model server
   areno serve --model-path ./checkpoints/step_001000 --port 8000

   # Terminal 2: start the Web UI
   python examples/agentic/tictactoe/web_ui.py \
     --base-url http://127.0.0.1:8000/v1 --model policy
   # open http://127.0.0.1:8767 in your browser

6 Hands-On 2: Coding Agent — A Complete Multi-Turn Tool-Calling Example
--------------------------------------------------------------------------

Coding Agent is a more realistic agentic scenario: the model works inside a temporary
workspace and fixes code through multi-turn tool calls.

The Tool Set
~~~~~~~~~~~~

.. code-block:: text

   # 9 restricted tools in total
   inspect_tree     # view the directory tree
   list_files       # list files
   read_file        # read file contents by line number
   rg               # regex-search file contents
   apply_patch      # apply a unified diff patch
   replace_text     # exact text replacement
   write_file       # create/overwrite/append a file
   run_command      # run a test command (with a timeout and output limits; forbids destructive commands like rm)
   submit           # submit the final state: solved / blocked

The Dataset
~~~~~~~~~~~

``examples/agentic/coding/dataset.jsonl``: 100 self-contained SWE-bench-style tasks,
from easy to hard. Each record contains:

.. code-block:: text

   {
     "instance_id": "task-001",
     "problem_statement": "Fix the bug in calculator.py...",
     "files": {"calculator.py": "def add(a, b): return a - b\n", ...},
     "test_commands": ["pytest test_calculator.py -v"],
     "FAIL_TO_PASS": ["test_add_positive"],
     "PASS_TO_PASS": ["test_subtract"]
   }

Tasks are reconstructed in a temporary directory (create the files → run the initial
tests to confirm they fail → the agent fixes the code → verify the tests pass).

The Training Command
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path examples/agentic/coding/dataset.jsonl \
     --dataset-loader-fn examples/agentic/coding/dataset_loader.py \
     --reward-fn-path examples/agentic/coding/reward.py \
     --agent-fn examples/agentic/coding/run_agent.py \
     --algo gspo \
     --tp-size 1 --world-size 1 \
     --batch-size 1 --n-samples 2 \
     --max-prompt-tokens 4096 --max-new-tokens 256

Recommendations for the coding-agent scenario:

- ``--max-prompt-tokens 4096``: coding tasks need more context
- ``--max-new-tokens 256``: the response for each tool call is short
- ``--agent-timeout-s 600``: multi-turn tool calling takes more time
- ``--batch-size 1 --n-samples 2``: each coding step takes a long time, so use a smaller batch

7 Other Agentic Examples
---------------------------

DuelGrid
~~~~~~~~

A browser-based, turn-based battle game (``examples/agentic/duelgrid/``):

- The model controls a character that moves on a map, collects items, and fights
- **Multi-action turns**: each turn may contain several actions (move + attack + use an item)
- GIFs visually show the before/after training effect
- The README has the full reward curve and before/after training animations

> **When to use it**: reach for DuelGrid after you have TicTacToe working — when
> you need richer environment state or a visual replay loop.

CodeBreaker
~~~~~~~~~~~

A number-guessing game (``examples/agentic/codebreaker/``):

- The model infers a hidden number combination by asking questions
- Has a terminal TUI so you can interactively experience the model's reasoning

Shopping
~~~~~~~~

A shopping scenario (``examples/agentic/shopping/``):

- The model acts as a shopping assistant, searching and comparing prices to help the
  user find the best product
- Tool set: ``search_products``, ``get_product_details``, ``compare_prices``

8 Hands-On 3: Vision-Language — When the Model Can "See"
------------------------------------------------------------

``examples/vl/tictactoe_image/``: the image-based version of TicTacToe — instead of a
text board, the model **looks at a board image** to decide its move.

Core differences:

- The input is an **image** (a board PNG), not a textual board representation
- Uses ``areno/api/multimodal.py`` to process image inputs
- The message format becomes ``{"role": "user", "content": [{"type": "image_url", "image_url": {...}}, {"type": "text", "text": "..."}]}``

Training command (requires a multimodal model, e.g. Qwen2.5-VL):

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen2.5-VL-7B-Instruct \
     --dataset-path /tmp/tictactoe_image.jsonl \
     --dataset-loader-fn examples/vl/tictactoe_image/dataset_loader.py \
     --reward-fn-path examples/vl/tictactoe_image/reward.py \
     --agent-fn examples/vl/tictactoe_image/run_agent.py \
     --algo gspo \
     --tp-size 1 --world-size 1

9 Writing Your Own Agentic RL Task
-------------------------------------

To recap, an agentic RL task needs three files:

1. The Agent Function (``--agent-fn``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def run_agent(ctx, batch) -> AgentTrajectory:
       """Must return an AgentTrajectory or a list of AgentTrajectoryTurn"""
       client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key)
       turns = []
       for item in batch.iter_samples():
           messages = [{"role": "user", "content": item.prompt}]
           response = await client.chat.completions.create(
               model="policy", messages=messages,
               tools=[...],  # your tool definitions
               stream=False,
           )
           turns.append(AgentTrajectoryTurn(item=item, messages=messages, response=response, ...))
       return AgentTrajectory(turns=turns)

2. The Reward Function (``--reward-fn-path``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def reward_fn(record) -> float:
       """Extract the tool call from record.tool_calls and judge correctness with the environment"""
       for call in record.tool_calls:
           if call["name"] == "your_tool":
               return your_scoring_logic(call["arguments"])
       return 0.0

3. The Dataset Loader (``--dataset-loader-fn``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def load_training_dataset(dataset_path, *, default_loader, **kwargs):
       """At least produce prompt (a string) and metadata (used for reward_fn scoring)"""
       dataset = default_loader(dataset_path)
       return [{"prompt": row["prompt"], "board": row["board"], ...} for row in dataset]

A Multi-Turn Tool-Calling Agent Function Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def run_agent(ctx, batch):
       client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key)
       turns = []
       max_turns = 10

       for item in batch.iter_samples():
           messages = [{"role": "user", "content": item.prompt}]
           for _ in range(max_turns):
               response = await client.chat.completions.create(
                   model="policy", messages=messages,
                   tools=TOOLS, stream=False,
               )
               msg = response.choices[0].message

               # if the model decides not to call any more tools → stop
               if not msg.tool_calls:
                   turns.append(AgentTrajectoryTurn(item=item, messages=list(messages), response=response))
                   break

               # record this turn
               turns.append(AgentTrajectoryTurn(item=item, messages=list(messages), response=response))

               # execute the tool calls
               messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
               for tc in msg.tool_calls:
                   result = execute_tool(tc.function.name, tc.function.arguments)
                   messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

       return AgentTrajectory(turns=turns)

10 The Boundary of the Environment Layer: Dividing Responsibilities with Harnesses Like Harbor
-------------------------------------------------------------------------------------------------

The "environments" in the earlier examples (TicTacToe, Coding Agent) are demo-purpose
local executors — they are stuffed directly into ``run_agent``. But when you do
agentic RL in a real scenario, the environment and evaluation usually come from a
framework like **Harbor** (the agent evaluation and optimization framework built by
the Terminal-Bench authors). How they divide responsibilities with AReno and how you
plug them together is what this section clarifies.

10.1 Division of Responsibilities in One Sentence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Harbor owns the "task side"**: it defines versioned benchmark suites (``harbor
  datasets list`` lists what it supports — SWE-Bench, Terminal-Bench, Aider Polyglot,
  ...), spins up docker sandboxes or cloud sandboxes, drives any agent framework
  (Claude Code, Codex CLI, OpenHands, ...), runs the full trajectory, and scores it at
  the end. Its interface is three flags: ``--dataset``, ``--agent``, ``--model``.
- **AReno owns the "policy side"**: the same model does three jobs — it acts as the
  agent's "brain" (generation), stuffs per-token **logprobs** into the response (the
  raw material for RL gradients), and updates the weights through training in the same
  process. It also provides the ``areno serve`` standard OpenAI endpoint.

.. list-table::
   :header-rows: 1
   :widths: 42 28 30

   * - Capability
     - Harbor
     - AReno
   * - Benchmark registry / eval-set configuration
     - ✅ ``harbor datasets``
     - ❌ (tasks are a function contract, see 14.9)
   * - Docker sandbox / cloud environment
     - ✅
     - ❌ (the examples use a local toy environment)
   * - Run the agent, execute tools, score
     - ✅
     - ❌
   * - Model generation (with per-token logprobs)
     - ❌ it only looks at the ``--model`` endpoint
     - ✅ in-process proxy
   * - RL training loop / serving
     - ❌
     - ✅

10.2 The Handshake Point: One OpenAI-Compatible Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Harbor does not care who is behind it; it simply sends requests to the
OpenAI-compatible endpoint that ``--model`` points at:

.. code-block:: bash

   harbor run --dataset swe-bench@... --agent codex --model <AReno endpoint>

AReno provides this endpoint in two scenarios:

- **When deploying / evaluating** → ``areno serve``: a standard OpenAI service to the
  outside world;
- **When training** → the in-process proxy ``RolloutSession``: the response carries an
  extra ``areno.response_logprobs`` field.

To put it in one sentence: **for online RL with gradients, the model must be held by
AReno's training loop** — logprobs are only returned by the in-process proxy, and
``areno serve`` responses to ordinary clients do not carry this metadata by default.
Harbor can tell you whether a task ran correctly, but the gradient material can only
come from AReno's training loop.

10.3 Three Practical Integration Approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Posture A: Harbor provides the data, AReno provides the gradients (offline, zero intrusion)**

.. list-table::
   :widths: 100

   * - ``Harbor runs the benchmark set with the current baseline model``
   * - ``→ exports the rollout trajectories``
   * - ``→ normalizes them into AReno's dataset ({prompt, solutions} row schema)``
   * - ``→ areno train --algo gspo``
   * - ``→ after training, run the same benchmark set with Harbor again and compare the improvement``

- Pros: zero intrusion on both sides; each does its own thing; easy to understand.
- Cost: the trajectories are produced by the "old model", so they are not on-policy.
  Strictly speaking, this suits starting with SFT / DPO / preference fine-tuning, then
  moving to online fine-tuning.

**Posture B: AReno ships its own agentic, with no Harbor in the training loop (the official ready-made form)**

.. code-block:: bash

   areno train --agent-fn examples/agentic/coding/run_agent.py ...

Inside ``run_agent`` is AReno's own scenario code (a local temporary repo / game
environment); model calls go through the **in-process proxy** to get logprobs, so
on-policy holds. Harbor retreats to two positions: baseline probing before training +
evaluation after training.

**Posture C: embed Harbor as an "environment service" inside ``run_agent`` (DIY — but the interface exists for exactly this kind of integration)**

.. code-block:: python

   async def run_agent(ctx, batch) -> AgentTrajectory:
       client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key)  # ← AReno's in-process proxy
       turns = []
       for item in batch.iter_samples():
           messages = [{"role": "user", "content": item.prompt}]
           for _ in range(max_turns):
               response = await client.chat.completions.create(
                   model="policy", messages=messages, tools=TOOLS, stream=False)
               turn = AgentTrajectoryTurn(item=item, messages=list(messages), response=response)
               turns.append(turn)
               if not turn.parsed_tool_calls:
                   break                                       # the model decides to stop
               messages.append({"role": "assistant",
                                "tool_calls": to_openai_tool_calls(turn.parsed_tool_calls)})
               for call in turn.parsed_tool_calls:
                   result = harbor_execute(call)               # ← integrate with the Harbor sandbox here
                   messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
       return AgentTrajectory(turns=turns)

- Reasoning / gradients belong to AReno, environment / execution belong to Harbor, so
  online on-policy holds.
- Cost: there is currently no official bridge package; the "harbor_execute" glue lives
  in your ``run_agent``.

This is exactly how the sentence in the AReno README — "for any agent harness, use
proxy recording + trajectory reconstruction" — is realized.

10.4 How to Choose
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Your goal
     - Integration
   * - Quick start / teaching
     - B (consistent with this chapter's examples)
   * - Train a real coding agent
     - C (online) or A (offline data first)
   * - Mainly evaluating / comparing baselines
     - Use only Harbor, no training

11 Chapter Summary
---------------------

1. **Agentic RL = the model completes tasks through multi-turn tool calls**, not single-turn text generation
2. **``AgentTrajectoryTurn``** models each turn: messages + response + tool_calls + logprobs
3. **``loss_mask`` is a key design**: the assistant's tool_call and text replies participate in the loss, while tool results do not
4. **Three new components**: the local OpenAI proxy endpoint (``openai_chat.py``), the tool-call parser (``tool_call_parser.py``), and the trajectory builder (``agentic.py``)
5. **TicTacToe is the simplest starter**: one tool, a single-move decision, a fully runnable command
6. **Coding Agent is a complete multi-turn tool-calling example**: 9 tools, 100 SWE-bench-style tasks
7. **Writing your own agentic task needs three files**: ``run_agent.py`` + ``reward.py`` + ``dataset_loader.py``
8. **The environment layer lives outside AReno**: when you use a harness like Harbor, model calls still point at an OpenAI-compatible endpoint (in-process proxy for training, ``areno serve`` for deployment), and the trajectory contract stays the same; but on-policy gradient material only comes from AReno's training loop — see **14.10** for the task-side vs policy-side division of labor
