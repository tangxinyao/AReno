# Talk script: Ling-3.0-tiny × AReno: From Hermes Traces Toward an RSI Flywheel

Companion page: `dgx_spark_sft_talk.html` (← / → or J / K to move between sections)
Based on: `dgx_spark_sft_runbook.md`
Chinese version: `dgx_spark_sft_talk_script.md`

> **Before you speak**: the `areno` commands in the runbook **have not been run on a real machine yet**. Lines marked **[LIVE]** must be filled with real output or real numbers once the run works. Never present an unrun result as "here's what we got."

How to read this script: each section is one continuous piece of narration, meant to be spoken straight through. *Italic lines* are cues for you, not words to say.

Suggested length: about 18 minutes, plus 5 minutes of Q&A. All live demo prompts are in English.

| Section | Page anchor | Time | The one line the audience should remember |
| -- | -- | -- | -- |
| Opening | top | 1.5 min | A new model shipped, ours doesn't know it, and today we teach it |
| 1 | Mac Mini | 3 min | 1.3B inference cost, 7.9B of knowledge, runs on a desk |
| 2 | Wrong answer | 2 min | Wrong isn't dumb; the knowledge just isn't there |
| 3 | DGX Spark | 3 min | LoRA trains one small piece; one CLI for training and serving |
| 4 | Data | 3 min | Logs show what's missing; the dataset supplies the right answers |
| 5 | LoRA SFT | 2.5 min | One command, 500 steps, 10 checkpoints |
| 6 | Results | 2 min (plus live demo) | Same question, two ports, check the facts |
| 7 | Next | 1 min | The flywheel turns one step at a time |

---

## Opening

*On screen: the title and the six-step loop, with step 05 highlighted.*

Hi everyone. Today I want to walk you through something small that I find really interesting.

The model at the center of this talk is Ling-3.0-tiny, a small model that runs locally. Its knowledge stops on August 6, 2026, the day it was released. A little later, the Ant Ling team released a new model called Ling-3.0-flash-Fin, and tiny has never heard of it. So when we asked tiny in Hermes what flash-Fin is, it got it wrong. Today I'll show you how we fixed that, with a Mac Mini, a DGX Spark, and AReno, our own training framework.

The whole story is the loop you see here. A new model ships, and our model gets it wrong. We go back into the logs and find that wrong answer, turn the missing knowledge into a dataset, train a LoRA on it, and check the result on the spot. People talk a lot about RSI, recursive self-improvement, and the flywheel it promises. This is the smallest turn of that flywheel we could build. Let's start with where tiny lives.

---

## 1. This is a Mac Mini

*On screen: the Mac Mini photo, "1.3B / 7.9B" with the six-cell bar, then the four setup steps.*

This is a Mac Mini, and tiny runs on it. tiny is an open-source MoE model from the Ant Ling team: 7.9B parameters in total, but only 1.3B active per token, about one sixth, which is this lit cell. So it knows as much as a 7.9B model and runs as cheaply as a 1.3B one.

It has a 128K context window, and the deck lists 168 tokens per second, a figure from Artificial Analysis. A Mac Mini with 16 GB of memory or more runs the whole thing.

*(168 tokens/s is a third-party figure quoted in the deck, not something we measured.)*

Setting it up takes about half an hour. Most of that is downloading the GGUF weights from ModelScope, around fifteen minutes, and with Q8_0 you need at least 16 GB of memory. Then you install llama.cpp with Homebrew and start llama-server. Sending the first request takes a minute, and hooking it into an app you already have is one line: point `base_url` at `localhost:8080/v1`. The server speaks the OpenAI protocol, so any existing client just works.

So it's small and easy to set up. But even a good model has a limit it can't get around.

---

## 2. Good models still hit limits

*On screen: the Hermes chat on the left, the knowledge-cutoff timeline on the right.*

That limit is the knowledge cutoff. Everything tiny knows stops on August 6. Anything after that, it simply hasn't seen. So we asked it in Hermes: what is Ling-3.0-flash-Fin? It didn't know, so it guessed. It read the "Fin" in the name as fine-tuning, and built a whole wrong explanation on top of that.

*[LIVE] Show the original Hermes log and read tiny's actual answer. The page only has a placeholder; don't invent the answer.*

The model isn't wrong because it's dumb. It's wrong because this just isn't in what it knows. And this isn't an example we made up for the talk. It came from our real Hermes history, and in a few minutes I'll show you how we found it. But the conclusion comes first: if it doesn't know, we teach it.

---

## 3. A DGX Spark + AReno

*On screen: the DGX Spark photo and spec plate, the base + adapter diagram, then the three cards.*

The lightest way to teach it is LoRA SFT. Look at this diagram. The big block is tiny's original 7.9B weights, and they stay frozen the whole time; not a single number changes. The small orange block is a rank-16 adapter, and that's the only thing we train. So a training run is quick, a bad run costs nothing because you just delete the adapter, and the base model always stays clean.

Training needs a machine that can train, and that's the DGX Spark, with 128 GB of unified memory. In the deck it runs the much bigger Ling-3.0-flash, 124B total and 5.1B active, at 345 tokens per second, so training a LoRA for tiny is well within what one unit can do. The run itself is small, too: 99 question-and-answer pairs for 10 epochs, 500 steps, which should take on the order of ten minutes.

*(Ten minutes is an estimate from the step count. Replace it with the measured time.)*

We train with AReno, our own training and serving framework. It has one CLI with two verbs. `areno train` does the training, and you switch between SFT, DPO, GSPO, GRPO and PPO with a single flag; today it's SFT. `areno serve` starts an OpenAI-compatible endpoint and can load the adapter you just trained. So the moment training finishes, you can serve the result locally, without setting up a separate inference stack. That leaves one question, and it's the important one: where does the training data come from?

---

## 4. Where the data comes from

*On screen: state.db → LLM → bad_cases.jsonl, the four categories, the orange callout, then 21 / 99 / 42 / 0.*

Hermes already improves itself in its own way: it turns what it learns from past sessions into memory and reusable skills. That makes the agent better, but the model underneath stays exactly the same. What we want is for the model itself to improve, and that's the idea behind Dream RSI: real usage produces logs, the logs contain failures, and those failures become training data that goes back into the model. We're doing the smallest version of that.

Hermes records every session in a local SQLite file called `state.db`. We hand that file to an LLM, read-only, and ask it to read every session from start to finish and pull out the ones where something went wrong. It writes them to `bad_cases.jsonl`, one per line, with the exact words from the turn that failed and a short note on what the problem was. It looks for four kinds of problems: the model misunderstood what the user wanted, it hallucinated wrong or invented facts, it used tools badly, or it stopped before giving a final answer. Our flash-Fin answer lands in hallucination.

You might ask why we need an LLM for this instead of a few rules. It's because the flash-Fin session looks perfectly normal from the outside. It ran to completion, with no error and no interruption. The only way to see that it's wrong is to actually read the answer and understand it.

*[LIVE] Open `bad_cases.jsonl` and show the flash-Fin entry: its category, the problem note, and the original text. Only quote counts from the real output.*

We don't train on that log entry directly, though. A wrong answer doesn't contain the right one; what it tells us is which knowledge is missing. For the right answers, we went to public material and wrote 21 facts about flash-Fin as 99 question-and-answer pairs, each with its sources, 51 in Chinese and 48 in English. We also wrote 42 more questions, half in each language, that cover the same facts with different wording and never appear in training. That's our holdout, and it's how we'll tell learning apart from memorizing. The bad cases themselves we keep for later, as the rejected side for DPO or GSPO. With the data ready, training is a single command.

---

## 5. LoRA SFT

*On screen: the training command, the step math and three rules, then the checkpoint ruler.*

This is the whole training step. It uses the SFT algorithm, pulls tiny straight from ModelScope, points at our 99 rows, and trains only a rank-16 LoRA. Activation checkpointing and 4-bit Adam keep memory down, and the learning rate starts at 1e-4 and decays to 1e-5. The flag I want you to notice is `--disable-thinking`, and I'll come back to it in a moment.

The numbers are simple. 99 rows at 2 per step is 50 steps an epoch, and 10 epochs is 500 steps. We save every 50 steps, so we end up with 10 checkpoints.

Three things here are easy to get wrong. First, there's no final checkpoint. AReno only saves on multiples of 50, so if the length budget drops enough rows that the run is shorter than 50 steps, you end up with nothing. Second, `--disable-thinking` has to be on for both training and serving. If the model is trained without a thinking block and served with one, it sees something it never saw in training, and the output gets strange. Third, whenever you change the data or the hyperparameters, use a new save directory, or you'll end up demoing the wrong checkpoint.

Those ten checkpoints also give us a way out. Knowledge injection overfits easily, and the symptom is that every answer starts with the same boilerplate paragraph, whatever you ask. If that happens, you don't retrain; you just go back to an earlier checkpoint, like step 200. We'll start with step 500.

*[LIVE] If you have the training log, show a few `train_stats` lines with the loss going down. Otherwise don't quote loss numbers.*

So, did it work?

---

## 6. Results

*On screen: the question, the Before / After boxes, the four checks, then holdout, control group and overfitting signal.*

We started two servers from the same base model. Port 8001 is plain tiny, and port 8000 is tiny with the adapter we just trained. Let's ask both the same question: what is Ling-3.0-flash-Fin?

*[LIVE] Run the curl loop from section 6 of the runbook, 8001 first, then 8000, and read out the key sentences.*

When you compare the two, don't judge how fluent they sound. Check the facts. Does it say flash-Fin is 124B total with 5.1B active? Does it mention a 256K context, the MIT license, and seven benchmarks including FinFIRST?

*Tick each check on the page as you hear it. Only tick what the adapter actually said.*

Answering the questions it trained on isn't enough, though. So we also ask six of the English holdout questions, worded in ways it has never seen. If it gets those right, it learned the facts rather than the sentences. And we ask two control questions, fifteen times thirteen and what carbon dioxide is, to make sure the training didn't erase what it already knew.

*[LIVE] Run the holdout check and the control group if there's time. 15 × 13 = 195.*

If every answer came back with the same opening, that would be overfitting, and we'd switch to an earlier checkpoint. That's the whole loop. Let me finish with what it means.

---

## 7. A good start

*On screen: the three cards, Today / Next / Why this pairing.*

What you saw today is the simplest kind of model improvement there is. A new model came out, the model we use didn't know about it, we found the wrong answer in our own logs, built a dataset, trained a LoRA, and it got better on the spot.

Harder domains, like office work or specific industries, need a lot more than 99 questions. They need a much richer eval set and mocked environments, so that when we say the model got better, it really did, and didn't just memorize a few hundred sentences. I've already started on that with a benchmark I built called OPC, short for one-person company. It gives a local model the everyday work of a one-person company, like reconciling revenue, releasing a website or triaging customer email, and checks whether the agent can plan, use its tools, and hand over a result that passes verification. We don't have time today to go into how the benchmark is built, or how we use SFT plus GSPO or GRPO to improve agentic performance on it; that's a talk of its own. It's a bigger project, and it's worth doing one step at a time.

But I think AReno and tiny are a good place to start. The model is small enough to retrain on your own machine as often as you like, training and serving live in one CLI, and the path from real usage to logs to training data already works. The RSI flywheel gets built one turn at a time, and today's turn started with one wrong answer.

Thank you.

---

## Appendix: Q&A notes

| Likely question | Key points |
| -- | -- |
| Why not full fine-tuning? | With 99 examples, full fine-tuning can easily wash out general ability. A LoRA can be deleted or swapped and the base never changes, which makes demos and rollbacks easy |
| Why serve on the Mac Mini but train on the Spark? | The Mac Mini is enough for inference. Training needs CUDA; AReno's training path runs on NVIDIA GPUs |
| Are 99 examples enough? | For one topic, it's a minimal amount that works for a demo. Whether it generalizes is what the holdout questions check |
| Did the length budget drop any rows? | The budget is 128 prompt tokens and 512 response tokens; anything longer is silently dropped. The longest answer is 1,216 characters, probably within 512 tokens, but the `stage=sft_dataset_filter` line in the training log is what counts. **[LIVE]** fill in the real `skipped_long_or_empty` count |
| How long does one training run take? | 500 steps, on the order of ten minutes; **[LIVE]** replace with the measured time |
| Will this break what the model already knows? | The base is frozen and only the adapter is trained; the control group checks exactly this; if it breaks, delete the adapter |
| Why not RAG? | RAG solves "being able to look it up". Today shows "the model knows it itself", which is the "train it back" step of the RSI loop. They don't conflict |
| What happens to the bad cases later? | This round they show which knowledge is missing; later they can be the rejected side for DPO / GSPO |
| What if training or serving errors out? | Use the troubleshooting table in the runbook appendix: budget too small drops rows, empty save-path, adapter missing its config, and if flash-attn doesn't match the GPU, switch to `--attn-backend native` |
| ModelScope can't find the model? | It's a repo-id casing issue; use `inclusionAI/Ling-3.0-tiny`, or `modelscope download` it locally first |

## Appendix: pre-demo checklist

- [ ] The original Hermes log of the flash-Fin wrong answer is found: screenshotted, or present in `bad_cases.jsonl`
- [ ] Training has finished, and `step_000500` (plus the fallback `step_000200`) contains `adapter_config.json`
- [ ] Note `skipped_long_or_empty` and the loss trend from the training log
- [ ] Both servers on 8000 and 8001 are up, **both** with `--disable-thinking`
- [ ] Run the before/after curl once in advance and confirm the adapter says at least some of the checked facts
- [ ] Replace the [live output] placeholders on the page, or switch to the terminal during the demo
- [ ] After the demo, `pkill -f "areno serve"` to free the ports
