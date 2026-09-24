# Talk script: Ling-3.0-tiny × AReno: From Hermes Traces Toward an RSI Flywheel

Companion page: `dgx_spark_sft_talk.html` (← / → or J / K to move between sections)
Based on: `dgx_spark_sft_runbook.md`
Chinese version: `dgx_spark_sft_talk_script.md`

> **One thing to settle before you speak**: the `areno` commands in the runbook **have not been run on a real machine yet**. Everything marked **[LIVE]** below must be replaced with real output or real numbers once the run works. Never present an unrun result as "here's what we got."

> **Language note**: all live demo prompts are in English: the before/after question, the English half of the holdout set, and the control group.

Suggested length: about 18 minutes, plus 5 minutes of Q&A.

| Section | Page anchor | Time | The one line the audience should remember |
| -- | -- | -- | -- |
| Opening | top | 1.5 min | A new model shipped, ours doesn't know it, and today we teach it |
| 1 | Mac Mini | 3 min | 1.3B inference cost, 7.9B of knowledge, runs on a desk |
| 2 | Wrong answer | 2 min | Wrong isn't dumb; the knowledge just isn't there |
| 3 | DGX Spark | 3 min | LoRA trains one small piece; one CLI for training and serving |
| 4 | Data | 3 min | Logs show what went wrong; the dataset supplies the right answers |
| 5 | LoRA SFT | 2.5 min | One command, 500 steps, 10 checkpoints |
| 6 | Results | 2 min (plus live demo) | Same question, two ports, check the facts |
| 7 | Next | 1 min | Start from one wrong answer and go step by step |

---

## Opening (top of the page, the six-step loop)

**On screen**: the headline, and below it the six-step loop with step 05 "LoRA SFT" highlighted in orange.

**Say**:

> Hi everyone. Today I want to talk about something small that I find really interesting.
>
> We use a local model called Ling-3.0-tiny. Its knowledge cutoff is August 6, 2026, the day it was released. After that, Ling released a new model called Ling-3.0-flash-Fin. tiny doesn't know it exists.
>
> So when you ask tiny in Hermes "what is flash-Fin?", it gets it wrong.
>
> Today we'll fix that with a Mac Mini, a DGX Spark, and AReno, our own training framework.

**Pointing at the six steps**:

> The whole process is these six steps: a new model ships, we notice the wrong answer, we find the evidence in the logs, build a dataset, run LoRA SFT, and watch it get better on the spot. People talk a lot about RSI, recursive self-improvement. What we're showing today is its smallest possible version.

**Transition**:

> Let's start with where tiny lives.

---

## 1. This is a Mac Mini

**On screen**: the Mac Mini photo on the left; on the right, the big "1.3B / 7.9B" and the six-cell bar with one cell lit; below, the four setup steps.

**Say: what the model is**

> This is a Mac Mini, and tiny runs on it.
>
> tiny is an open-source model from the Ant Ling team with a MoE architecture. There are two numbers here, and you need to read them separately.
>
> The first is **7.9B total parameters**. That decides how capable the model is, because the knowledge lives in the experts.
>
> The second is **1.3B active parameters**. Each inference step only calls some of the experts, about one sixth, which is this one lit cell. Less computation means more speed.
>
> Put together: **the inference cost of 1.3B, with the knowledge of 7.9B.**

**Say: specs**

> 128K context. For speed, the deck lists 168 tokens per second, a figure from Artificial Analysis. A Mac Mini with 16 GB of memory or more runs the whole thing.

(Note: 168 tokens/s is a third-party figure quoted in the deck, not something we measured. Present it that way.)

**Say: setup**

> Setup is simple too: four steps, about 30 minutes.
>
> Step one, download the GGUF weights from ModelScope, about 15 minutes. With Q8_0 quantization you need at least 16 GB of memory.
> Step two, `brew install llama.cpp`, then start the server with `llama-server`. That brings you to about half an hour in total.
> Step three, send the first request. One minute.
> Step four, in your existing app, change `base_url` to `localhost:8080/v1`. One minute.
>
> The server speaks the OpenAI protocol, so any existing client connects by changing one line.

**Transition**:

> Small and easy to install is great. But even a good model has a boundary it can't get around.

---

## 2. Good models still hit limits

**On screen**: the Hermes chat on the left; on the right, the knowledge-cutoff timeline (the solid part is what tiny knows, the hatched part is what it can't see).

**Say**:

> That boundary is the **knowledge cutoff**. tiny's knowledge stops on August 6, 2026. It knows nothing about what happened after.
>
> We asked it in Hermes: "What is Ling-3.0-flash-Fin?"
>
> It doesn't know the model, so it guesses. It reads the "Fin" in the name as fine-tuning, and builds a wrong explanation on top of that.

**[LIVE]** Switch to the original Hermes log, or to that entry in `bad_cases.jsonl` from section 4, and read tiny's actual words to the audience. The page only has a placeholder here; don't invent its answer.

> I want to stress one thing: **the model isn't wrong because it's dumb. It's wrong because this simply isn't in what it knows.**
>
> And we didn't make this example up for the demo. It's in the real Hermes history. In section 4 I'll show how we found it in the logs.

**Transition**:

> If it doesn't know, we teach it.

---

## 3. A DGX Spark + AReno

**On screen**: the DGX Spark photo, the spec plate, the "base (hatched, frozen) + adapter (orange)" diagram; below, three cards: `areno train`, `areno serve`, No cloud.

**Say: why LoRA**

> The lightest way to change a model's behavior is further training, specifically LoRA SFT.
>
> Look at this diagram. The large block on the left is tiny's original weights, 7.9B, frozen the whole time. Not a single number changes. The small orange block on the right is the adapter, rank 16. That's the only thing we train.
>
> That gives us three things. First, it's fast: you can retrain in well under an hour. Second, if training goes wrong, you just delete the adapter. Third, the base model always stays clean.

**Say: the machine**

> Training needs a machine that can train. That's the DGX Spark, with 128 GB of unified memory.
>
> In the deck it runs the bigger Ling-3.0-flash, 124B total and 5.1B active, at 345 tokens per second. Today we use it to train a LoRA for tiny, and a single unit's unified memory is more than enough.
>
> The training run is tiny: 99 Q&A pairs, 10 epochs, 500 steps in total, on the order of ten minutes.

(Note: "about ten minutes" is an estimate from the step count; the runbook has no measured time. Replace it with the real time after the run.)

**Say: AReno**

> We train with AReno, our own training and serving framework.
>
> Its CLI has two verbs. `areno train` handles training and supports SFT, DPO, GSPO, GRPO, and PPO, switched with `--algo`. Today we use SFT. `areno serve` handles inference: it starts an OpenAI-compatible endpoint and can load the adapter we just trained.
>
> So **you can serve right after training**, and any OpenAI client can connect. No cloud, and no separate inference stack to set up.

**Possible questions**:

- "Why not full fine-tuning?" → With 99 examples, full fine-tuning can easily wash out general ability. A LoRA can be deleted or swapped and the base never changes, which makes demos and rollbacks easy.
- "Why serve on the Mac Mini but train on the Spark?" → The Mac Mini is enough for inference. Training needs CUDA; AReno's training path runs on NVIDIA GPUs.

**Transition**:

> We have the machine and the framework. The key question is: where does the training data come from?

---

## 4. Where the data comes from: mining bad cases from Hermes logs

**On screen**: a pipeline, `state.db` → read-only LLM analysis → `bad_cases.jsonl`; below it the four bad-case categories (hallucination highlighted), then the orange callout, then the four numbers 21 / 99 / 42 / 0.

**Say: the log pipeline**

> The Dream RSI idea goes like this: real usage produces logs, you mine the failures from the logs, turn them into training data, and train them back into the model. Today we do the smallest version.
>
> Hermes stores every session in a local `state.db`. In this step we have an LLM read the trajectories in that database directly, read-only, without changing anything. It reads each session in full, finds the conversations with problems, and writes them to `bad_cases.jsonl`, one per line, with the original text of the turn that went wrong and a note on what the problem was.

**Say: the four categories (point at them)**

> The LLM looks for four kinds of problems.
>
> One, intent errors: it misunderstood what the user wanted and answered something else.
> Two, hallucination: wrong facts, invented details, or claiming something released after the cutoff doesn't exist.
> Three, tool errors: it should have called a tool and didn't, called the wrong one, or passed the wrong arguments.
> Four, incomplete: the conversation stopped midway without a final answer.
>
> The flash-Fin answer falls into category two, hallucination.

**Say: why an LLM (point at the orange callout)**

> Why not just write rules to filter? Because the flash-Fin session ran to completion. No error, no interruption. Structurally it looks exactly like a normal Q&A. To see that the answer was wrong, you have to actually understand the content, so we hand it to an LLM.

**[LIVE]** Open `bad_cases.jsonl` and show the flash-Fin entry's `category`, `problem`, and original text. How many sessions were analyzed and how many fall in each category come from the real output only. If it hasn't been run, don't give numbers.

**Say: the dataset**

> So do we train directly on that log entry? No.
>
> A wrong answer in the logs has no correct answer in it; it only shows that something is wrong. Knowledge injection needs **reference answers**. So we compiled public material into 21 facts and 99 Q&A pairs, each row citing its source, in `examples/sft/ling_flash_fin`.
>
> The training set has 99 rows: 51 in Chinese, 48 in English.
> There's also an eval set of 42 rows, 21 in each language. They cover the same facts with different phrasing and **were never in the training set**. The overlap between the two is zero. That's how we check the model actually learned instead of memorizing the questions.
>
> So in this round, the bad cases the LLM finds tell us which knowledge the model is missing. They don't go into the training set themselves; later they can serve as negative samples for DPO or GSPO.

**Possible questions**:

- "Are 99 examples enough?" → For injecting knowledge about one topic, it's a minimal amount that works for a demo. Whether it generalizes is what the 42 holdout questions in section 6 are for.
- "Did the length budget drop any rows?" → The budget is 128 prompt tokens and 512 response tokens, and anything over is silently dropped. The longest answer is 1,216 characters, probably within 512 tokens, but the `stage=sft_dataset_filter` line in the training log is what counts. **[LIVE]** Fill in the real `skipped_long_or_empty` count.

**Transition**:

> The data's ready. Training really is one command.

---

## 5. LoRA SFT

**On screen**: the command (`areno train --algo sft`, `--disable-thinking`, `--lora-rank 16` highlighted in orange); to the right, the "99 ÷ 2 = 50, × 10 = 500" math and the three hard rules; at the bottom, the checkpoint ruler.

**Say: the command**

> This is the entire training step. A few flags worth pointing out.
>
> `--algo sft` selects supervised fine-tuning.
> `--ckpt` with `--model-hub modelscope` pulls tiny's weights from ModelScope automatically.
> `--dataset-path` points at those 99 rows.
> `--lora-rank 16` means we train only the adapter.
> `--disable-thinking` turns off the thinking block. More on that in a moment.
> `--activation-checkpointing` and `--adam-4bit` save memory.
> Learning rate 1e-4, decaying to 1e-5.

**Say: step count**

> Here's the step math: 99 rows, 2 per step, so 50 steps per epoch. Ten epochs makes 500 steps. We save every 50 steps, so we end up with 10 checkpoints, from step_000050 to step_000500.

**Say: the three hard rules**

> Three things you have to get right.
>
> First, **there's no final checkpoint**. AReno's SFT saves only when the step is a multiple of 50, and doesn't save an extra copy at the end. With 500 steps that's fine. But if the budget drops too many rows and the total falls under 50 steps, the save directory ends up empty.
>
> Second, **use `--disable-thinking` for both training and serving**. If training has no thinking block but serving does, the template adds something the model never saw, and the output gets strange.
>
> Third, **new data or new hyperparameters means a new save directory**. Otherwise old and new checkpoints end up mixed together, and it's easy to load the wrong one in a demo.

**Say: the ruler**

> Finally, this ruler. There's a checkpoint every 50 steps, and the earlier ones are less trained.
>
> That's actually useful. Knowledge injection overfits easily. The typical symptom is that whatever you ask, it opens with the same boilerplate paragraph. When that happens, you don't retrain; you go back to an earlier checkpoint, such as step_000200.
>
> By default we try step_000500 first.

**[LIVE]** If you have the training log, show a few `train_stats` lines with loss going down. The runbook has no loss numbers, so don't quote any from memory.

**Transition**:

> Training's done. Did it work? Let's look.

---

## 6. Results: base vs adapter

**On screen**: the question "What is Ling-3.0-flash-Fin?" at the top; two boxes below, red "Before :8001" on the left and green "After :8000" on the right, each with a [live output] placeholder. Further down, four clickable checks, then three columns: holdout, control group, overfitting signal.

**Say**:

> We started two servers from the same base model. Port 8001 is the original tiny. Port 8000 is the original tiny plus the adapter we just trained. We ask both the same question: "What is Ling-3.0-flash-Fin?"

**[LIVE]** Switch to the terminal and run the `for p in 8001 8000` curl loop from section 6 of the runbook, 8001 first, then 8000. Read out the key sentences from both, or paste them into the two boxes on the page.

**Say: how to judge (click the checks as you go)**

> To judge whether it got better, don't look at how smooth the wording is. Look at the **facts**. Did it mention these four:
>
> Total and active parameters, 124B and 5.1B; 256K context; the MIT license; and 7 benchmarks, including FinFIRST.

(Click each check on the page as you hear it. Only tick what the adapter actually said.)

**Say: holdout and control group**

> Getting the training questions right isn't enough. We pick 6 of the 21 English eval questions, phrased in ways it never saw in training. If it answers those correctly, it learned; it didn't just memorize.
>
> We also have a control group: one math question, 15 times 13, and one general-knowledge question, what is carbon dioxide. This checks that the training didn't wipe out what it already knew.

**[LIVE]** Decide based on time whether to run the holdout check and the control group. 15 × 13 is 195.

> If every answer starts with the same opening paragraph, it has overfit. Switch to an earlier checkpoint and try again.

**Transition**:

> That's the full loop we wanted to show today. Let me close with what it means.

---

## 7. A good start

**On screen**: three cards (Today / Next / Why this pairing).

**Say**:

> We had limited time today, so this was the simplest example of a model improving: a new model ships, the model we use doesn't know it, we find the wrong answer in real logs, build a dataset, run LoRA SFT, and it gets better on the spot.
>
> Harder settings, like office work or specific industries, need more than 99 Q&A pairs. You need a much richer eval set and mocked environments to make sure "better" really is better, and not just a few hundred memorized sentences. That's a bigger project. It's worth doing, but one step at a time.
>
> Still, I think AReno plus tiny is a good start. The model is small enough to retrain locally again and again. Training and serving share one CLI. And the path from real usage to logs to a dataset already works.
>
> RSI gets built one step at a time. Today, it started with one wrong answer.

**Close**:

> Thank you.

---

## Appendix: Q&A notes

| Likely question | Key points |
| -- | -- |
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
