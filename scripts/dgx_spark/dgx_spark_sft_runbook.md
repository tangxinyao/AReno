# 从一台 Mac Mini 开始：给 Ling-3.0-tiny 加训，让模型学会它不知道的新知识

> 一句话场景：Ling-3.0-tiny 的知识截止在 2026-08-06，而 Ling-3.0-flash-Fin 是它**之后**才发布的新模型。我们平时在 hermes 里问它「什么是 Ling-3.0-flash-Fin？」，它会答错。本文用一台 Mac Mini + 一台 DGX Spark + AReno，把这个「发现乱答 → 从日志洗出 bad case → 生成训练集 → LoRA SFT → 当场变好」的 RSI 闭环最小版走一遍。

> 执行环境：本文档在 Mac 上写，**没有在任何一台机器上实际跑过 `areno` 命令**。文中所有命令都在 **DGX Spark 上的仓库根目录**执行（`--dataset-path` / `--dataset-loader-fn` 是仓库相对路径）；跑通之后，把真实数字填进对应小节。

---

## 一、这是一台 Mac Mini

![Mac Mini 48G](assets/mac-mini.png)

tiny 就住在里面

**Ling-3.0-tiny 是 inclusionAI 开源的模型，MoE 架构：总参 7.9B，推理时只激活 1.3B，上下文 128K。** 两个数字要分开看：

- **总参数量 7.9B → 影响智能水平**：知识装在专家里；
- **激活量 1.3B → 影响推理开销**：每次前向只激活约六分之一，算得少，自然快。

也就是 **1.3B 的推理开销，跑出 7.9B 的知识面**。**一台 Mac Mini（¥6,999，16G 起步）整机就能跑**（deck 标注 168 tokens/s，来自 Artificial Analysis）。

### 安装非常方便：四步约 30 分钟

照着 deck 的部署路线图那五页（`RoadmapOverview` / `StepDownload` / `StepServe` / `StepVerify` / `StepWire`）来就行，服务是 OpenAI 协议的，现成的客户端直接就能连：

| 步骤 | 动作 | 耗时 |
| -- | ---- | ---- |
| 1 下载权重 | `snapshot_download('inclusionAI/Ling-3.0-tiny-GGUF', ...)`，Q8_0 最少 16G 显存 | 约 15 分钟 |
| 2 起推理服务 | `brew install llama.cpp && llama-server -m Ling-3.0-tiny.gguf` | 累计约 30 分钟 |
| 3 发第一个请求 | `client.chat.completions.create(...)` | 约 1 分钟 |
| 4 改一行接入 | `base_url = "http://localhost:8080/v1"` | 约 1 分钟 |

---

## 二、但好用的模型也会遇到问题，举个真实例子

模型再小再方便，也有一个硬边界：**知识截止**。tiny 的知识截止在它发布那天（2026-08-06），它不知道之后的世界。

所以当你平时在 hermes 里问——

> **问：Ling-3.0-flash-Fin 是什么？**

——tiny 会说没有这个模型，然后开始乱编。**模型乱答，不是它笨，是它的知识里没有这个东西。**

这个例子来自 hermes 的历史日志，第 4 节会讲怎么把它找出来。

---

## 三、那我们就让它学会：一台 DGX Spark + AReno

要改掉模型的行为，最轻的办法是**加训（LoRA SFT）**：base 权重全程冻结，只训一个 rank 16 的 adapter——几十分钟就能重训一遍，改坏了就删，base 始终是干净的那一份。

加训需要一台能训练的机器：

![DGX Spark 128G](assets/dgx-spark.jpg)

### 这是一台 DGX Spark

- **价格 ¥32,999，128G 统一内存**（deck P4 规格表）；
- deck 里它跑的是 Ling-3.0-flash（124B / 激活 5.1B / 345 tokens/s）；我们拿它给 tiny 训 LoRA，单卡统一内存绰绰有余；
- 这次要训的量极小：**99 条问答 × 10 epoch = 500 步**，十分钟级别。

### AReno：训练和推理收在同一个 CLI 里

AReno 是我们自己的训练和服务框架，一个 CLI 里同时有 `areno train`（SFT / DPO / GSPO / GRPO / PPO）和 `areno serve`（OpenAI 兼容的 `/v1/chat/completions`）。**训完直接起服务**，接任何 OpenAI 客户端就行——不上云，也不用再搭推理栈。

```bash
git clone git@github.com:tangxinyao/AReno.git ~/AReno
cd ~/AReno
uv sync                       # 带 torch>=2.6 等依赖，生成 .venv
source .venv/bin/activate     # 之后才有路径上的 areno；或者用 uv run areno ...
which areno
python -c "import torch; print('GPU:', torch.cuda.is_available())"   # 期望 True
```

> 前置：Linux + NVIDIA GPU + CUDA（Spark 自带）。`uv sync` 会连 `areno/accel` 的 CUDA 扩展一起编，**需要先有 torch>=2.6**；编不过就先 `ARENO_BUILD_EXT=0 uv sync` 只装元数据，GPU 训练前再补编译。

**模型两种拿法（二选一）**：训练时 `--model-hub modelscope` 自动下载，不用你自己存；第 6 节 serve 要本地路径，可以提前下好：

```bash
modelscope download --model inclusionAI/Ling-3.0-tiny --local_dir /home/tangxinyao/Ling-3.0-tiny
```

> ⚠️ **ModelScope id 的大小写**：`inclusionai/ling-3.0-tiny`（小写）在训练命令里能拉；拉不到就换 `inclusionAI/Ling-3.0-tiny`。

---

## 四、训练数据哪来：从 hermes 日志里洗「bad case」

**Dream RSI** 的思路是：真实使用 → 日志 → 洗出失败样本 → 变成训练数据 → 训回去；我们这里做的就是它的最小版。

hermes 每次会话都落在 `$HERMES_HOME/state.db`（默认 `~/.hermes`）。`export_sharegpt.py` 对它是**只读**的，把真实轨迹导成 ShareGPT：

```bash
mkdir -p outputs/hermes-collect
# TODO(agent): 本仓库里没有 export_sharegpt.py，路径和参数沿用原 runbook，未验证
python3 .agents/skills/areno-collect-hermes-history/scripts/export_sharegpt.py \
  --out outputs/hermes-collect/all.jsonl \
  --min-turns 1 --summary
```

### 用 LLM 分析 happy / bad case

导出的轨迹**直接交给 LLM 逐条分析**，由它判断每段对话是 happy 还是 bad。判断的是「答得对不对」，而不只是「跑完了没有」：第 2 节那次乱答，对话本身是完整跑完的，只有读懂内容才能看出它答错了。

| 分类 | LLM 的判定 | 用途 |
| ---- | ---- | ---- |
| **happy case** | 回答正确，完成了用户的请求 | 将来 SFT 正样本候选 |
| **bad case** | 答错、编造，或没完成用户的请求 | 这一轮用来定位模型缺哪块知识；将来可做 DPO / GSPO 的 rejected 侧 |

第 2 节 flash-Fin 那次对话，就是这一步里被判成 bad case 的那一条。

<!-- TODO(agent): 补上实际使用的 LLM 分析脚本、模型和判定 prompt，以及跑出来的 happy / bad 条数 -->

> ⚠️ **这一步在哪台机器跑**：hermes 历史只在**运行过 hermes 的机器**上。Spark 上跑过就直接跑（必要时给 `--search-root`，多个就重复该参数）；历史在你的 Mac，就在 Mac 上跑同一条命令，把 `all.jsonl` `scp` 过来。

### 这份训练集长什么样

**训练数据不是 hermes 轨迹本身**：乱答里没有标准答案。知识注入要的是规范答案，所以把公开材料整理成 **21 个知识点 / 99 条问答**（`examples/sft/ling_flash_fin`，每行带 `sources`）。LLM 挑出的 bad case 告诉我们模型缺的是哪块知识，规范答案则要从公开材料里整理。

```bash
wc -l examples/sft/ling_flash_fin/data/train.jsonl examples/sft/ling_flash_fin/data/eval.jsonl
jq -r '.lang' examples/sft/ling_flash_fin/data/train.jsonl | sort | uniq -c       # zh 51 / en 48
jq -r '.response | length' examples/sft/ling_flash_fin/data/train.jsonl | sort -n | tail -1   # 最长答案字符
comm -12 <(jq -r '.prompt' examples/sft/ling_flash_fin/data/train.jsonl | sort) \
         <(jq -r '.prompt' examples/sft/ling_flash_fin/data/eval.jsonl   | sort) | wc -l        # 期望 0
```

- `train.jsonl`：**99 条**（zh 51 / en 48），`prompt` 问题 / `response` 规范答案；
- `eval.jsonl`：**42 条**（zh 21 / en 21），**没进过训练集**的改写提问 holdout，带 `reference`；
- loader（`dataset_loader.py`）只读 `train.jsonl`，把 `data/` 目录传给它也会收窄，不会误读 `eval.jsonl`。

**预算会不会砍行**（`prompt<=128 / response<=512` token，超了**静默丢**）：prompt 都短没问题；最长 response 1216 字符，中英混合下大概率在 512 token 内，但**只有第 5 节训练日志里的一行 `stage=sft_dataset_filter` 说了算**。

---

## 五、LoRA SFT：把 21 个知识点灌回去

下面就是全部动作——`areno train` 一条命令：

```bash
areno train --algo sft --ckpt inclusionai/ling-3.0-tiny --model-hub modelscope \
  --dataset-path examples/sft/ling_flash_fin/data/train.jsonl \
  --dataset-loader-fn examples/sft/ling_flash_fin/dataset_loader.py \
  --world-size 1 --tp-size 1 \
  --batch-size 2 --mini-bs 1 --epochs 10 \
  --max-prompt-tokens 128 --max-new-tokens 512 \
  --disable-thinking --attn-backend flash \
  --activation-checkpointing --adam-4bit \
  --lr 1e-4 --min-lr 1e-5 --lora-rank 16 \
  --save-path /home/tangxinyao/fin_sft_lora --save-interval 50
```

**step 推算**：99 行 / `batch-size 2` = **50 步/epoch**，10 epoch = **500 步**；`save-interval 50` → `step_000050 ... step_000500` 共 10 个 checkpoint。

挂后台 + 看进度（分两个终端）：

```bash
mkdir -p runs
nohup areno train --algo sft --ckpt inclusionai/ling-3.0-tiny --model-hub modelscope \
  --dataset-path examples/sft/ling_flash_fin/data/train.jsonl \
  --dataset-loader-fn examples/sft/ling_flash_fin/dataset_loader.py \
  --world-size 1 --tp-size 1 \
  --batch-size 2 --mini-bs 1 --epochs 10 \
  --max-prompt-tokens 128 --max-new-tokens 512 \
  --disable-thinking --attn-backend flash \
  --activation-checkpointing --adam-4bit \
  --lr 1e-4 --min-lr 1e-5 --lora-rank 16 \
  --save-path /home/tangxinyao/fin_sft_lora --save-interval 50 \
  > runs/train.log 2>&1 &
```

```bash
# 终端 B：关键日志行（loss 在 train_stats 里，盯它降不降）
grep -E "stage=|train_stats|epoch=|loss" runs/train.log | tail -n 15
# 或直接跟：
tail -f runs/train.log
```

| 日志行 | 什么意思 |
| ---- | ---- |
| `stage=sft_dataset_filter skipped_long_or_empty=N` | 被 128/512 预算砍掉 N 行（N 大就是预算给小了） |
| `epoch=E stage=epoch_start / epoch_end` | epoch 边界 |
| `train_stats={...}` | loss 在这个 dict 里 |
| `stage=save_checkpoint_start/end path=...` | 存盘，path 就是产物目录 |

三条硬口径：

1. **没有 final checkpoint。** SFT 只在 `step % save_interval == 0` 落盘，跑完不补（`areno/api/trainers/sft.py`）。500 步 >= 50，所以会有产物；万一预算把行砍到总步数 < 50，`--save-path` 就是空的。
2. **`--disable-thinking` 必须和推理一致。** 第 6 节 serve 的 adapter **和 base** 都要带，否则模板会渲染出训练时从没见过的 thinking 段。
3. 想改数据或改超参就改上面命令里的字，**换了 `--save-path`** 再重跑，否则新旧 checkpoint 混在一个目录里。

### 挑 checkpoint

```bash
ls -d /home/tangxinyao/fin_sft_lora/step_*
ls /home/tangxinyao/fin_sft_lora/step_000500/
head -c 400 /home/tangxinyao/fin_sft_lora/step_000500/adapter_config.json
```

目录名 = `step_{step+1:06d}`。越早的 checkpoint 训得越少。如果知识注入过拟合（每个问题都答成通稿式开头），就换用更早的 step，比如用 `step_000200` 而不是 `step_000500`。

---

## 六、看效果：base vs 加训后的 adapter

**同一台基座、同一个问题，两个端口分别回答** —— 这就是知识注入的全部说服力。adapter 在 `:8000`，base 在 `:8001`：

```bash
nohup areno serve --model-path /home/tangxinyao/Ling-3.0-tiny \
  --lora-adapter-path /home/tangxinyao/fin_sft_lora/step_000500 \
  --disable-thinking \
  --tp-size 1 --world-size 1 --port 8000 --max-running-prompts 1 \
  > runs/serve-adapter.log 2>&1 &

nohup areno serve --model-path /home/tangxinyao/Ling-3.0-tiny \
  --disable-thinking \
  --tp-size 1 --world-size 1 --port 8001 --max-running-prompts 1 \
  > runs/serve-base.log 2>&1 &
```

> ⚠️ **三件必须对齐**：base 模型同一个；`--disable-thinking` 训练 / 服务**都带**；`--lora-adapter-path` 给了就别再给 `--lora-rank`（形状以 `adapter_config.json` 为准）。另外 `--lora-adapter-path` 只收**一个**值，写成 `/home/tangxinyao/fin_sft_lora/step_000500` 这种单路径。

等端口起来，然后就是全场开场那一屏——同一个问题，加训前 vs 加训后：

```bash
ss -ltnp | grep -E ':(8000|8001)'
curl -s http://127.0.0.1:8000/v1/models | head -c 300

for p in 8001 8000; do
  echo "---- :$p"
  curl -s http://127.0.0.1:$p/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"areno","messages":[{"role":"user","content":"什么是 Ling-3.0-flash-Fin？"}],"max_tokens":512}' \
    | jq -r '.choices[0].message.content' | head -c 600
  echo
done
```

**eval holdout 抽查**（42 条里看前 6 条；`head -n 6` 改数字看更多）：

```bash
jq -r '.prompt' examples/sft/ling_flash_fin/data/eval.jsonl | head -n 6 | while read -r q; do
  echo "==== $q"
  for p in 8001 8000; do
    echo "--- :$p"
    curl -s http://127.0.0.1:$p/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"areno\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":1024}" \
      | jq -r '.choices[0].message.content' | head -c 280
    echo
  done
done
```

**通用能力控制组**（数学 / 常识各一条，确认没被冲掉）：

```bash
for p in 8001 8000; do
  echo "---- :$p"
  curl -s http://127.0.0.1:$p/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"areno","messages":[{"role":"user","content":"What is 15*13? Reply with the number and one line of check."}],"max_tokens":160}' \
    | jq -r '.choices[0].message.content'
  curl -s http://127.0.0.1:$p/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"areno","messages":[{"role":"user","content":"二氧化碳是什么？用一句话回答。"}],"max_tokens":160}' \
    | jq -r '.choices[0].message.content'
done
```

重点看这几个数：**adapter 有没有说出 124B/5.1B、256K、MIT、含 FinFIRST 在内的 7 个基准**，而不是措辞是不是更顺；控制组没崩，说明通用能力没被冲掉。若每个问题都答成同一段通稿式开头 → 过拟合，回第 5 节换更早的 `step_*`。

收尾：

```bash
pkill -f "areno serve" || true
pkill -f "areno train" || true
ss -ltnp | grep -E ':(8000|8001)' || echo "端口已释放"
nvidia-smi
```

---

## 七、这是一个好的开始（最后）

今天时间有限，我们讲的是**最简单的模型进化例子**：一个新模型发布 → 我们用的模型不知道 → 在真实日志里发现它乱答 → 洗出 bad case → 整理成训练集 → LoRA SFT → 当场变好。

更复杂一点的，是办公、行业这类领域：那就不能只看 99 条问答，要**构造足够复杂的评测集、mock 各种环境**，才能确认「变好」是真的变好，而不是背了几百个字。那是更大的工程，值得，但要一步步来。

不过 **AReno + tiny 这个组合是一个好的开始**：模型小到本地就能反复训，训练和推理收在同一个 CLI 里，数据链是「真实使用 → 日志 → 训练集」的现成一条路。**RSI 要一步一步做，这次是从最简单的一个「乱答」开始。**

---

## 附录 · 故障排查

| 现象 | 原因 | 处理 |
| ---- | ---- | ---- |
| `SFT dataset produced no valid training rows after filtering` | 所有行超预算或空 target | 提高 `--max-prompt-tokens/--max-new-tokens`（response 超 512 token 就提到 1024）；先跑第 4 节量一遍 |
| `--save-path` 下空的，没有 `step_*` | 总步数 < `save-interval`，SFT 不补 final | 调小 `--save-interval` 或加 epochs；本配置下不该发生，除非预算把行砍太多 |
| 服务起不来 / `adapter_path must contain a PEFT LoRA artifact` | adapter 目录缺 `adapter_config.json` 或 `peft_type` 不是 LORA | 换一个完整的 `step_*`；回看训练有没有真的带 `--lora-rank` |
| `target_modules must be a non-empty subset of (...)` | adapter 的 target 不在 AReno native LoRA 支持列表里 | 训练时显式给 `--lora-target-modules`，只用支持的几个 |
| 所有回答都是同一段通稿式开头 | 99 行过拟合 | 用更早的 `step_*`（比如 `step_000200` 而不是 `step_000500`） |
| flash-attn kernel / 能力报错 | 容器里的 flash 实现和这块 GPU 不匹配 | 换 `--attn-backend native`（慢，功能一致） |
| ModelScope 拉不到 `inclusionai/ling-3.0-tiny` | repo id 大小写 / 网络 | 换成 `inclusionAI/Ling-3.0-tiny`，或提前 `modelscope download` 本地跑 |
| 端口被占 | 8000/8001 上有旧服务 | `ss -ltnp` 看一下，`pkill -f "areno serve"` 或直接 kill |
| loss 是 nan / 不降 | LR 对 LoRA 偏大，或数据里有脏 target | 先降到 `--lr 1e-5` 量级试一轮，再看 `train_stats` |

重跑不需要清理旧 `step_*`（同名会被覆盖）；但**换数据集或换超参重跑要换 `--save-path`**，否则新旧 checkpoint 混在一个目录里，演示时容易拿错。