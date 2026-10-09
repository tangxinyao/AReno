# Ling-3.0-tiny Tool-Call Distillation SFT Example

Ling-3.0-tiny (总参数 7.9B,激活参数 1.3B 的混合推理 MoE 模型)的 base checkpoint
没有原生的 Hermes 风格邮件工具调用能力——它不会按 `list_emails` / `read_email`
等接口稳定生成结构化调用。本示例实现报告 §1–§5 的工程闭环:先用 LangGraph
搭一个 Planner → ToolRouter → ToolExecution → Verifier → Summarizer 的邮件
Agent 工作流作为 harness,让 tiny 在「思考-行动-观察」循环里产生执行轨迹,
再把轨迹转成 SFT 数据、用 LoRA 微调,使模型在部署时无需外部 harness 即可自主
读邮件。

理论基础是两篇论文:**Harness-Zero** 证明 harness 诱导的行为可以蒸馏进权重
(在 28 种行为上平均恢复 82.3%);**FlowMind** 的 Execute-Summarize 框架先让模型
用工具完成任务产生轨迹(Execute),再从轨迹中重构结构化工作流(Summarize),
避免「边执行边输出工作流」带来的目标干扰。LangGraph 的 Checkpoint 机制负责
把每次状态变更落成一串 checkpoint,作为天然轨迹记录器(报告 §3.1)。

## 这个示例与 AReno 的对应关系

AReno 的 SFT trainer(`areno/api/trainers/sft.py`)是一个 data→loss 的离线循环:
它从 `--dataset-loader-fn` 返回的 `prompt`/`response` 文本行算下一个 token 的
似然。它**不是** LangGraph 宿主。所以本示例把闭环拆成两段:

1. **轨迹采集**(离线,需 GPU):`collect_trajectories.py` 跑 LangGraph 工作流,
   通过 `areno serve` 起的 OpenAI 兼容端口让 tiny 充当 student,产出
   `raw_trajectories.jsonl`。
2. **微调**(LoRA,需 GPU):`areno train --algo sft --lora-rank 16` 把转换后的
   轨迹行训练进权重。和已有的 Ling-flash-Fin 知识注入示例用同一条 CLI 路径。

闭环里的 RL 段(`areno serve` + `--algo gspo` 分组优化)是这份示例之后的下一
阶段,不在本次范围。

## 文件

| 文件 | 作用 |
| --- | --- |
| `tools.py` | 邮件 action space:`TOOLS` 工具 schema + `MailEnv` 执行器 + Hermes 解析/渲染。采集、转换、部署共用一份,保证 action-space 一致(报告 §5.1)。纯标准库。 |
| `mail_graph.py` | LangGraph 工作流(Planner/Router/ToolExecution/Verifier/Summarizer 节点)+ `MemorySaver` checkpointer + 一个只用 stdlib 的 OpenAI 兼容客户端。`langgraph` 惰性导入。 |
| `collect_trajectories.py` | 离线采集 CLI:读 `tasks.jsonl`,跑工作流,写 `raw_trajectories.jsonl`。**这一步需要 GPU + 运行中的 served 模型。** |
| `build_dataset.py` | 轨迹→SFT 转换 + 筛选(报告 §4.1/§5.2/§5.3)。纯标准库,可在 CPU 上跑,也是 CPU 测试的目标。 |
| `dataset_loader.py` | 读 `data/train.jsonl` 为 `{prompt, response}` 行,与 flash-fin 示例同契约。 |
| `data/train.jsonl` | 15 行种子训练数据(手工轨迹 + 4 条 no-tool 负例)。 |
| `data/eval.jsonl` | 3 行 held-out 评测探针(与 train 不重叠)。 |
| `tasks.jsonl` | 采集器的任务输入示例。 |

## 数据的形状

每条被采纳的轨迹被**按 assistant step 展开成多行 SFT**:`prompt` = 渲染到
当前 tool observation 为止的对话(system + tools JSON + user + 既有
assistant/tool 历史),`response` = 下一步的 assistant 输出(一个
Hermes `<tool_call>...` 工具调用块,或最终的自然语言回答)。tool result 是
上下文不是标签,只进 `prompt`。这样每行 prompt ~660 token、response ~36 token,
远低于训练预算,不会有行被 trainer 因超长丢弃。

另外加了 **no-tool 负例**(纯闲聊→直接文字回答,不含任何工具调用块)。报告
§5.2 引了一项 Llama-3.2-1B 的研究:在全是工具调用的数据集上做 SFT,模型在
irrelevance 类别上从 35.8% 骤降到 5.8%,学会了「有工具就调用」。负例是对策。

| 文件 | 行数 | 字段 |
| --- | --- | --- |
| `data/train.jsonl` | 15 | `prompt`, `response`, `lang`, `trajectory_id`, `step` |
| `data/eval.jsonl` | 3 | `prompt`, `reference`, `lang`, `task_id`(与 train 的 task_id 不重叠) |

种子数据由 `build_dataset.py` 内嵌的 `SEED_TRAJECTORIES` 生成,确定性可复现。

## 种子数据自测(无需 GPU)

```bash
cd examples/sft/ling_tiny_toolcall
python build_dataset.py        # 从内嵌种子重生成 data/{train,eval}.jsonl
pytest tests/test_sft_ling_tiny_toolcall_example_cpu.py -k cpu
```

## 一、搭工作流并采集轨迹(需 GPU)

先 serve 一个 Ling-3.0-tiny(OpenAI 兼容端口):

```bash
areno serve --model-path inclusionai/ling-3.0-tiny --model-hub modelscope \
  --disable-thinking --tp-size 1 --world-size 1 --port 8000 &
```

装 LangGraph(只在采集这一步需要):

```bash
pip install langgraph
```

采轨迹:

```bash
python collect_trajectories.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model inclusionai/ling-3.0-tiny \
  --tasks tasks.jsonl \
  --output raw_trajectories.jsonl
```

采集器对每个任务跑一遍工作流,写一行 `{task_id, lang, prompt, done, tool_calls,
messages}`。trajectory 是否被采纳由后续 `build_dataset.py` 的筛选决定:只有
Verdifier 标记 `done` 且每个 assistant 工具调用块都是合法 Hermes(有效 JSON、
已知工具名、参数匹配)的轨迹才保留(报告 §5.3 正向筛选 + 负向剔除)。

把真实采集结果转成训练数据(替换内嵌种子):

```bash
python build_dataset.py --trajectories raw_trajectories.jsonl
```

不接 `--trajectories` 时,`build_dataset.py` 用内嵌的手工种子轨迹,产出与仓库
里已提交的 `data/{train,eval}.jsonl` 字节一致,用于无 GPU 场景下的端到端冒烟。

## 二、LoRA 微调(需 GPU)

```bash
areno train --algo sft --ckpt inclusionai/ling-3.0-tiny --model-hub modelscope \
  --dataset-path examples/sft/ling_tiny_toolcall/data/train.jsonl \
  --dataset-loader-fn examples/sft/ling_tiny_toolcall/dataset_loader.py \
  --tp-size 1 --world-size 1 --batch-size 2 --mini-bs 1 --epochs 20 \
  --max-prompt-tokens 768 --max-new-tokens 256 --disable-thinking \
  --activation-checkpointing --adam-4bit --lr 1e-4 --min-lr 1e-5 --lora-rank 16 \
  --save-path outputs/ling-tiny-toolcall-lora --save-interval 30
```

- 传 loader。不传会连 `eval.jsonl` 一起读,其 schema 不同,加载会报错。
- 15 行、batch 2 ≈ 8 步/epoch,20 epoch ≈ 160 步。SFT trainer 只按
  `--save-interval` 存 checkpoint,结束不再存一次,所以间隔要比总步数小。
- `--max-prompt-tokens 768` 对齐 step-wise 行的实测最大 prompt(~660 token)。
  真实采集轨迹更长时,超长的 assistant step 会被 trainer 静默丢弃
  (`areno/api/trainers/sft.py` 的超预算分支),此时调高预算或先做截断,不要
  默认以为全收。
- 想增强工具调用能力,可把 `--lora-rank` 提到 32,或在数据稳定后进入下一阶段
  的 GSPO 强化。

## 三、对比效果

LoRA 与裸 base 各起一个端口,在 held-out 问题上对比:

```bash
areno serve --model-path inclusionai/ling-3.0-tiny --model-hub modelscope \
  --lora-adapter-path outputs/ling-tiny-toolcall-lora/step_000150 \
  --disable-thinking --tp-size 1 --world-size 1 --port 8000 &
areno serve --model-path inclusionai/ling-3.0-tiny --model-hub modelscope \
  --disable-thinking --tp-size 1 --world-size 1 --port 8001 &

python - <<'PY'
import json, sys, urllib.request
sys.path.insert(0, "examples/sft/ling_tiny_toolcall")
from tools import TOOLS

def ask(port, prompt, *, with_tools):
    body = json.dumps({
        "model": "inclusionai/ling-3.0-tiny", "max_tokens": 256, "temperature": 0.0,
        "tools": TOOLS if with_tools else None,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["choices"][0]["message"]

for line in open("examples/sft/ling_tiny_toolcall/data/eval.jsonl", encoding="utf-8"):
    row = json.loads(line)
    print("Q:", row["prompt"])
    for with_tools in (True, False):           # 带 tools 应调用;裸问应直接答
        for port, name in ((8001, "base"), (8000, "lora")):
            msg = ask(port, row["prompt"], with_tools=with_tools)
            kind = "tools" if msg.get("tool_calls") else "text"
            print(f"  [{name}/tools={with_tools}/{kind}]",
                  (msg.get("content") or str(msg.get("tool_calls")))[:240])
    print("  [ref] ", row["reference"][:240], "\n")
PY
```

训练后,带工具时 LoRA 模型应能输出合法的 `<tool_call>...` 调用并用结果作答;
裸问时应直接回答而不强行调用(对应负例的作用)。base 模型对这两类输入通常
都会要么不会调、要么格式崩。也问几个与邮件无关的问题,确认 LoRA 没把通用
能力破坏掉。

## 关键风险(报告 §5)

- **Action space 匹配是首要约束(§5.1)**:`tools.py` 里的工具名、参数 schema、
  调用格式,以及 `tool_result_to_text` 的 JSON 结果形状,是采集/训练/部署三处
  共用的唯一合同。改动作空间必须三处一起改,否则蒸馏效果会大幅下降。
- **小模型的「格式崩溃」与「过度调用」(§5.2)**:tiny 在微调初期容易输出非法
  JSON。`build_dataset.py` 的 `assert_valid_tool_call` 是格式守卫,任何非法块
  在转换期就会被剔除。过度调用靠 no-tool 负例压住——不要删那几行。
- **Harness ≠ LangGraph 工作流,但可映射(§5.3)**:论文里的 harness 是「模型与
  环境交互的全部中介层」(工具定义、记忆、重试策略);LangGraph 工作流是
  「节点和边的编排」。本示例把 LangGraph 工作流当作 harness 的编排部分,节点
  里的工具调用执行与错误处理实现 harness 的其余职责。训练数据的 action space
  必须覆盖模型在 LangGraph 编排里实际「看到」和「做」的全部内容。
