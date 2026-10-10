# Ling-3.0-tiny Hermes Tool-Call Distillation SFT Example

Ling-3.0-tiny(总参数 7.9B、激活参数 1.3B 的混合推理 MoE 模型)的 base checkpoint
不会按 Hermes 的风格稳定生成结构化工具调用,尤其不会驱动真实技能里的多步
CLI 工作流。本示例用「**Google 邮箱登录**」这个真实 episode 做蒸馏目标:

```
用户:帮我登录我的谷歌邮箱
模型:skill_view("google-workspace") + skill_view("himalaya")   # 发现两个同类技能
模型:clarify(选哪条路线?)                                       # 让用户二选一
工具:{"user_response": "邮箱 + 日历/云盘/文档等(OAuth 授权)"}
模型:clarify(client_secret JSON 路径?)
工具:{"user_response": "文件路径是:~/Downloads/client_secret_xxx.json"}
模型:terminal(setup.py --client-secret ...)   → 凭证已保存
模型:terminal(setup.py --auth-url)            → 返回授权链接
模型:<纯文本>请你打开此链接授权
用户:http://localhost:1/?code=4/0A...
模型:terminal(setup.py --auth-code '...')     → AUTHENTICATED
模型:<纯文本>登录完成,账号 sha7tang@gmail.com …
```

本示例实现报告 §1–§5 的工程闭环:先用 LangGraph 搭一个
Planner → Router → ToolExecution → Verifier → Summarizer 工作流作为 harness,
让 tiny 在「思考-行动-观察」循环里产生执行轨迹,再把轨迹转成 SFT 数据、用 LoRA
微调,使模型在部署时无需外部 harness 即可自主完成这类多步工具任务。

理论基础是两篇论文:**Harness-Zero** 证明 harness 诱导的行为可以蒸馏进权重
(在 28 种行为上平均恢复 82.3%);**FlowMind** 的 Execute-Summarize 框架先让模型
用工具完成任务产生轨迹(Execute),再从轨迹中重构结构化工作流(Summarize)。
LangGraph 的 Checkpoint 机制把每次状态变更落成一串 checkpoint,作为天然轨迹
记录器(报告 §3.1)。

## 为什么迁移真实的 Hermes 资产

早期版本的这份示例自造了一组 `list_emails` / `read_email` / `search_emails` /
`send_email` 邮件工具。Harness-Zero 的核心警告是:**在一种 action space 下采集的
轨迹,迁移到 action space 不同的目标上效果很差**(报告 §5.1)。而 Hermes 里根本
没有这些工具——真实的邮件能力是「`skill_view` 读技能 + `terminal` 跑技能里的
CLI 脚本」。所以本版本把 Hermes 的真实资产原样搬进示例:

- **系统提示**:`SOUL.md` 身份块 + Hermes 帮助指引 + `## Skills (mandatory)`
  技能索引(含 `<available_skills>`),见 `mail_graph.py`。
- **工具 schema**:`skills_list` / `skill_view` / `clarify` / `terminal`,描述文本
  取自 Hermes(见 `tools.py` 的注释来源)。
- **技能正文**:`skills/` 目录下原样拷贝的 `google-workspace` 与 `himalaya`
  `SKILL.md`(含 `references/gmail-search-syntax.md`、`references/configuration.md`),
  与用户机器上的 Hermes 安装逐字节一致。

示例是**自包含**的:不依赖 `~/.hermes/hermes-agent` 仓库运行。技能正文只是
被 `skill_view` 当作文本读出来;`terminal` 由一个确定性的 fake shell 承载
(见下)。

## 这个示例与 AReno 的对应关系

AReno 的 SFT trainer(`areno/api/trainers/sft.py`)是一个 data→loss 的离线循环:
它从 `--dataset-loader-fn` 返回的 `prompt`/`response` 文本行算下一个 token 的
似然。它**不是** LangGraph 宿主。所以本示例把闭环拆成两段:

1. **轨迹采集**(离线,需 GPU):`collect_trajectories.py` 跑 LangGraph 工作流,
   通过 `areno serve` 起的 OpenAI 兼容端口让 tiny 充当 student,产出
   `raw_trajectories.jsonl`。
2. **微调**(LoRA,需 GPU):`areno train --algo sft --lora-rank 16` 把转换后的
   轨迹行训练进权重。

闭环里的 RL 段(`areno serve` + `--algo gspo` 分组优化)是下一阶段,不在本次范围。

## 文件

| 文件 | 作用 |
| --- | --- |
| `tools.py` | Hermes action space:`skills_list` / `skill_view` / `clarify` / `terminal` 的真实 schema + `AgentEnv`(技能发现、脚本化 `clarify` 决策)+ 确定性 fake shell + Hermes 解析/渲染。采集、转换、部署共用一份。纯标准库。 |
| `mail_graph.py` | Hermes 系统提示构建(`build_system_prompt`)+ LangGraph 工作流(Planner/Router/ToolExecution/Verifier/Summarizer)+ stdlib OpenAI 客户端。`langgraph` 惰性导入。 |
| `collect_trajectories.py` | 离线采集 CLI:读 `tasks.jsonl`,跑工作流,写 `raw_trajectories.jsonl`。**需要 GPU + 运行中的 served 模型。** |
| `build_dataset.py` | 轨迹→SFT 转换 + 筛选(含格式守卫)。纯标准库,CPU 可跑,也是 CPU 测试目标。 |
| `dataset_loader.py` | 读 `data/train.jsonl` 为 `{prompt, response}` 行,与 flash-fin 示例同契约。 |
| `skills/` | 原样迁移的 Hermes 技能(`google-workspace`、`himalaya` + references)。 |
| `data/train.jsonl` | 种子训练数据(手工轨迹 + no-tool 负例)。 |
| `data/eval.jsonl` | held-out 评测探针(与 train 不重叠)。 |
| `tasks.jsonl` | 采集器的任务输入示例。 |

## fake shell 的边界

`terminal` 不执行任意命令,只**模式匹配** SKILL.md 里写明的命令形状,回放确定
性输出:

- `setup.py --check` → `NOT_AUTHENTICATED …` (exit 1)
- `setup.py --client-secret <path>` → `OK: Client secret saved …`
- `setup.py --auth-url` → 一条形状真实的 Google OAuth URL
- `setup.py --auth-code <url>` → `OK: Authenticated … AUTHENTICATED …`
- `google_api.py gmail search/get …` → 固定的未读邮件 JSON

**未识别的命令返回 `command not found`(exit 127),不静默成功**,这样采集轨迹
里一个拼错的调用会直接暴露出来。这样既离线可复现,又保留了真实 agent 会发出的
确切命令字符串。

## 数据的形状

每条被采纳的轨迹按 **assistant step 展开成多行 SFT**:`prompt` = 渲染到当前
tool observation 为止的对话(system 身份/索引 + tools JSON + user + 既有
assistant/tool 历史),`response` = 下一步的 assistant 输出(一个 Hermes
`<tool_call>{json}</tool_call>` 块,或最终的自然语言回答)。tool result 是上下文
不是标签,只进 `prompt`。

另外加了 **no-tool 负例**(纯闲聊→直接文字回答,不含工具调用块)。报告 §5.2 引了
一项 Llama-3.2-1B 的研究:在全是工具调用的数据集上做 SFT,模型在 irrelevance
类别上从 35.8% 骤降到 5.8%,学会了「有工具就调用」。负例是对策。

| 文件 | 字段 |
| --- | --- |
| `data/train.jsonl` | `prompt`, `response`, `lang`, `trajectory_id`, `step` |
| `data/eval.jsonl` | `prompt`, `reference`, `lang`, `task_id`(与 train 的 task_id 不重叠) |

种子数据由 `build_dataset.py` 内嵌的 `SEED_TRAJECTORIES` 生成,确定性可复现。

## 种子数据自测(无需 GPU)

```bash
cd examples/sft/ling_tiny_toolcall
python build_dataset.py        # 从内嵌种子重生成 data/{train,eval}.jsonl
uv run --no-sync python -m pytest tests/test_sft_ling_tiny_toolcall_example_cpu.py -v
```

## 一、搭工作流并采集轨迹(需 GPU)

先 serve 一个 Ling-3.0-tiny(OpenAI 兼容端口):

```bash
uv run --no-sync areno serve --model-path inclusionai/ling-3.0-tiny \
  --model-hub modelscope --disable-thinking --tp-size 1 --world-size 1 \
  --port 8000 --max-running-prompts 1 &
```

装 LangGraph(只在采集这一步需要,别写进依赖):

```bash
uv run --no-sync --with langgraph python collect_trajectories.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model inclusionai/ling-3.0-tiny \
  --tasks tasks.jsonl \
  --output raw_trajectories.jsonl
```

采集器对每个任务跑一遍工作流,写一行 `{task_id, lang, prompt, done, tool_calls,
messages}`。轨迹是否被采纳由后续 `build_dataset.py` 的筛选决定:只有 Verifier
标记 `done` 且每个 assistant 工具调用块都是合法 Hermes(有效 JSON、已知工具名、
参数匹配)的轨迹才保留(报告 §5.3)。

把真实采集结果转成训练数据(替换内嵌种子):

```bash
python build_dataset.py --trajectories raw_trajectories.jsonl
```

不接 `--trajectories` 时用内嵌手工种子轨迹,产出与仓库里已提交的
`data/{train,eval}.jsonl` 字节一致,用于无 GPU 端到端冒烟。

## 二、LoRA 微调(需 GPU)

```bash
uv run --no-sync areno train --algo sft --ckpt inclusionai/ling-3.0-tiny \
  --model-hub modelscope \
  --dataset-path examples/sft/ling_tiny_toolcall/data/train.jsonl \
  --dataset-loader-fn examples/sft/ling_tiny_toolcall/dataset_loader.py \
  --tp-size 1 --world-size 1 --batch-size 2 --mini-bs 1 --epochs 20 \
  --max-prompt-tokens 1024 --max-new-tokens 256 --disable-thinking \
  --activation-checkpointing --adam-4bit --lr 1e-4 --min-lr 1e-5 --lora-rank 16 \
  --save-path outputs/ling-tiny-toolcall-lora --save-interval 30
```

- 传 loader。不传会连 `eval.jsonl` 一起读,其 schema 不同,加载会报错。
- SFT trainer 只按 `--save-interval` 存 checkpoint,结束不再存一次,所以间隔要
  比总步数小。
- 真实的 `skill_view` 结果(整篇 SKILL.md + linked_files)会让 prompt 显著变长。
  `--max-prompt-tokens` 要留够余量;超长的 assistant step 会被 trainer 静默丢弃
  (`areno/api/trainers/sft.py` 的超预算分支),此时调高预算或先做截断,不要默认
  以为全收。

## 三、对比效果

LoRA 与裸 base 各起一个端口,在 held-out 问题上对比(带 tools 应调用;裸问应
直接回答):

```bash
uv run --no-sync areno serve --model-path inclusionai/ling-3.0-tiny \
  --model-hub modelscope --lora-adapter-path outputs/ling-tiny-toolcall-lora/step_XXX \
  --disable-thinking --tp-size 1 --world-size 1 --port 8000 --max-running-prompts 1 &
```

用 `tools.TOOLS` + `mail_graph.build_system_prompt(AgentEnv.default())` 组请求,
按 `data/eval.jsonl` 的 held-out 问题逐条对比 base / LoRA。训练后 LoRA 应稳定
输出合法工具调用并驱动工作流;base 通常格式崩溃或直接乱答。

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
  里的工具调用执行与错误处理实现 harness 的其余职责。