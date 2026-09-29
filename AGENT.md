# Agent Instructions

本仓库是 Kernel Design Agents (KDA) 的**多硬件扩展版**工作流参考（个人定制版，基于 mit-han-lab/kernel-design-agents）。它应当保持小而任务无关（task-agnostic）。

## Repository Rules

- 本仓库面向个人使用：文档、提示词、注释与提交信息以**中文为主**，术语、命令、代码保持英文。
- 任务专属内容（提示词、数据集、校验器、生成的实现、benchmark 日志、候选产物）一律不进本仓库。
- benchmark 竞赛（含 MLSys 类工作）视为 KDA 的下游应用，不属于本仓库职责范围。
- 生成产物放入 `runs/`、`outputs/` 或 `profile/`（这些路径被 git 忽略；本仓库自身仅作参考，不承载任务产物）。
- 优先沉淀可复用机制：**硬件档案 > 通用流程文档 > 任务私有 harness**。

## Hardware Profiles（硬件档案）

绑定硬件的机制全部参数化到 `hardware/` 目录：

1. 每个任务契约必须声明 `Hardware profile` 字段，指向任务工作区中可读取的档案；推荐使用工作区根目录的 `hardware-profile.md`。
2. 若实现工作区独立于本仓库，启动 agent 前先把选定的内置 `hardware/<profile>.md` 复制到该文件；自定义档案也放在工作区内。
3. agent 在写 `docs/draft.md` **之前必须先读 profile**，以其命令约定、精度默认值、纪律条款为默认值；任务契约可覆盖 profile。
4. 新硬件 = 复制 `hardware/TEMPLATE.md` 填写 + 在 `docs/hardware-profiles.md` 登记；不要为单个任务直接修改内置档案。
5. 内置档案：`nvidia-cuda`、`nvidia-triton`、`ascend-ascendc`、`ascend-triton`。

## Expected Agent Workflow

For a new task:

0. Create or enter a separate implementation workspace.
1. 选定并声明硬件档案（见上节），复制到任务工作区的 `hardware-profile.md`，并把 `prompts/basic-flow.md` / `prompts/triton-flow.md` / `prompts/ascendc-flow.md` 之一作为 starter prompt。
2. Define the task objective, constraints, validation command, and promotion criteria.
3. Read local task code and documentation before proposing implementation changes.
4. Write the initial plan draft to `docs/draft.md` inside the task workspace.
5. Convert the draft into an executable plan.
6. Implement in small iterations, validating each meaningful candidate（Triton/AscendC 候选必须标注"算法变体 / 配置扫描"分组）。
7. Record candidate relationships, evaluation results, and profiling evidence when applicable.
8. Keep this repository focused on the reusable flow.

## Optional Skills（按硬件档案映射）

Use external skills only when they are relevant to the active task:

- **NVIDIA（CUDA/Triton）**：`skills/KernelWiki`（内核优化知识）、`skills/ncu-report-skill`（NCU 报告解读）。
- **昇腾（AscendC/Triton）**：`skills/ascend/` 下的 cannbot 技能族（npu-arch、ascendc-*、triton-*、ops-profiling、msnpureport-toolkit、asys-toolkit、cann-env-setup 等），映射关系见 `hardware/ascend-*.md` 的技能映射表。
- `humanize` 插件用于计划生成与实现循环（可选）。
- 领域知识技能用于背景调研；profiling / 报告分析技能用于性能证据。

## Provenance

- 工作流骨架：[mit-han-lab/kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents)（经 NVlabs/kda、fangfangssj/kda 转发）。
- 昇腾技能层：[cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills) 的 Skill 层快照（`c1ad585`），详见 `skills/ascend/README.md`。
