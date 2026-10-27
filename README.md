# Kernel Design Agents（昇腾开发工作流）

Kernel Design Agents (KDA) is an agent-centric workflow for using coding agents to research, implement, verify, and iterate on performance-sensitive kernel tasks.

本仓库是基于 [mit-han-lab/kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents)（经 NVlabs/kda、fangfangssj/kda 转发）的**昇腾开发工作流**，面向 Ascend NPU 算子与 kernel 开发：

1. **AscendC 开发流程** —— 从任务契约、tiling 设计和实现，到 ST/精度验证、上板测量与证据记录。
2. **昇腾硬件档案** —— `hardware/` 记录 Ascend 芯片代际、CANN 工具链、验证和 profiling 约定；当前内置 Ascend C 档案。
3. **接入 cannbot 的 Skill 层** —— `skills/ascend/` 下 19 个昇腾技能（取自 [cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)，仅 Skill 层，不含其 Plugin 编排层）。

This repository documents an early research prototype. Upstream's MLSys Kernel Contest (#1–3) solutions live at [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest).

## Contents

| Path | Purpose |
|---|---|
| `docs/agent-flow.md` | Minimal end-to-end KDA workflow. |
| `docs/hardware-profiles.md` | 昇腾硬件档案说明与维护方法。 |
| `hardware/` | 内置 Ascend C 档案：`ascend-ascendc.md`，以及 Ascend 芯片档案模板。 |
| `prompts/README.md` | How to use prompt templates. |
| `prompts/basic-flow.md` | Ascend 开发任务的通用 starter prompt。 |
| `prompts/ascendc-flow.md` | Ascend C 核函数 starter prompt。 |
| `skills/ascend/` | 昇腾技能层（cannbot Skill 层快照，19 个技能）。 |
| `AGENT.md` | Repository-facing agent instructions. |
| `CONTRIBUTING.md` | Upstream contribution process（本定制版未沿用）。 |
| `THIRD_PARTY_NOTICES.md` | Third-party component and license disclosures. |

## 硬件支持矩阵

| 档案 | 硬件 | 实现语言 | starter prompt |
|---|---|---|---|
| `ascend-ascendc` | Atlas A2/A3（910B/910C） | Ascend C | `ascendc-flow.md` |

当前仓库面向昇腾开发。内置档案覆盖 Atlas A2/A3（910B/910C）；新增昇腾芯片型号或架构代际时，可按 `docs/hardware-profiles.md` 维护对应档案。

## Getting Started

```bash
git clone https://github.com/fangfangssj/kda.git
cd kda

# link Ascend skills (cannbot skill layer)
mkdir -p ~/.claude/skills
for d in "$(pwd)"/skills/ascend/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$d" ~/.claude/skills/"$(basename "$d")"
done

# optional: humanize plugin（Claude Code 内执行）
# /plugin marketplace add PolyArch/humanize && /plugin install humanize@PolyArch
```

昇腾技能需要本机具备 CANN ≥ 8.x + torch_npu 环境；技能本身只按 description 触发，全量 symlink 不会污染无关会话。

## Minimal Flow

1. Create a separate implementation workspace for the target task.
2. Define the Ascend task contract: objective, **Ascend hardware profile**, target SoC/NpuArch, constraints, validation command, and promotion criteria.
3. Start an agent session in the implementation workspace.
4. Copy `hardware/ascend-ascendc.md` into the task workspace as `hardware-profile.md`; record any task-specific SoC or CANN details in the task contract.
5. Give the agent the matching starter prompt（`basic-flow` / `ascendc-flow`），filled in with the task-specific details. The agent reads `hardware-profile.md` first, then writes a plan draft to `docs/draft.md`.
6. Convert the draft into an executable plan, either manually or with a planning tool such as Humanize.
7. Implement in small iterations, verifying after each meaningful change（AscendC 候选按"算法变体 / 配置扫描"分组推进）。
8. Record candidates, benchmark results, profiling evidence, and final promotion decisions.

## Recommended Workspace Layout

Use this repository as reference material, then do implementation work elsewhere:

```text
task-workspace/
  docs/
    draft.md
    plan.md
  runs/
  outputs/
  profile/
  benchmark.csv
  candidates.jsonl
  hardware-profile.md   # Ascend 芯片与工具链约定
```

`skills/ascend/` 内容取自 cannbot-skills（CANN Open Software License v2.0，快照 `c1ad585`），仅个人使用；再分发请遵守其许可并保留来源说明（见 `skills/ascend/README.md`）。

## License

Except for the third-party components identified above, first-party documentation, prompts, skills-style content, and assets are licensed under the [Creative Commons Attribution 4.0 International License](LICENSE), and first-party source code is licensed under the [Apache License 2.0](LICENSE). 本定制版新增文件沿用同样许可（个人使用）。
