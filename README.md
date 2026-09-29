# Kernel Design Agents（多硬件扩展版）

Kernel Design Agents (KDA) is an agent-centric workflow for using coding agents to research, implement, verify, and iterate on performance-sensitive kernel tasks.

本仓库是基于 [mit-han-lab/kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents)（经 NVlabs/kda、fangfangssj/kda 转发）的**个人定制版**，核心扩展：

1. **硬件档案（Hardware Profiles）机制** —— 把"实现语言、正确性验证约定、性能测量约定、profiling 归因、知识技能映射"参数化到 `hardware/` 目录，最小循环本身保持硬件无关。
2. **接入 cannbot 的 Skill 层** —— `skills/ascend/` 下 25 个昇腾技能（取自 [cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)，仅 Skill 层，不含其 Plugin 编排层）。
3. **Triton 算子一等公民** —— 新增 `prompts/triton-flow.md`（算法变体 vs 配置扫描分组纪律），同时覆盖 NVIDIA Triton 与 triton_ascend。

This repository documents an early research prototype. Upstream's MLSys Kernel Contest (#1–3) solutions live at [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest).

## Contents

| Path | Purpose |
|---|---|
| `docs/agent-flow.md` | Minimal end-to-end KDA workflow. |
| `docs/hardware-profiles.md` | 硬件档案机制说明 + 新硬件扩展指南。 |
| `hardware/` | 内置硬件档案：`nvidia-cuda` / `nvidia-triton` / `ascend-ascendc` / `ascend-triton` + `TEMPLATE.md`。 |
| `prompts/README.md` | How to use prompt templates. |
| `prompts/basic-flow.md` | Generic starter prompt for a new task. |
| `prompts/triton-flow.md` | Triton 算子 starter prompt（NVIDIA / Ascend 通用）。 |
| `prompts/ascendc-flow.md` | Ascend C 核函数 starter prompt。 |
| `skills/ascend/` | 昇腾技能层（cannbot Skill 层快照，25 个技能）。 |
| `AGENT.md` | Repository-facing agent instructions. |
| `CONTRIBUTING.md` | Upstream contribution process（本定制版未沿用）。 |
| `THIRD_PARTY_NOTICES.md` | Third-party component and license disclosures. |

## 硬件支持矩阵

| 档案 | 硬件 | 实现语言 | starter prompt |
|---|---|---|---|
| `nvidia-cuda` | NVIDIA A100/H100/B200/B300 | CUDA C++ | `basic-flow.md` |
| `nvidia-triton` | 同上 | OpenAI Triton | `triton-flow.md` |
| `ascend-ascendc` | Atlas A2/A3（910B/910C） | Ascend C | `ascendc-flow.md` |
| `ascend-triton` | 同上 | triton_ascend | `triton-flow.md` |

新增硬件（AMD ROCm / Intel XPU / MLU / CPU triton 等）：复制 `hardware/TEMPLATE.md` 填写一份档案 + 在 `docs/hardware-profiles.md` 登记即可，**流程骨架零改动**。

## Getting Started

```bash
git clone --recurse-submodules https://github.com/fangfangssj/kda.git
cd kda

# link NVIDIA skills
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/ncu-report-skill" ~/.claude/skills/ncu-report-skill
ln -s "$(pwd)/skills/KernelWiki" ~/.claude/skills/KernelWiki

# link Ascend skills (cannbot skill layer)
for d in "$(pwd)"/skills/ascend/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$d" ~/.claude/skills/"$(basename "$d")"
done

# optional: humanize plugin（Claude Code 内执行）
# /plugin marketplace add PolyArch/humanize && /plugin install humanize@PolyArch
```

昇腾技能需要本机具备 CANN ≥ 8.x + torch_npu 环境；技能本身只按 description 触发，全量 symlink 不会污染无关会话。

## Minimal Flow

1. Create a separate implementation workspace for the target task.
2. Define the task contract: objective, **hardware profile**, constraints, validation command, and promotion criteria.
3. Start an agent session in the implementation workspace.
4. Copy the selected built-in profile from this repository into the task workspace as `hardware-profile.md`（自定义档案也放在任务工作区内）。
5. Give the agent the matching starter prompt（`basic-flow` / `triton-flow` / `ascendc-flow`），filled in with the task-specific details. The agent reads `hardware-profile.md` first, then writes a plan draft to `docs/draft.md`.
6. Convert the draft into an executable plan, either manually or with a planning tool such as Humanize.
7. Implement in small iterations, verifying after each meaningful change（Triton/AscendC 候选按"算法变体 / 配置扫描"分组推进）。
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
  hardware-profile.md   # 任务档案：复制所选内置档案，或放入自定义档案
```

## Community Kernel Wishlist（上游功能）

Have a kernel that needs optimization? [Submit a request](https://github.com/NVlabs/kda/tree/wishlist#submit) as a pull request to upstream's `wishlist` branch, with a reproducible definition, representative workloads, and your best-known baseline implementation. The upstream community wishlist currently supports NVIDIA B200 and B300 GPUs only（本定制版不受此限制，本地任务可指向任意硬件档案）.

## Submodules（上游第三方）

| Path | Source | Pinned revision | License |
|---|---|---|---|
| `skills/KernelWiki` | [mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki.git) | `76d27b5` | MIT（嵌入产物保留上游条款） |
| `skills/ncu-report-skill` | [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill.git) | `d188794` | MIT |

Use KernelWiki from this repository's pinned submodule; a direct upstream checkout may contain artifact snapshots governed by additional terms.

`skills/ascend/` 内容取自 cannbot-skills（CANN Open Software License v2.0，快照 `c1ad585`），仅个人使用；再分发请遵守其许可并保留来源说明（见 `skills/ascend/README.md`）。

## License

Except for the third-party components identified above, first-party documentation, prompts, skills-style content, and assets are licensed under the [Creative Commons Attribution 4.0 International License](LICENSE), and first-party source code is licensed under the [Apache License 2.0](LICENSE). 本定制版新增文件沿用同样许可（个人使用）。
