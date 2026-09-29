# Prompt Templates

This directory stores generic prompts for the Kernel Design Agents workflow.

The templates are intentionally task-agnostic. Fill in the task objective, **hardware profile**, constraints, validation command, and promotion criteria before starting an agent session in a separate implementation workspace.

## Available Templates

| Path | Purpose | 配套硬件档案 |
|---|---|---|
| `basic-flow.md` | Minimal prompt for research, planning, implementation, validation, and iteration. | 任选（CUDA 推荐） |
| `triton-flow.md` | Triton kernel 专用：算法变体 vs 配置扫描分组、autotune 纪律、profile 约定的验证/测量/归因。 | `nvidia-triton` / `ascend-triton` |
| `ascendc-flow.md` | Ascend C 核函数专用：环境检查 → tiling 设计 → 模板脚手架 → ST/精度验证 → msprof 归因。 | `ascend-ascendc` |

## How To Use

1. Create or enter the task implementation workspace.
2. Copy the selected built-in profile from `hardware/` into the task workspace as `hardware-profile.md`; put custom profiles in the task workspace too.
3. Copy the relevant template content into the agent session.
4. Replace placeholders with task-specific details（**必须填写 Hardware profile 字段**，指向工作区内的 `hardware-profile.md` 或自定义档案）。
5. Ask the agent to read the hardware profile first, then read the workspace and write `docs/draft.md`.
6. Convert that draft into an executable plan.
7. Run the implementation loop with validation after each meaningful change.

Task-specific prompts should live with the task they describe. Do not add benchmark-specific datasets, acceptance tables, or private evaluator details to this generic repository.
