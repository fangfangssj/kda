# Prompt Templates

This directory stores prompts for Ascend development with the Kernel Design Agents workflow.

The templates are task-agnostic within the Ascend domain. Fill in the task objective, Ascend SoC, **hardware profile**, constraints, validation command, and promotion criteria before starting an agent session in a separate implementation workspace.

## Available Templates

| Path | Purpose | 配套硬件档案 |
|---|---|---|
| `basic-flow.md` | Ascend 任务的通用研究、计划、实现与验证流程。 | `ascend-ascendc` |
| `ascendc-flow.md` | Ascend C 核函数专用：环境检查 → tiling 设计 → 模板脚手架 → ST/精度验证 → msprof 归因。 | `ascend-ascendc` |

## How To Use

1. Create or enter the task implementation workspace.
2. Copy `hardware/ascend-ascendc.md` into the task workspace as `hardware-profile.md`; record the target SoC/NpuArch in the task contract.
3. Copy the relevant template content into the agent session.
4. Replace placeholders with task-specific details（**必须填写 Ascend hardware profile 字段**，并注明目标 SoC/NpuArch）。
5. Ask the agent to read the hardware profile first, then read the workspace and write `docs/draft.md`.
6. Convert that draft into an executable plan.
7. Run the implementation loop with validation after each meaningful change.

Task-specific prompts should live with the task they describe. Do not add benchmark-specific datasets, acceptance tables, or private evaluator details to this generic repository.
