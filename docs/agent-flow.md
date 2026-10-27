# Agent Flow

Kernel Design Agents is a repeatable workflow for agent-driven Ascend NPU operator and kernel development. It is useful when a task needs exploration, correctness checks, and evidence-based promotion decisions.

## Principle

Keep the reusable workflow separate from the task workspace. This repository documents the Ascend development loop; the task workspace owns kernel source, tests, datasets, benchmark scripts, private rules, and generated artifacts. The workflow applies across Ascend operator kernels, runtime integration, and performance optimization tasks.

## Minimal Loop

1. Define the task contract, including the hardware profile.
2. Have the agent read the hardware profile, then inspect the local workspace.
3. Make the agent write `docs/draft.md`.
4. Convert the draft into an executable plan.
5. Implement the first candidate.
6. Validate correctness.
7. Measure the target metric when applicable.
8. Record evidence and decide whether to keep, revise, or reject the candidate.
9. Repeat until the promotion criteria are met or the remaining blockers are explicit.

## Task Contract

Each task should state:

- Objective.
- **Ascend hardware profile**（推荐使用 `hardware-profile.md`；独立工作区先复制 `hardware/ascend-ascendc.md`，并在任务契约中写明目标 SoC/NpuArch）。
- Inputs and outputs.
- Correctness requirements.
- Constraints on implementation language, dependencies, APIs, or deployment.
- Validation command.
- Evaluation command, if different from validation.
- Promotion criteria.

The Ascend hardware profile records the target SoC/NpuArch, CANN toolchain, Ascend C conventions, correctness checks, measurement and profiling commands, and mapped skills. The workflow stays consistent across Ascend tasks while the task contract captures each target's details.

## Evidence Records

Use simple files in the task workspace:

- `docs/draft.md` for the first plan draft.
- `docs/plan.md` for the executable plan.
- `benchmark.csv` or another tabular log for measurable results.
- `candidates.jsonl` for candidate names, parent links, and status.
- `profile/` for profiler output or report summaries.
- `runs/` or `outputs/` for generated artifacts.

The exact format is less important than consistency. A future reader should be able to reconstruct what changed, what was measured, and why a candidate was promoted.

## Promotion Rule

Promote a candidate only when it satisfies the task contract and has evidence that it improves or preserves the target metric. If a candidate is rejected, record the reason instead of silently discarding it.
