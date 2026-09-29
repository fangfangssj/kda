# Ascend C Kernel Flow Prompt

You are working in a task implementation workspace on Ascend NPU. Your job is to produce the best correct Ascend C kernel for the task described below. Before starting the agent, copy the selected built-in profile into the task workspace as `hardware-profile.md` and use it with this prompt.

## Task Contract

- Task name: `<fill in>`
- Objective: `<fill in the user-facing goal>`
- Hardware profile: `hardware-profile.md`（启动 agent 前，将 `hardware/ascend-ascendc.md` 复制到任务工作区此路径；自定义档案也放在任务工作区内）
- Target SoC / NpuArch: `<e.g. 910B / Ascend910B1，用 npu-arch 技能确认>`
- Reference implementation (authoritative): `<e.g. torch_npu ACLNN 同名算子 / CPU golden>`
- Correctness requirements: `<按 ops-precision-standard 选择比对方式与阈值；tiling 分支全覆盖>`
- Performance target: `<e.g. ≥1.2x vs ACLNN 算子，ops-profiling compare 口径>`
- Allowed implementation approaches: `<Ascend C 直调工程，禁止事项>`
- Validation command: `<ST 用例 + 精度比对命令，任务工作区实现>`
- Evaluation command: `<ops-profiling 采集命令（标准 / --compare / --batch）>`
- Promotion criteria: `<精度全过 且 compare 报告显示加速>`

## Workflow

0. **先读 `hardware-profile.md`**。若文件缺失，先要求将选定档案复制到任务工作区。随后跑环境检查（`cann-env-setup`、`ascendc-env-check` 技能），并用 `npu-arch` 技能确认 SoC Version 与架构代际，全部记录进 `docs/draft.md`。
1. Read the repository structure, existing implementation, tests, and task documentation.
2. Identify the baseline behavior and the validation path.
3. Research only what this task needs（按 profile 技能映射：tiling 设计 → `ascendc-tiling-design`；API → `ascendc-api-best-practices`；性能手法 → `ascendc-performance-best-practices`）。
4. Write an implementation-plan draft to `docs/draft.md`. The draft MUST contain:
   - **Tiling 设计方案**（多核切分 / UB 切分 / Buffer 规划）及其分支覆盖清单——tiling 就是这个领域的"算法候选"。
   - tiling 参数（块大小、buffer 深度等）的"配置候选"清单。
   - 风险与未知项（如未确认的 API 支持度、跨代际差异）。
5. Turn the draft into an executable plan before editing code；工程从 `ascendc-direct-invoke-template` 脚手架起步。
6. Implement one candidate at a time（一个 tiling 方案 = 一个算法候选；其内部参数扫描 = 配置候选，分组推进、分别标注）。
7. Run the validation command after each meaningful candidate（ST 全分支 + 精度比对；精度问题走 `ascendc-precision-debug`，崩溃走 `ascendc-crash-debug`，运行时报错走 `ascendc-runtime-debug`，同步疑点走 `ascendc-sync-audit`）。
8. Record evidence in the workspace:
   - `benchmark.csv`: one row per (candidate, tiling, shape, dtype)，数据来自 `ops-profiling` 的 compare/quick 输出。
   - `candidates.jsonl`: name, group (algorithm/config), parent, status, rejection reason if any。
   - `profile/`: msprof 采集结果与瓶颈分析（msprof_perf_summary.py 输出）。
9. Keep the final change scoped to the task contract。

## Ascend C-specific Rules

- 性能结论必须来自上板采集（ops-profiling 对比模式），不接受推断或"感觉变快"。
- 候选必须声明目标 NpuArch；跨代际候选分开记录、分别验证。
- 禁止绕过 tiling 写死 shape（除非契约声明固定 shape 并记录代价）。
- 代码提交前按 `ascendc-code-review` 技能的检视规则过一遍。
- Device 侧日志与维测（黑匣子、错误码、core 分析）用 `msnpureport-toolkit` / `asys-toolkit` 技能。

## Plan Draft Requirements

The draft in `docs/draft.md` should include:

- 环境清单（CANN 版本、SoC、NpuArch、torch_npu 版本）。
- The current baseline and how it is validated.
- Tiling 设计方案与分支覆盖清单（多核切分边界、UB 余数分支）。
- Candidate directions ranked by expected value and risk, **grouped as tiling 方案 vs tiling 参数**。
- The exact validation and evaluation commands to run.
- The evidence required to promote, revise, or reject a candidate.

Do not start implementation until the draft exists.
