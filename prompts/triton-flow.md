# Triton Kernel Flow Prompt

You are working in a task implementation workspace. Your job is to produce the best correct Triton kernel for the task described below. This template works for both NVIDIA (OpenAI Triton) and Ascend (triton_ascend) — the hardware profile decides the toolchain, validation and profiling conventions.

## Task Contract

- Task name: `<fill in>`
- Objective: `<fill in the user-facing goal>`
- Hardware profile: `hardware-profile.md`（启动 agent 前，将所选内置档案复制到任务工作区此路径；自定义档案也放在任务工作区内）
- Reference implementation (authoritative): `<e.g. torch eager / torch SDPA / torch_npu ACLNN>`
- Correctness requirements: `<allclose(rtol, atol) per dtype; shapes to cover, incl. non-power-of-2 and edge shapes>`
- Performance target: `<e.g. ≥1.5x vs baseline, measured with the profile's bench command>`
- Allowed implementation approaches: `<Triton only / 宿主框架、禁止事项>`
- Validation command: `<profile 约定的验证命令，任务工作区实现>`
- Evaluation command: `<profile 约定的基准命令（NVIDIA: triton.testing.do_bench；Ascend: ops-profiling compare）>`
- Promotion criteria: `<e.g. 全部形状精度通过 且 median 耗时优于当前最优候选>`

## Workflow

0. **先读 `hardware-profile.md`**，把工具链版本清单（torch/triton/GPU 或 NPU 型号）写进 `docs/draft.md` 的环境一节。若文件缺失，先要求将所选档案复制到任务工作区，再开始计划。
1. Read the repository structure, existing implementation, tests, and task documentation.
2. Identify the baseline behavior and the validation path.
3. Research only the references needed for this task（NVIDIA：KernelWiki + ncu-report-skill；Ascend：triton-op-designer / triton-op-coding / triton-op-verifier 等，按 profile 的技能映射）。
4. Write an implementation-plan draft to `docs/draft.md`. The draft MUST classify candidate directions into two groups:
   - **Algorithm variants**（算法变体）：different tiling schemes, split-K, persistent kernel, fused vs unfused…
   - **Config sweeps**（配置扫描）：BLOCK_SIZE / num_warps / num_stages / autotune keys.
5. Turn the draft into an executable plan before editing code.
6. Implement one candidate at a time. A candidate belongs to exactly one group and changes only that group's dimension（不得把算法改动与纯配置改动混在一个候选里而不标注）。
7. Run the validation command after each meaningful candidate — all contracted shapes and dtypes, every time.
8. Record evidence in the workspace:
   - `benchmark.csv`: one row per (candidate, config, shape, dtype), with baseline comparison.
   - `candidates.jsonl`: name, group (algorithm/config), parent, status, rejection reason if any.
   - `profile/`: profiler reports for promoted candidates AND for the best rejected candidate（没有归因证据的拒绝必须补分析再记录）。
9. Keep the final change scoped to the task contract.

## Triton-specific Rules

- **先算法变体后配置扫描**：算法变体未穷尽前，不对同一算法做大规模配置枚举；配置扫描必须报告搜索空间大小与逐配置耗时。
- autotune 的 `key` 必须覆盖影响最优配置的 shape 语义（分桶），防止跨 shape 污染。
- JIT 编译墙钟时间单独记录，不计入基准结果；constexpr 特化组合数要心里有数。
- 计时/采集与归因严格按硬件档案执行（NVIDIA：do_bench + ncu + TTGIR/PTX 检查；Ascend：ops-profiling + 上板为准，仿真只作预筛）。
- 精度判定用契约声明的 rtol/atol，bf16/fp16 累加确认 accumulator 为 fp32（NVIDIA 侧）。

## Plan Draft Requirements

The draft in `docs/draft.md` should include:

- 环境清单（profile 第 1 节要求的命令输出）。
- The current baseline and how it is validated.
- The main risks and unknowns.
- Candidate directions ranked by expected value and risk, **grouped as algorithm vs config**.
- The first concrete implementation steps.
- The exact validation and evaluation commands to run.
- The evidence required to promote, revise, or reject a candidate.

Do not start implementation until the draft exists.
