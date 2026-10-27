---
type: Runbook
title: "K2 workspace must be pre-zeroed when K1 cores may skip their slot (fused scatter+reduce)"
description: "K2 workspace must be pre-zeroed when K1 cores may skip their slot (fused scatter+reduce)"
tags: [task.debugging, topic.implementation.pipeline, topic.data_movement, topic.performance.analysis, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-37-k2-workspace-must-be-pre-zeroed-when-k1-cores-may.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-37"]
---

# K2 workspace must be pre-zeroed when K1 cores may skip their slot (fused scatter+reduce)

> 原 OKF v1 卡号：**EC-37**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - In a multi-kernel pipeline, K1 writes per-core local accumulations into ws[nblk, H], and K2 sums along the nblk dimension. If K1 partitions by rows (or any axis

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: In a multi-kernel pipeline, K1 writes per-core local accumulations into `ws[nblk, H]`, and K2 sums along the nblk dimension. If K1 partitions by rows (or any axis with fewer logical units than nblk), some cores have `my_rows_=0` and `return` directly — their ws slices are never written, so K2 sum reads garbage.
- **Trigger conditions**:
  - Output shape `[nblk, H]` (per-core exclusive slot)
  - Logical work units `logical_work_units < nblk` (e.g., when BS < 56, rows do not fill all 56 cores)
  - K1 has `if (my_rows_ == 0) return;` fast-path
  - workspace comes from `torch::empty(...)` or similar uninitialised allocation
- **Root cause**: "per-core-exclusive workspace" does NOT imply "every core writes". Adjacent trap to OL-66 (torch::zeros not stream-ordered): OL-66 is about host-side zeros not being synchronised; this one is about device-side zeroing never happening at all.
- **Fix (3 options)**:
  1. **Preferred**: pybind calls `aclrtlaunch_memzero(ws_ptr, nblk*H*sizeof(T))` before K1, stream-ordered (automatically sequenced with K1/K2 on the same stream). Cost: one 56-core memzero launch (< 0.1 ms for 1 MB ws).
  2. K1 path guarantees every core writes (even if my_rows_=0, still Duplicate 0 into the core's own ws slot). Cost: every K1 variant must remember to handle this.
  3. Use `torch::zeros` instead of `torch::empty` (carries the OL-66 risk — host zeros ≠ stream-ordered, may overlap with K1).
- **Detection signal**: pass/fail flips across the N/nblk boundary — BS ≤ 32 PASS (some benchmark), BS ∈ (32, 2048) some cores idle → FAIL, BS = 2048 PASS again (all cores share work evenly). Failing case's output `[i][1]` (K2 product) has large-magnitude garbage (max_abs_diff ≥ 100), while `[i][0]` (K1 direct product) is completely correct.
- **Prevention (Phase B checklist)**: List every kernel's workspace; for each workspace, answer "do all cores in K1 write every slot? If not, who zeroes it before K2 reads?"
- **Evidence**: op#17 EmbeddingWithInitialLayernormBackward Phase D iter 1 (2026-04-23). `gnw_workspace[56, H]`, K1 partitioned by rows; when BS < 56, several cores skipped with my_rows_=0 → K2 sum read garbage → 30/57 FAIL. Fix: pybind added `K4_memzero(gnw_ws)` launch before K1 → 57/57 PASS.
- **Related**: OL-66 (torch::zeros not stream-ordered) — different failure mode same root theme.

<!-- 迁移自 porter kb/target/ascendc/（EC-37，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
