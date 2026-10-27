---
type: Runbook
title: "Branchless merge requires `CHUNK >= TOPK_CAP` invariant"
description: "Branchless merge requires `CHUNK >= TOPK_CAP` invariant"
tags: [task.debugging, topic.precision, topic.data_movement, topic.memory, topic.tiling, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-14-branchless-merge-requires-chunk-topk-cap-invariant.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["PB-14"]
---

# Branchless merge requires `CHUNK >= TOPK_CAP` invariant

> 原 OKF v1 卡号：**PB-14**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - If CHUNK < TOPK_CAP, the branchless merge reads past sortValOut[CHUNK..TOPK_CAP) which is OOB. Result: most cases FAIL precision with max_abs_diff=3.4e38, mean_

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Status**: CONFIRMED (2026-04-19)
- **Affected**: Any kernel using the R3b-style "branchless merge with tail-padded sentinel" pattern. See `output/npukernelbench/src/kernels/9_TopKTopP/kernel/topktopp_kernel.h` Phase 1 merge.
- **Symptom**: If `CHUNK < TOPK_CAP`, the branchless merge reads past `sortValOut[CHUNK..TOPK_CAP)` which is OOB. Result: most cases FAIL precision with `max_abs_diff=3.4e38, mean_abs_diff=inf`.
- **Root cause**: Branchless merge is designed under the invariant that both inputs (existing top buffer and new chunk's sorted output) have at least TOPK_CAP usable slots (with sentinel padding in unused slots). If the chunk's `sortValOut` is only `CHUNK` elements, reading `CHUNK..TOPK_CAP` goes into whatever's after the chunk buffer in UB — typically garbage values that compare greater than real data.
- **Fix**: Enforce `CHUNK >= TOPK_CAP` via `static_assert(CHUNK >= TOPK_CAP, "branchless merge requires CHUNK >= TOPK_CAP")` in tiling.h. If CHUNK needs to be smaller for UB reasons, use conditional-merge variant instead (slower but safe).
- **Evidence**: 9_TopKTopP V3.3 kind-2 Phase D iter 1 (2026-04-19) — CHUNK=1024 < TOPK_CAP=1088 caused 49/50 FAIL. Fixed CHUNK→2048 + static_assert → 50/50 PASS.
- **Related**: P-P59 Layer 1 canonical sketch. Add this invariant to any canonical sketch that uses branchless merge.

<!-- 迁移自 porter kb/target/ascendc/（PB-14，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
