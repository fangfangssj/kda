---
type: Runbook
title: "Cube tiling host POD size mismatch — `TCubeTiling` is 50 int32 (200 B), not 51"
description: "Cube tiling host POD size mismatch — `TCubeTiling` is 50 int32 (200 B), not 51"
tags: [task.debugging, topic.programming_model.execution_unit.cube, topic.data_movement, topic.memory, topic.tiling, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-40-cube-tiling-host-pod-size-mismatch-tcubetiling-is.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-40", "TCubeTiling"]
---

# Cube tiling host POD size mismatch — `TCubeTiling` is 50 int32 (200 B), not 51

> 原 OKF v1 卡号：**EC-40**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Symptom (compile error)**:
  ```
  batchmatmul_tiling.h:80:37: error: static assertion failed:
      <name>TilingHost size must match TCubeTiling
  ```
- **Trigger**: a host-side POD that mirrors `TCubeTiling` (so pybind can `torch::empty + .copy_()` the tiling into NPU memory) is declared with the wrong field count — typically estimated as 51 from incomplete docs, but the actual count for CANN 9.0.0 is **50 int32 fields = 200 bytes**.
- **Fix**: Inspect `kernel_tiling.h` `struct TCubeTiling` field-by-field and count exactly. For CANN 9.0.0: 50 int32 fields. Mirror with `static_assert(sizeof(TilingHost) == 50 * sizeof(int32_t))`. **Recount whenever CANN version changes** — this is brittle to upstream additions.
- **Better long-term fix (Opt2 path, OL-91 step 3)**: Eliminate the host POD entirely by using on-stack `TCubeTiling tiling{}` filled in the kernel from scalar args + the non-`__gm__` `Init(const TCubeTiling*, TPipe*)` overload. No host H2D, no size-match concern, ~5–10 µs faster per call.
- **Evidence**: 1_BatchMatmul (2026-04-28) Phase C iter 2. Once op#1 documented the exact count, op#4 / op#5 / op#3 all built clean on first try.

<!-- 迁移自 porter kb/target/ascendc/（EC-40，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
