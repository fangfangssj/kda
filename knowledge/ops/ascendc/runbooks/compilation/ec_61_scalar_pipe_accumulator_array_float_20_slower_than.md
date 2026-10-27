---
type: Runbook
title: "Scalar-pipe accumulator array (float[]) ~20× slower than VEC-pipe accumulator (TBuf<VECCALC>) for per-element reduction ops on V220"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=reduction (per-element accumulator)"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-61-scalar-pipe-accumulator-array-float-20-slower-than.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3]
aliases: ["EC-61"]
---

# Scalar-pipe accumulator array (float[]) ~20× slower than VEC-pipe accumulator (TBuf<VECCALC>) for per-element reduction ops on V220

> 原 OKF v1 卡号：**EC-61**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Kernel produces correct output but perf ratio < 0.1×. aiv_vec_ratio near 0, high scalar pipe utilization.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=reduction (per-element accumulator)`

- **Severity**: PERFORMANCE (kernel works correctly but 20-50× slower than achievable)
- **Status**: CONFIRMED 2026-05-22 27_MaxPool3d a3-ds (0.047× perf ratio, scalar pipe bottleneck)
- **Symptom**: Kernel produces correct output but perf ratio < 0.1×. `aiv_vec_ratio` near 0, high scalar pipe utilization.
- **Root cause**: `float acc[N]` (S-pipe scalar array). Every `acc[i] = val` is a scalar store (1 element/cycle) vs VEC pipe (8+ elements/cycle for fp32).
- **Fix**: Replace `float acc_[TILE_W]` with `TBuf<TPosition::VECCALC> accBuf_`. Use VEC `Duplicate` for init, `Max` for accumulation, `Cast` + direct `DataCopy` for output.
- **Evidence**: 27_MaxPool3d a3-ds kw-1 (2026-05-22, Ascend910_9382 V220).
- **Cross-ref**: P-P47 (VEC halving for reductions); OL-161 (V220 SIMD UB element duplication).

<!-- 迁移自 porter kb/target/ascendc/（EC-61，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
