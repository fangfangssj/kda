---
type: Runbook
title: "TBuf workspace buffer used without `pipe_.InitBuffer()` — unallocated UB access → 507035 vector core exception on V220"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=all"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement, topic.memory]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-62-tbuf-workspace-buffer-used-without-pipe-initbuffer.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3]
aliases: ["EC-62"]
---

# TBuf workspace buffer used without `pipe_.InitBuffer()` — unallocated UB access → 507035 vector core exception on V220

> 原 OKF v1 卡号：**EC-62**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Kernel compiles and links, crashes 507035 (subErrType:4, ADDR_MISALIGN) on every launch.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=all`

- **Severity**: CRITICAL (kernel crashes at runtime with 507035, compiles cleanly)
- **Status**: CONFIRMED 2026-05-23 1_RotaryMul a3-ds kw-1
- **Symptom**: Kernel compiles and links, crashes 507035 (subErrType:4, ADDR_MISALIGN) on every launch.
- **Root cause**: Worker init'd TQue buffers but forgot TBuf workspace InitBuffer. TBuf::Get<T>(n) on unallocated UB → hardware fault.
- **Fix**: Add `pipe_.InitBuffer(tbuf_name, size_bytes)` for every TBuf member. All TBufs must be explicitly initialized.
- **Detection**: grep `TBuf<.*> \w+_;` in kernel.h. For each, verify `pipe_.InitBuffer(name, ...)` exists. Now mandated by kw_brief Phase C self-audit.
- **Evidence**: 1_RotaryMul a3-ds kw-1 (2026-05-23, V220): 5 TBuf workspace buffers had no InitBuffer. Adding them fixed fp16+fp32 on A3 NPU0.
- **Cross-ref**: EC-60 (blockDim=0), EC-61 (scalar-pipe acc), PB-22 (DataCopy 32B limit). All four are "compiles, crashes" V220 classes kw_brief must preempt.

<!-- 迁移自 porter kb/target/ascendc/（EC-62，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
