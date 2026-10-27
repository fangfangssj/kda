---
type: Runbook
title: "V220 `Abs<half>` / `Abs<bfloat16_t>` requires raw `__ubuf__*` TBuf pointer — use Cast→fp32 TBuf→Abs→Cast back for fp16/bf16 abs [V220]"
description: "V220 `Abs<half>` / `Abs<bfloat16_t>` requires raw `__ubuf__*` TBuf pointer — use Cast→fp32 TBuf→Abs→Cast back for fp16/bf16 abs [V220]"
tags: [task.debugging, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-129-v220-abs-half-abs-bfloat16-t-requires-raw-ubuf-tbu.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3]
aliases: ["OL-129", "Abs<half>", "Abs<bfloat16_t>"]
---

# V220 `Abs<half>` / `Abs<bfloat16_t>` requires raw `__ubuf__*` TBuf pointer — use Cast→fp32 TBuf→Abs→Cast back for fp16/bf16 abs [V220]

> 原 OKF v1 卡号：**OL-129**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - paradigm: ascendc

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

**Principle**: On V220 (dav_c220), the VEC `Abs<half>` and `Abs<bfloat16_t>` APIs do not accept TQue-dequeued `LocalTensor<half>` — they require a raw `__ubuf__ half*` (bare TBuf) as the first parameter.

**Fix**: For fp16/bf16 abs, promote to fp32 via a TBuf, compute Abs in-place on fp32, then cast back to native type with CAST_RINT. This is bit-exact because fp32 can represent all fp16/bf16 values exactly.

**Evidence**: op#4 Abs DS kw-1 (2026-05-02): 1 build iter to diagnose. Cast→fp32→Abs→Cast workaround produces bit-exact results vs CPU truth (50/50 PASS, MERE=0/MARE=0).

**Cross-ref**: OL-127 (CANN API surface gaps V220), OL-81 (CAST_RINT for fp16/bf16 output).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-129（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
