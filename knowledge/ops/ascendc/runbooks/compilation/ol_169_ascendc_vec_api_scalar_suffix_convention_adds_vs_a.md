---
type: Runbook
title: "AscendC VEC API scalar-suffix convention (`Adds` vs `Add`, `Muls` vs `Mul`, ...)"
description: "AscendC VEC API scalar-suffix convention (`Adds` vs `Add`, `Muls` vs `Mul`, ...)"
tags: [task.debugging, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-169-ascendc-vec-api-scalar-suffix-convention-adds-vs-a.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-169", "Adds", "Add", "Muls"]
---

# AscendC VEC API scalar-suffix convention (`Adds` vs `Add`, `Muls` vs `Mul`, ...)

> 原 OKF v1 卡号：**OL-169**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - writing any VEC arithmetic — Add, Mul, Sub, Div, Max, Min, etc.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: API_glossary
- **Loaded by**: All workers (Phase A KB load), aog-kernel-optimizer
- **Trigger**: writing any VEC arithmetic — `Add`, `Mul`, `Sub`, `Div`, `Max`, `Min`, etc.
- **Lesson**: AscendC VEC ops follow a strict naming convention:
  - **No `s` suffix** → vector ⊕ vector: `Add(dst, src1, src2, count)`. Both operands are LocalTensors of the same length.
  - **`s` suffix** → vector ⊕ scalar broadcast: `Adds(dst, src, scalar, count)`. The scalar is a single value (int/float, NOT a tensor) broadcast to all elements.
  - **Same convention for**: `Muls/Mul`, `Subs/Sub`, `Divs/Div`, `Mins/Min`, `Maxs/Max`, `Adds/Add`.
  - **Note `Cast` does NOT follow this** — `Cast(dst, src, mode, count)` is the only signature; the dtype-conversion is the operation.
- **Common mistake**: writing `Add(dst, src, 1, count)` when you mean `Adds(dst, src, 1, count)`. Compiler may accept but result is undefined or compile error depending on overload resolution.
- **Quick rule**: count of LocalTensor args = arity. `Adds` has 1 LocalTensor (the src), `Add` has 2.
- **Evidence**: CANN advance_step_spec.h uses `Adds(buf, buf, accepted_num_i, tokenEachReqs)` and `Add(buf, buf, slotMappingLocal, count)` in the same Compute() function; mixing them up at line 178 vs 250 would silently break index arithmetic.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-169（category=api_glossary，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
