---
type: Optimization
title: "SetAtomicAdd + DataCopyPad — hardware atomic write in SIMD mode"
description: "Problem: in SIMD mode, scatter-add (e.g. grad_in[expert] += weight grad_out) needs an atomic write, but SIMT atomicAdd goes through a VEC-pipe CAS loop (slow), and SetValue in AIV mode is unreliable ("
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p25-setatomicadd-datacopypad-hardware-atomic-write-i.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# SetAtomicAdd + DataCopyPad — hardware atomic write in SIMD mode

> 原 OKF v1 卡号：**P-P25**。

## 优化点

**Problem**: in SIMD mode, scatter-add (e.g. `grad_in[expert] += weight * grad_out`) needs an atomic write, but SIMT `atomicAdd` goes through a VEC-pipe CAS loop (slow), and `SetValue` in AIV mode is unreliable (OL-19).

**Correct pattern**: `SetAtomicAdd<T>()` + `DataCopyPad` — hardware atomic on the MTE3 pipe

```cpp
// SIMD backward: write each token's grad_in contribution back to the expert slot
Muls(gradInLocal, gradOutLocal, expertWeight, hdim);
// EnQue + DeQue for pipeline sync
gradInOutQue_.EnQue(gradInLocal);
LocalTensor<float> gradInOut = gradInOutQue_.DeQue<float>();
// MTE3 atomic add: atomicity guaranteed by hardware, no VEC CAS
SetAtomicAdd<float>();
DataCopyPad(gradInGm_[expertIdx * hdim], gradInOut, copyParams);
SetAtomicNone();
gradInOutQue_.FreeTensor(gradInOut);
```

**Comparison**:
| Method | Pipe | Mechanism | Speed |
|--------|------|-----------|-------|
| SIMT `atomicAdd(ptr, val)` | VEC | CAS loop | Slow (serialized under contention) |
| SIMD `SetAtomicAdd` + `DataCopyPad` | **MTE3** | Hardware atomic DMA | **Fast** (bulk atomic add) |
| SIMD `SetValue` (GM) | Scalar | — | Unreliable (OL-19) |

**Key advantage**: DataCopyPad transfers the whole hdim vector in one atomic add, not per-element CAS.
**No sorting needed**: per-token SIMD + SetAtomicAdd writes back directly; no counting-sort preprocessing required.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P25，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `memory_access`；trigger: SIMD scatter-add to GM；优先级: CRITICAL。

> 索引行原文：`| P-P25 | SetAtomicAdd + DataCopyPad | memory_access | SIMD scatter-add to GM | **CRITICAL** |`
