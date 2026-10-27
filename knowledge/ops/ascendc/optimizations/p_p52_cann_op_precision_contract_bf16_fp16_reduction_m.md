---
type: Optimization
title: "CANN op precision contract — bf16/fp16 reduction must use fp32 promotion"
description: "Trigger: kernel needs to reduce a bf16/fp16 tensor (sum, mean, max, min, prod, etc.). Pattern: every CANN reduction operator's DAG follows a fixed 5-step structure: CopyIn<T> → Cast<fp32, T> → ReduceX"
tags: [task.optimization, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p52-cann-op-precision-contract-bf16-fp16-reduction-m.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# CANN op precision contract — bf16/fp16 reduction must use fp32 promotion

> 原 OKF v1 卡号：**P-P52**。

## 优化点

**Trigger**: kernel needs to reduce a bf16/fp16 tensor (sum, mean, max, min, prod, etc.).

**Pattern**: every CANN reduction operator's DAG follows a **fixed 5-step structure**:
```
CopyIn<T>  →  Cast<fp32, T>  →  ReduceXxxOp<fp32>  →  Cast<T, fp32>  →  CopyOut<T>
```
i.e., **bf16/fp16 input must first be cast to fp32, reduction is performed in fp32, and the result is cast back to the original dtype**.

**Source evidence** (`reduce_sum_dag.h:28-38`):
```cpp
template <typename T, typename PromteT>  // T = input dtype, PromteT = fp32
struct ReduceSumDag {
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;    // T → fp32
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, Cast0>;    // reduce in fp32
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;     // fp32 → T
};
```

**In our kernel writing style**:
```cpp
// bf16/fp16 input
Cast(fp32Buf, srcBf16, RoundMode::CAST_NONE, count);  // T → fp32 (lossless)
PipeBarrier<PIPE_V>();
ReduceSumP47(fp32Buf, count);  // or AscendC::ReduceSum<float, Pattern::Reduce::AR, true>
// result in fp32Buf[0]
// if you need to put the scalar result back into T, then Cast(..., CAST_RINT)
```

**Key**: Cast must use `CAST_RINT` (IEEE RNE), not `CAST_ROUND` (round half up) — see OL-81.

**Applicability**: all ops with bf16/fp16 input + internal reduction (LayerNorm, GroupNorm, AdaIN, Softmax, Sum, Mean, Var, etc., including backward)

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P52，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `precision`；trigger: all CANN reduce ops follow a fixed DAG: CopyIn→Cast<fp32,T>→ReduceOp<fp32>→Cast<T,fp32>→CopyOut; use CAST_RINT. [CPU-truth audit 2026-04-29: ✅ VALIDATED — co-occurs with CANN-pass op 17_EmbeddingWithInitialLayernormBackward. Note: aligns with CPU's fp32 promotion convention for reductions, so this IS CPU-truth-aligned. Do not downgrade.]；优先级: CRITICAL。

> 索引行原文：`| P-P52 | CANN reduction precision contract — bf16/fp16 must use fp32 promotion | precision | all CANN reduce ops follow a fixed DAG: CopyIn→Cast<fp32,T>→ReduceOp<fp32>→Cast<T,fp32>→CopyOut; use CAST_RINT. **[CPU-truth audit 2026-04-29: ✅ VALIDATED — co-occurs with CANN-pass op 17_EmbeddingWithInitialLayernormBackward. Note: aligns with CPU's fp32 promotion convention for reductions, so this IS CPU-truth-aligned. Do not downgrade.]** | **CRITICAL** |`
