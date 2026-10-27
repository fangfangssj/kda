---
type: Optimization
title: "Double-Buffer Accumulation with Flush Threshold"
description: "Problem: When accumulating many grad rows for the same index after sorting, if the accumulation count is large (high fan-in), fp16/bf16 precision is gradually lost. Pattern: Maintain two UB accumulati"
tags: [task.optimization, topic.precision, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p40-double-buffer-accumulation-with-flush-threshold.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Double-Buffer Accumulation with Flush Threshold

> 原 OKF v1 卡号：**P-P40**。

## 优化点

**Problem**: When accumulating many grad rows for the same index after sorting, if the accumulation count is large (high fan-in), fp16/bf16 precision is gradually lost.

**Pattern**: Maintain two UB accumulation buffers (addRes[0] and addRes[1]) and switch them when the index changes:
```
switchId = false
for each (index, grad) in sorted_data:
    if index == lastIndex:
        Add(addRes[switchId], addRes[switchId], grad)  // UB vector add
        count++
        if count >= FLUSH_LIMIT:  // fp16: 10, fp32: can be larger
            CopyOut(addRes[switchId], GM_output)  // uses SetAtomicAdd
            Duplicate(addRes[switchId], 0)  // clear
            count = 0
    else:
        CopyOut(addRes[switchId], GM_output)  // flush old buffer
        switchId = !switchId
        Duplicate(addRes[switchId], 0)  // clear new buffer
        Copy(addRes[switchId], grad)  // begin new accumulation
        lastIndex = index
        count = 1
```

**Flush threshold**: **LIMIT_COUNT_NUM = 10** (fp16/bf16). After accumulating > 10 fp16 values, precision loss starts to appear.

**Reason for dual buffers**: while buffer A is flushing to GM (MTE3 pipe), buffer B can start accumulating the next index (VEC pipe), achieving MTE3/VEC overlap.

**Evidence**: CANN embedding_dense_grad_v2.h:255-332, LIMIT_COUNT_NUM=10 (line 26). E1 level.

**Stop condition**: When the index is random (unsorted), it degrades to flushing after every single accumulation. Requires `embDim * sizeof(CT) * 2 < UB available space`.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P40，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `scatter_add`；trigger: UB accumulation double-buffer swap after sort + fp16 precision protection；优先级: HIGH。

> 索引行原文：`| P-P40 | Double-buffer accumulation with flush | scatter_add | UB accumulation double-buffer swap after sort + fp16 precision protection | HIGH |`
