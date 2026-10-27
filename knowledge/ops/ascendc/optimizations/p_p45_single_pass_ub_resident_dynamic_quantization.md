---
type: Optimization
title: "Single-Pass UB-Resident Dynamic Quantization"
description: "Problem: Naive 2-pass implementation: Pass1 reads HBM to compute absmax -> Pass2 re-reads HBM to do scale+quantize. HBM bandwidth is used 2x. Pattern: When an entire row of D fits in UB, all operation"
tags: [task.optimization, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p45-single-pass-ub-resident-dynamic-quantization.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Single-Pass UB-Resident Dynamic Quantization

> 原 OKF v1 卡号：**P-P45**。

## 优化点

**Problem**: Naive 2-pass implementation: Pass1 reads HBM to compute absmax -> Pass2 re-reads HBM to do scale+quantize. HBM bandwidth is used 2x.

**Pattern**: When an entire row of D fits in UB, all operations complete in UB; HBM is only read/written once each:
```cpp
// Load row into UB (1 HBM read)
DataCopy(xLocal, xGm[rowOffset], D);
Cast(xFp32, xLocal, CAST_NONE, D);  // cast to fp32 if needed

// Step 1: Find absmax (ALL IN UB)
Abs(tmpLocal, xFp32, D);
ReduceMaxInplace(tmpLocal, D);  // -> tmpLocal[0] = max
pipe_barrier(PIPE_V);
float maxVal = tmpLocal.GetValue(0);

// Step 2: Compute scale (scalar unit)
float scaleTemp = 127.0f / maxVal;
float outScale = maxVal / 127.0f;  // save to output

// Step 3: Scale + quantize (ALL IN UB)
pipe_barrier(PIPE_S);  // S->V sync: scalar result to vector pipe
Muls(xFp32, xFp32, scaleTemp, D);

// Step 4: Cast chain to int8 (P-P46)
Cast(int32Local, xFp32, CAST_RINT, D);   // round to nearest
SetDeqScale(half(1.0f));
Cast(fp16Local, int32Local, CAST_NONE, D); // int32->fp16
Cast(int8Local, fp16Local, CAST_TRUNC, D); // fp16->int8

// CopyOut (1 HBM write)
DataCopy(yGm[rowOffset], int8Local, D);
```

**Key**: After data is loaded from HBM into UB, absmax, scale, and quantize all complete in UB. **No write-back to HBM followed by re-read is needed.**

**UB space requirement**: D * sizeof(float) * 2 (original data + temporary buffer) + intermediate-type buffer.
For D=8192 fp32: about 64KB * 2 = 128KB; 256KB UB is sufficient.

**Multi-row batching**: If UB space allows, multiRowNum rows can be processed simultaneously, using the Brcb instruction to broadcast per-row scale to full row width.

**Evidence**: CANN add_rms_norm_dynamic_quant_normal_kernel.h:329-353 (ScaleTensor), dynamic_quant_single_row.h:138-187. E1 level.

**Stop condition**: When D * sizeof(float) * 2 > available UB space, the D dimension must be tiled (see OL-60 Slice-D workspace pattern).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/reduction_quant.md（P-P45，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `reduction_quant`；trigger: absmax+scale+quantize all in UB, avoid 2-pass HBM；优先级: CRITICAL。

> 索引行原文：`| P-P45 | Single-pass UB-resident dynamic quantization | reduction_quant | absmax+scale+quantize all in UB, avoid 2-pass HBM | **CRITICAL** |`
