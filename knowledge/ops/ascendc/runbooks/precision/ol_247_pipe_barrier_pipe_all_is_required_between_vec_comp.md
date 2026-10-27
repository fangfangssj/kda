---
type: Runbook
title: "`pipe_barrier(PIPE_ALL)` is REQUIRED between VEC compute (Add/Muls) and MTE2 DataCopy when both touch the same UB buffer — silent precision corruption without it"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all (SIMD PipelineV, VEC+MTE2 data hazard)"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-247-pipe-barrier-pipe-all-is-required-between-vec-comp.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-247"]
---

# `pipe_barrier(PIPE_ALL)` is REQUIRED between VEC compute (Add/Muls) and MTE2 DataCopy when both touch the same UB buffer — silent precision corruption without it

> 原 OKF v1 卡号：**OL-247**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all (SIMD PipelineV, VEC+MTE2 data hazard)

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all (SIMD PipelineV, VEC+MTE2 data hazard)`
`verified_on: soc=Ascend950PR; cann=9.1.T500 (adaptive_avg_pool3d optimization, 2026-06-16 to 2026-06-17; 3 independent reproductions)`

### Principle

AscendC's PipelineV executes MTE2 (data movement) and VEC (compute) in parallel. When both pipelines touch the **same UB buffer** and there is no explicit barrier, the MTE2 `DataCopy` can **overtake** the VEC pipeline — reading stale data or writing over data VEC hasn't finished computing. This is a **RAW (Read-After-Write) hazard** at the pipeline level.

Two specific hazard patterns recurred 3 times in adaptive_avg_pool3d optimization:

1. **VEC `Add` → MTE2 `DataCopy` (output write)**: VEC accumulates partial sums into tile buffer; MTE2 loads that SAME buffer for GM writeout. Without barrier, MTE2 reads un-accumulated (or partially accumulated) data → silent garbage outputs.
2. **VEC `Muls` → MTE2 `DataCopy` (normalize-then-write)**: VEC normalizes (`Muls(1/winVol)`) the accumulated sum in UB; MTE2 immediately `DataCopy`s the normalized result to GM. Without barrier, MTE2 may read the raw (un-divided) sum → output = raw sums, not averages.

**Why it's silent**: The corruption is data-dependent (timing race between pipelines), not a trap/exception. Small shapes may pass by luck (MTE2 hasn't started yet when VEC finishes). Identity shapes (factor=1.0) coincidentally pass because normalization is a no-op.

**Why pipe_barrier(PIPE_ALL) and not something lighter**: On Ascend950PR, valid PipeBarrier pipes are PIPE_MTE2(4), PIPE_V(5), PIPE_MTE3(6). `PIPE_ALL` = `PIPE_MTE2|PIPE_V|PIPE_MTE3` — this correctly waits for BOTH the VEC compute AND the MTE2 load to drain before proceeding. A narrower pipe barrier (e.g., just PIPE_V) would not guarantee MTE2 drain.

### Decision rule

- After ANY VEC operation (Add, Muls, Sub, Abs, Max, Min, Relu, Cast, etc.) that writes to a UB buffer which a subsequent MTE2 `DataCopy` will READ → insert `pipe_barrier(PIPE_ALL)`.
- The pattern: `VEC_write_to_buf_X → pipe_barrier(PIPE_ALL) → MTE2_read_from_buf_X`.
- Double-buffering (ping-pong buffers) avoids the barrier for the STEADY state (VEC reads bufA while MTE2 writes bufB, or vice-versa), but the FIRST and LAST iterations still need the barrier (first: VEC reads bufA after MTE2 loaded it; last: MTE2 writes bufA that VEC just produced).
- This is distinct from EC-15 (PIPE_S not valid for PipeBarrier) — that's a compile-time error; this is a runtime correctness bug with no compiler diagnostic.

### Concrete anchor

```cpp
// BUG (silent corruption — MTE2 may read before VEC Muls finishes):
Muls(sumBuf, sumBuf, scalarInvVol, owCount * cAlign);  // VEC normalize
DataCopy(outGM[outOff], sumBuf, outSlice);               // MTE2 write → reads STALE sumBuf!

// FIX — split into two phases with barrier:
// Phase 1: batch all Muls
for (uint32_t ow = 0; ow < owCount; ++ow) {
    Muls(sumAll[ow * cAlign], sumAll[ow * cAlign], scalarInvVol, cAlign);
}
// Phase 2: barrier ensures VEC drained
pipe_barrier(PIPE_ALL);
// Phase 3: batch all DataCopy
for (uint32_t ow = 0; ow < owCount; ++ow) {
    DataCopy(outGM[outOff + ow * cAlign], sumAll[ow * cAlign], cAlign);
}
```

### Evidence

3 independent reproductions in adaptive_avg_pool3d optimization (2026-06-16 to 2026-06-17):
1. **Multi-column tiling** (09:20, 2026-06-17): Missing barrier after inner `Add` loop → VEC reads `tileL` while next iteration's MTE2 `DataCopy` overwrites `tileL`. max_diff=0.974 for small shapes, 0.019 for global. Fixed by adding `pipe_barrier(PIPE_ALL)` after Add loop.
2. **Double-buffering** (09:20, 2026-06-17): Missing barrier after `Add` in `isFirst` branch → VEC reads tileA, next iteration MTE2 DataCopy writes tileA (RAW hazard when bufState==0). Fixed same way.
3. **Loop reordering** (10:45, 2026-06-17): Missing barrier between `Muls` and `DataCopy` in normalization phase → identity cases coincidentally correct (factor=1.0 → Muls is a no-op), all non-identity cases produced raw sums (un-divided). Fixed by splitting normalization into Muls phase → pipe_barrier(PIPE_ALL) → DataCopy phase.

All 3 fixes restored bit-exact or near-bit-exact precision (max_diff ≤ 2.38e-07).

### Other instances (predicted)

Any SIMD kernel with a VEC→MTE2 pipeline where both touch the same UB buffer: reduction+writeback patterns (sum→store, max→store), activation+store (Relu→DataCopy), normalization+store (LayerNorm/RMSNorm-style Muls+DataCopy), elementwise compute→scatter write. The hazard is universal to PipelineV — not pooling-specific.

### Cross-references

- EC-15 (PIPE_S not valid for PipeBarrier — compile-time sibling, different root cause)
- OL-94 (TQue vs TBuf sync decision table — SetFlag/WaitFlag for cross-pipe signaling)
- OL-213 (AiCore-only PipelineV MUST use PipeBarrier not SyncAll)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-247（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
