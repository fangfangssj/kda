---
type: Runbook
title: "SIMT helper APIs — AtomicAdd / UintDiv / `__local_mem__` UB access [V351, simt-l3]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=scatter-with-conflict,gather-with-div,thread-shared-state"
tags: [task.debugging, topic.implementation.pipeline, topic.programming_model.execution_model.simt, topic.implementation.atomic, topic.data_movement, topic.memory]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-151-simt-helper-apis-atomicadd-uintdiv-local-mem-ub-ac.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-151", "__local_mem__"]
---

# SIMT helper APIs — AtomicAdd / UintDiv / `__local_mem__` UB access [V351, simt-l3]

> 原 OKF v1 卡号：**OL-151**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=scatter-with-conflict,gather-with-div,thread-shared-state

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=scatter-with-conflict,gather-with-div,thread-shared-state`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 references/l3-simt-optimization-guide.md §125-172`

**Rule**: SIMT kernels have NO hardware divider (per-thread) and NO scalar-write race protection — so the SIMT runtime provides 3 helpers for the common patterns: atomic accumulation, fast integer division by predicomputed multiplier, and shared UB read/write via `__local_mem__` parameter.

### Helper 1 — `Simt::AtomicAdd` for multi-thread write-to-same-address

When multiple SIMT threads write to the same GM address (e.g. `IndexAdd`-style scatter accumulation), naive write races. Use `Simt::AtomicAdd` instead:

```cpp
// FP32 — direct atomic add
Simt::AtomicAdd(var + varOffset, updates[i] * alphaValue);

// FP16 / BF16 — must compute in float, cast on store side
Simt::AtomicAdd(
    var + varOffset,
    static_cast<half>(static_cast<float>(updates[i]) * static_cast<float>(alphaValue))
);

// INT8 / UINT8 / INT16 — atomic accumulation NOT supported on these widths
// Workaround: accumulate to int32/fp32 workspace, then cast back on a post-pass
Simt::AtomicAdd(varWorkspaceGm + varOffset, static_cast<int32_t>(updates[i]));
// Then separate kernel pass:  workspace_int32 → output_int8 with clamp
```

**Hard rules**:
- Supported dtypes: `fp32`, `fp16`, `bf16` (with fp32 intermediate compute)
- NOT supported: `int8`, `uint8`, `int16` — use int32/fp32 workspace + post-pass
- Race-free guarantee is per-address — concurrent writes to DIFFERENT addresses don't need AtomicAdd

### Helper 2 — `Simt::UintDiv` for fast in-thread integer division

SIMT threads have no hardware divider; integer division via standard `/` is dramatically slow. Use precomputed magic-number division:

```cpp
// In Tiling (host code) — precompute magic constants for the divisor:
// m0 = ceil(2^32 / divisor)
// shift0 = 32 + log2(divisor)
// Pass m0 and shift0 in tiling struct to the kernel.

// In SIMT kernel:
INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);
// Mathematically equivalent to:   yIndex / innerSize
// But uses (multiply-high + shift) — pipelined on SIMT issue path
```

**Use when**: any `yIndex / constant_innerSize` pattern in SIMT — e.g. 2D Gather where row = idx / cols. Divisor must be known at tiling time (so magic constants can be precomputed).

**Don't use when**: divisor varies per thread / per element — magic-number division requires a fixed divisor.

### Helper 3 — `__local_mem__` parameter for SIMT→UB access

SIMT threads can read/write UB via a `__local_mem__` typed pointer passed as kernel-function argument. This enables thread-level shared state without going through GM.

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd,
    __gm__ int32_t *sortedExpertIdGmAddr,
    __local_mem__ int32_t *expertFirstIndexLocalAddr)   // ← UB pointer
{
    for (auto i = Simt::GetThreadIdx(); i < elementNum; i += Simt::GetThreadNum()) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) break;
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;   // write UB
        }
    }
}
```

**Use cases**:
- Per-expert / per-bucket first-index tables (MoeInitRoutingV3)
- Small histogram-like reductions where threads update different bins
- Cross-thread message passing within a small SIMT batch

**Hard rules**:
- `__local_mem__` pointer is passed FROM the launcher (host code in `_apt.cpp` allocates a `LocalTensor<T>` then passes `.GetPhyAddr()` to `Simt::VF_CALL`)
- Threads writing to the SAME `__local_mem__` index race — application must ensure thread index → unique UB index OR use atomic primitives
- Total `__local_mem__` allocation comes out of UB — Tiling must account for it ALONGSIDE the 40KB SIMT DCache reservation (PB-32)

**Evidence**:
- PR 103 l3-guide §125-172 codifies all three helpers
- `MoeInplaceIndexAdd` uses Simt::AtomicAdd
- `GatherV2SimtTwoDim` uses Simt::UintDiv
- `MoeInitRoutingV3 / ComputeExpertFirstIndexSimt` uses __local_mem__

**Other instances (predicted)**:
- `index_put_with_sort` (cohort 1 BLOCKED) — `Simt::AtomicAdd` candidate for fp32/fp16 cases; workspace+post-pass for int variants
- `flash_attention_score` (cohort 2 ACTIONABLE) — `__local_mem__` per-row state candidate
- `moe_init_routing_v3` (cohort 2) — known user

**Cross-reference**:
- OL-150 (SIMT core programming model)
- PB-32 (SIMT DCache 40KB — UB reservation MUST include `__local_mem__` allocations)
- OL-143 (L1/L2/L3 classifier — Simt::AtomicAdd indicates L3 candidate with race-prone Scatter)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-151（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
