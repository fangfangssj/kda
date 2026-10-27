---
type: Runbook
title: "upstream tiling-key variant sprawl is PERF-only, not correctness — for a mathematically-uniform op (pure copy/"
description: "上游 tiling-key 变体膨胀只是性能特化不是正确性要求——数学统一的算子（纯拷贝/赋值）一个通用 kernel 就够，别克隆变体树；纯搬运 kernel 位精确是构造性保证，非零 diff 永远是索引 bug 不是精度上限"
tags: [task.optimization, topic.data_movement.data_copy, topic.implementation.tiling, topic.compilation, topic.performance.analysis]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-410"]
---

# upstream tiling-key variant sprawl is PERF-only, not correctness — for a mathematically-uniform op (pure copy/

## 优化点

**OL-410** — **upstream tiling-key variant sprawl is PERF-only, not correctness — for a mathematically-uniform op (pure copy/assignment, no per-variant algorithm divergence) a single general-case kernel suffices for the correctness-verify path; author from A3 algorithm source, don't clone the variant tree.** Bit-exact-by-construction sub-principle: a copy-only/no-Cast/no-FP data-movement kernel is bit-exact by construction — a non-zero `max_abs_diff` is always an indexing/copy bug (cf. [`OL-167`](../compilation/ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md)/[`OL-407`](../compilation/ol_407_datacopy_count_is_an_element_count_on_every_arch.md)), never a precision ceiling to tolerance-loosen (Cast-bearing data movement → [`OL-44`](../compilation/ol_44_cast_bf16_f32_bf16_roundtrip_is_not_necessarily_bi.md)). Scope: does NOT apply when tiling keys encode genuinely different algorithms (reduction order, transpose layout). Anchor `代码片段 scatter_list/scatter_list_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]` (15 variants → 1 ~150-line kernel). Evidence: scatter_list 310p 2026-08-09, 12/12 bit-exact. Cross-ref OL-37, [`OL-44`](../compilation/ol_44_cast_bf16_f32_bf16_roundtrip_is_not_necessarily_bi.md), [`OL-167`](../compilation/ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md), [`OL-407`](../compilation/ol_407_datacopy_count_is_an_element_count_on_every_arch.md), [`OL-160`](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=data-movement / scatter-gather / index (generalizes to any mathematically-uniform op)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5), Ascend910_220x (A3) — this is an algorithmic-design rule (arch-independent), but evidenced only on dav-m200; the principle transfers, re-confirm the "general kernel suffices" claim per-op on other SoCs.`

**Principle**: upstream CANN op sources frequently carry a large family of tiling-key variants (e.g. `TransposeSmall/Large`, `NegLarge/LargeE`, `RSBSE/RLBSE`, dim-2-large, …) — often dozens of variants and hundreds of KB of code. These exist to pick a fast code path for a specific shape regime; they encode PERFORMANCE specializations, not the operation's mathematical definition. When the operation is mathematically uniform (the same transform applies regardless of shape — e.g. `reduce="update"` is pure assignment, a copy), the correctness-verify path does NOT need to replicate the variant tree: a single general-case kernel that distributes work across cores and moves/computes the data correctly produces bit-identical output to any of the variants, because the variants differ only in HOW they move/compute the same bytes, not in WHAT result they produce. Author the generated kernel from the A3 *algorithm source* (the math), not by cloning the variant tree.

**Why correctness is preserved — bit-exact-by-construction (folded sub-principle)**: a pure data-movement kernel — copy-only, no `Cast`, no floating-point op — is bit-exact with the reference BY CONSTRUCTION: it relocates bytes without transforming a bit. So for such ops a non-zero `max_abs_diff` is NEVER a "precision ceiling" to tolerance-loosen — it is always a real bug (wrong index, OOB, a missed copy, a truncated DataCopy per OL-167/OL-407). Diagnostic flip-side: when your data-movement kernel is not bit-exact, chase the indexing/copy bug, do NOT reach for a tolerance. (For data movement WITH a dtype `Cast`, see OL-44 — the cast roundtrip is the one place a data path is not automatically bit-exact.)

**Scope boundary — do NOT over-generalize (regression-risk)**: this applies ONLY when the operation is mathematically uniform across the tiling keys. It does NOT license collapsing variants when different tiling keys encode GENUINELY DIFFERENT algorithms (different reduction strategies, transpose-vs-straight layouts, approximate-vs-exact paths). Test the claim: if removing the variant tree changes output on ANY shape, the variants were correctness-relevant and this rule does not apply. The verify-path simplification is also orthogonal to the "generate from A3 source, do not wrap arch35" mandate — the simplified kernel is still GENERATED from the algorithm, not copied from upstream.

**Concrete anchor** (scatter_list 310p, `代码片段 scatter_list/scatter_list_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`): the 220x `scatter_list_base.h` source carries ~15 tiling-key variants (~200 KB). The dav-m200 generated kernel replaces ALL of them with one ~150-line class: distribute (b,n) pairs across the cores, copy var→UB→varOut, conditionally copy updates→UB→varOut. Correct for `reduce="update"` because assignment is shape-invariant. 12/12 cases `max_abs_diff=0.0`.

**Evidence**: scatter_list 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind verify path) — 12/12 bit-exact. `[first evidence]`

**Other instances (predicted)**: scatter_add (when the reduce semantics are uniform), index_copy/index_put pure-assignment paths, gather/gather_elements copy-only paths — any "many upstream tiling variants but the math is a copy/assignment" op. **Counter-examples (rule does NOT apply)**: reduction-class ops whose tiling keys choose different accumulation orders; transpose-class ops where layout-affecting variants change the result's memory order.

**Cross-ref**: OL-37 (data-movement VECIN→VECOUT bridge — the in-kernel pipe pattern such a kernel still needs), [`OL-44`](../compilation/ol_44_cast_bf16_f32_bf16_roundtrip_is_not_necessarily_bi.md) (cast roundtrip — the non-bit-exact data-movement case), [`OL-167`](../compilation/ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md)/[`OL-407`](../compilation/ol_407_datacopy_count_is_an_element_count_on_every_arch.md) (DataCopy count semantics the simplified kernel must still get right), [`OL-160`](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md) (canonical verify-path entry-point contract).

## 内嵌代码片段

### scatter_list/scatter_list_kernel.h.snippet.h

```cpp
#ifndef SCATTER_LIST_KERNEL_H_
#define SCATTER_LIST_KERNEL_H_

#include "kernel_operator.h"

namespace scatter_list_ops {

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPosition;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
        D_ = D;
        rowElems_ = S1 * D;     // elements per (b,n) row of var
        updateElems_ = S2 * D;  // elements per (b,n) row of updates

        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(varAddr));
        varOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(varOutAddr));
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updatesAddr));
        maskPtr_ = reinterpret_cast<__gm__ uint8_t*>(maskAddr);
        indicePtr_ = reinterpret_cast<__gm__ int32_t*>(indiceAddr);

        // UB buffer for data staging (max of row, update sizes, minimum 8 for overlap-tail)
        int64_t maxElems = rowElems_ > updateElems_ ? rowElems_ : updateElems_;
            int64_t n = idx % N_;
            ProcessPair(b, n);
        }
    }

private:
    // ── Pipeline sync helpers ──────────────────────────────────────────
    __aicore__ inline void SyncMte2ToMte3()
    {
        event_t e = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(e);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(e);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        event_t e = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(e);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(e);
    }

    __aicore__ inline void SyncSToMte2()
    {
        event_t e = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(e);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(e);
    }
```
