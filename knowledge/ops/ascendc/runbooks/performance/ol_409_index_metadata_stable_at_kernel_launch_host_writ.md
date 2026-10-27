---
type: Runbook
title: "index/metadata stable at kernel launch (host-written or prior stream-ordered op) needs NO in-kernel MTE2_S syn"
description: "host 写入或前序流序算子准备好的 index/元数据，kernel 标量侧直接读 __gm__ 是安全的，不需要 EC-73 的 in-kernel MTE2_S 同步——那条 fence 只对 kernel 自己 DMA 进 UB 的数据必须"
tags: [task.optimization, topic.data_movement.data_copy, topic.implementation.synchronization, topic.compilation, topic.programming_model.execution_unit.scalar, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-409"]
---

# index/metadata stable at kernel launch (host-written or prior stream-ordered op) needs NO in-kernel MTE2_S syn

## 优化点

**OL-409** — **index/metadata stable at kernel launch (host-written or prior stream-ordered op) needs NO in-kernel MTE2_S sync — scalar `__gm__` deref is safe; [`EC-73`](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md)'s fence is required ONLY for data the kernel itself DMA-loads to UB.** Carve-out FROM [`EC-73`](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md) (complement, not repeal): no in-kernel MTE2 DMA of the buffer → no scalar-read race → no fence. Lets scatter/gather/index kernels read small host-staged index/mask tensors scalar-side without fencing every read or DMA-ing them to UB. Scope boundary: if the kernel DMAs the index to UB first, [`EC-73`](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md) STILL applies. Anchor `代码片段 scatter_list/scatter_list_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`. Evidence: scatter_list 310p 2026-08-09, 12/12 bit-exact. Cross-ref [`EC-73`](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md), [`OL-41`](../compilation/ol_41_int64_index_tensor_in_simt_direct_gm_read_works.md), [`OL-20`](ol_20_msprof_vec_ratio_atomicadd_share_simt_scalar_gm_re.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any kernel that reads index/offset/shape/mask metadata scalar-side)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5), Ascend910_220x (A3) — the rule is a logical consequence of Ascend stream-ordering + per-pipe DMA semantics and is arch-independent, but evidenced only on dav-m200 so far; the carve-out from EC-73 transfers, re-confirm on A5/A3 before relying on it there.`

**Principle**: EC-73's `HardEvent::MTE2_S` fence is conditioned on a SPECIFIC hazard — a scalar (S-pipe) read racing an MTE2 DMA that the SAME kernel issued to load data into UB. When the index/metadata the kernel reads scalar-side was written to GM *before* the kernel launched — by the host (pybind tensor assignment / `aclrtMemcpy`) or by a prior op on the same stream — it is already coherent at kernel entry (stream ordering guarantees the producer completed before this kernel starts). There is no in-kernel MTE2 DMA for the scalar read to race, so a bare `__gm__ T* idx; … = idx[i];` (or a `LocalTensor.GetValue` after NO preceding in-kernel DMA of that buffer) is safe WITHOUT any `SetFlag`/`WaitFlag`. This lets scatter/gather/index kernels read small host-staged index/mask tensors scalar-side without fencing every read or DMA-ing them to UB just to satisfy EC-73.

**Concrete anchor** (scatter_list 310p, `代码片段 scatter_list/scatter_list_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`) — the mask (uint8) and indice (int32) buffers are host-staged `__gm__` pointers read scalar-side with no DMA-to-UB and no fence:
```cpp
maskPtr_   = reinterpret_cast<__gm__ uint8_t*>(maskAddr);    // host-written before launch
indicePtr_ = reinterpret_cast<__gm__ int32_t*>(indiceAddr);  // host-written before launch
…
if (maskPtr_[b] == 0) { continue; }        // scalar read — safe: no in-kernel DMA of maskPtr_
int32_t startIdx = indicePtr_[b];           // scalar read — safe: same
// (EC-73's fence would only be needed if a DataCopy(idxLocal, idxGm, …) preceded these.)
```

**Scope boundary — do NOT over-generalize (regression-risk)**: this is the COMPLEMENT of EC-73, not a repeal. If the kernel itself issues `DataCopy`/`DataCopyPad` of the index buffer to UB and THEN reads it scalar-side, EC-73's `MTE2_S` Set/Wait is STILL mandatory — the host-written exemption applies only to data the kernel did NOT DMA. When unsure which case applies, the safe default is the EC-73 fence.

**Evidence**: scatter_list 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — mask/indice read scalar-side via `__gm__` deref, no `MTE2_S`, 12/12 cases bit-exact (`max_abs_diff=0.0`, `reduce="update"` pure data movement). `[first evidence]`

**Other instances (predicted)**: gather / scatter / index_select / index_put kernels whose index tensor is a kernel INPUT (host-staged) rather than kernel-DMA'd; dynamic-shape tiling readers where tiling metadata is passed as a host-staged `__gm__` struct and read field-by-field scalar-side; any kernel reading small per-block metadata cheaper to deref scalar-side than to DMA to UB.

**Cross-ref**: [`EC-73`](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md) (the fence this entry carves out from — DMA'd-to-UB data still needs it), [`OL-41`](../compilation/ol_41_int64_index_tensor_in_simt_direct_gm_read_works.md) (int64 `__gm__` read works in SIMT), [`OL-20`](ol_20_msprof_vec_ratio_atomicadd_share_simt_scalar_gm_re.md) (scalar GM reads travel the VEC pipe — a perf note, not a correctness one).

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
