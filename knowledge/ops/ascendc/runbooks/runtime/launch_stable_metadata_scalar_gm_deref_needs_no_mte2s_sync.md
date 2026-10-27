---
type: Runbook
title: "Index/metadata stable at kernel launch needs no in-kernel MTE2_S sync — scalar __gm__ deref is safe; EC-73's fence is required only for data the kernel itself DMA-loads to UB"
description: "Index/metadata stable at kernel launch needs no in-kernel MTE2_S sync — scalar __gm__ deref is safe; EC-73's fence is required only for data the kernel itself DMA-loads to UB"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["MTE2_S"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-409 — OL-409: A5Ops additive extension

> 关联：同一 OL-409 主题已由既有卡 [ol_409_index_metadata_stable_at_kernel_launch_host_writ.md](../performance/ol_409_index_metadata_stable_at_kernel_launch_host_writ.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any kernel that reads index/offset/shape/mask metadata scalar-side)。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0。

**OL-409** — **index/metadata stable at kernel launch (host-written or prior stream-ordered op) needs NO in-kernel MTE2_S sync — scalar `__gm__` deref is safe; EC-73's fence is required ONLY for data the kernel itself DMA-loads to UB.** Carve-out FROM EC-73 (complement, not repeal): no in-kernel MTE2 DMA of the buffer → no scalar-read race → no fence. Lets scatter/gather/index kernels read small host-staged index/mask tensors scalar-side without fencing every read or DMA-ing them to UB. Scope boundary: if the kernel DMAs the index to UB first, EC-73 STILL applies. Anchor 内部路径（不公开）workspace/scatter_list/kernel/scatter_list_kernel.h:57-58,113-119. Evidence: scatter_list 310p 2026-08-09, 12/12 bit-exact. Cross-ref [EC-73](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md), [OL-41](../compilation/ol_41_int64_index_tensor_in_simt_direct_gm_read_works.md), [OL-20](../performance/ol_20_msprof_vec_ratio_atomicadd_share_simt_scalar_gm_re.md). backend=ascendc.

- Former A5Ops ID: `OL-409`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any kernel that reads index/offset/shape/mask metadata scalar-side)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5), Ascend910_220x (A3) — the rule is a logical consequence of Ascend stream-ordering + per-pipe DMA semantics and is arch-independent, but evidenced only on dav-m200 so far; the carve-out from EC-73 transfers, re-confirm on A5/A3 before relying on it there.`

**Principle**: EC-73's `HardEvent::MTE2_S` fence is conditioned on a SPECIFIC hazard — a scalar (S-pipe) read racing an MTE2 DMA that the SAME kernel issued to load data into UB. When the index/metadata the kernel reads scalar-side was written to GM *before* the kernel launched — by the host (pybind tensor assignment / `aclrtMemcpy`) or by a prior op on the same stream — it is already coherent at kernel entry (stream ordering guarantees the producer completed before this kernel starts). There is no in-kernel MTE2 DMA for the scalar read to race, so a bare `__gm__ T* idx; … = idx[i];` (or a `LocalTensor.GetValue` after NO preceding in-kernel DMA of that buffer) is safe WITHOUT any `SetFlag`/`WaitFlag`. This lets scatter/gather/index kernels read small host-staged index/mask tensors scalar-side without fencing every read or DMA-ing them to UB just to satisfy EC-73.

**Concrete anchor** (scatter_list 310p, 内部路径（不公开）workspace/scatter_list/kernel/scatter_list_kernel.h:57-58,113-119) — the mask (uint8) and indice (int32) buffers are host-staged `__gm__` pointers read scalar-side with no DMA-to-UB and no fence:
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

**Cross-ref**: EC-73 (the fence this entry carves out from — DMA'd-to-UB data still needs it), OL-41 (int64 `__gm__` read works in SIMT), OL-20 (scalar GM reads travel the VEC pipe — a perf note, not a correctness one).
