---
type: Runbook
title: "host-tiling-bound high-level APIs (SoftmaxGrad, any primitive whose Compute path needs a host TilingFunc) are"
description: "依赖 host TilingFunc 的高级 API（如 SoftmaxGrad）在无 op_host 注册的裸 launch 验证路径上不可用——要手工拆解成裸 VEC 链"
tags: [task.debugging, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-414"]
---

# host-tiling-bound high-level APIs (SoftmaxGrad, any primitive whose Compute path needs a host TilingFunc) are

**OL-414** — **host-tiling-bound high-level APIs (`SoftmaxGrad`, any primitive whose `Compute` path needs a host `*TilingFunc`) are unusable on the standalone pybind/`ACLRT_LAUNCH_KERNEL` verify path (no op_host registration) — decompose the math into a manual raw-VEC chain.** Softmax backward → Mul(yGrad,y)→two-pass `WholeReduceSum`→`GetValue`→Adds(-dot)→Mul, all fp32. Anchor `代码片段 scaled_masked_softmax_grad_v2/scaled_masked_softmax_grad_v2_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`. Evidence: scaled_masked_softmax_grad_v2 310p 2026-08-10, 9/9 fp32 PASS (device provenance UNVERIFIED, dav-m200 NOT_RUN; determinism not separately measured). Cross-ref [`OL-160`](ol_160_safety_net_file_name_coupling_is_an_architectural.md), [`OL-159`](ol_159_fa_class_forward_operators_with_softmax_attention.md), P-P101. backend=ascendc.

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=softmax-backward (and any grad/reduction op whose CANN high-level API requires a host-side TilingFunc)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3; cann=9.0.0 (scaled_masked_softmax_grad_v2 310p 2026-08-10)`
`unverified_on: soc=Ascend950PR (A5), Ascend910_220x (A3) — the principle (the API's host-tiling dependency vs the pybind path's lack of host registration) is structural and arch-general, but no A5/A3 pybind-path evidence exists yet; re-confirm the specific high-level API cannot be instantiated before relying on the decomposition.`

### Concrete anchor — softmax backward (the `SoftmaxGrad` case)

`SoftmaxGrad` requires `SoftMaxGradTilingFunc`. On the pybind path, replace it with the manual fp32 gradient chain — mathematically identical to `dL/dx_i = y_i * (dL/dy_i - Σ_j y_j·dL/dy_j)`:

```cpp
Mul(yGrad_y, yGrad, y, count);                              // y * dL/dy
WholeReduceSum<float>(dot, yGrad_y, MASK, rep, 1, 1, 8);    // pass 1: 1024 -> 16
WholeReduceSum<float>(dot, dot, rep, 1, 1, 1, 8);           // pass 2: 16 -> 1
PipeBarrier<PIPE_V>();
float s = dot.GetValue(0);                                  // Σ_j y_j·dL/dy_j
Adds(yGrad, yGrad, -s, count);                              // dL/dy - dot
Mul(out, yGrad_y, yGrad, count);                            // y * (dL/dy - dot)
// then optional Muls(scale) and mask Mul, both plain VEC
```

The reduction-to-scalar uses two `WholeReduceSum` passes + a scalar `GetValue(0)` (deliberately NOT `ReduceSum`/`GetAccVal`) — see `CAND-DAV-M200-REDUCESUM-RETURNS-ZERO` for why that primitive choice matters on arch 200x.

### Evidence

- scaled_masked_softmax_grad_v2 310p (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind path): the source has a `SoftmaxGrad`-API `norm_headdim` path AND a manual `large_headdim` path; only the manual Mul→WholeReduceSum→Adds→Mul chain above was shipped, because the `SoftmaxGrad` path's `SoftMaxGradTilingFunc` cannot be satisfied without op_host registration. Result: 9/9 fp32 PASS vs CPU fp64 truth (`scaled_masked_softmax_grad_v2/verification.json`, tier1_pass 9/9, all fp32). **Device provenance UNVERIFIED** — `back/310p-hw-spec-pages` declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN`; the 9/9 device PASS carries no confirmed device weight. The *principle* (`SoftmaxGrad`'s host-tiling dependency vs the pybind path's lack of it) is a CANN/CPU-side structural fact and holds regardless; the device PASS is the only device-dependent leg. (Determinism was not separately measured — the runner's determinism block is template-emitted, see the CPU-truth cohort audit; only the precision-vs-CPU-truth leg is cited here.)

## 内嵌代码片段

### scaled_masked_softmax_grad_v2/scaled_masked_softmax_grad_v2_kernel.h.snippet.h

```cpp

            // Step 2: dot = Σ(tmp) per row
            DoReduce(reduce, tmp);
            float dot = reduce.GetValue(0);

            // Step 3: xGrad = yGrad - dot
            Adds(xGradLocal, yGradLocal, -dot, paddedHeadDim_);
    __aicore__ inline void DoReduce(LocalTensor<float>& dst, LocalTensor<float>& src)
    {
        // Full pairwise tree reduction: 10 levels for 1024 elements.
        // This matches PyTorch's pairwise summation precision — each Add
        // round is a single fp32 addition with no hardware reduction tree
        // artifacts. Reduces normal-region precision mismatches vs the
        // 2-level WholeReduceSum approach.
        //
        // Step 1: Copy src to dst for in-place tree reduction
        DataCopy<float>(dst, src, paddedHeadDim_);
        PipeBarrier<PIPE_V>();

        // Step 2: Pairwise halving — full tree to 1 element
```

**孪生卡 ol-641 合并的独有内容**（ol-641-host-tiling-bound-high-level-apis-softmaxgrad-an.md 已删除）：

> Some AscendC high-level operators bind their `__aicore__` `Compute` body to a **host-side tiling function** (`SoftMaxGradTilingFunc`, and the `REGISTER_TILING_FOR_TILINGKEY`-style macros) that the op_host / ops-nn registration pipeline is expected to populate. The standalone pybind / `ACLRT_LAUNCH_KERNEL` verify path (the P140 path — `build_ascendc.py` compiles only `<op>/kernel/` and launches with a hand-written raw tiling struct) **runs no op_host registration**, so those host-tiling dependencies are unresolved and the high-level API cannot be instantiated there. The correct response is NOT to stage a fake op_host shim — it is to **decompose the op's math into raw VEC primitives that carry no host-tiling dependency**, preserving the algorithm exactly.

> This is the small, self-contained instance of the same structural reason FA-family arch35 kernels (which `#include` a multi-thousand-LOC shared-common library driven by host tiling) are infeasible on the pybind path (OL-159).

> WholeReduceSum<float>(dot, yGrad_y, MASK, rep, 1, 1, 8); // pass 1: 1024 -> 16

> WholeReduceSum<float>(dot, dot, rep, 1, 1, 1, 8); // pass 2: 16 -> 1

> - log-softmax backward, cross-entropy / NLL backward (softmax-then-reduce grads) whose CANN reference uses a `*Grad` high-level API with a `TilingFunc`.

> - Any grad op whose upstream `__aicore__` body is a single high-level call (`SoftmaxGrad`, a high-level `LayerNormGrad`, etc.) — port it as the decomposed arithmetic chain on the pybind path.

> - Forward softmax that used the `Softmax`/`SoftmaxFlashV2` high-level APIs — same host-tiling constraint; P-P101 documents the de-scalarized manual online-softmax for the forward case.

> **Cross-ref**: the P140 standalone pybind/`ACLRT_LAUNCH_KERNEL` verify path; [`OL-160`](ol_160_safety_net_file_name_coupling_is_an_architectural.md) (canonical entry-point contract the pybind shim satisfies); [`OL-159`](ol_159_fa_class_forward_operators_with_softmax_attention.md) (FA-family arch35 infeasibility on the pybind path — the same `REGISTER_TILING_FOR_TILINGKEY` / host-tiling dependency, at scale); P-P101 (manual online-softmax forward — the forward-side analogue). backend=ascendc.

## 补充证据（2026-10-10 合并自 `host_tiling_bound_apis_unusable_on_pybind_verify_path.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-414 — OL-414: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=softmax-backward (and any grad/reduction op whose CANN high-level API requires a host-side TilingFunc)。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (scaled_masked_softmax_grad_v2 310p 2026-08-10)。

**OL-414** — **host-tiling-bound high-level APIs (`SoftmaxGrad`, any primitive whose `Compute` path needs a host `*TilingFunc`) are unusable on the standalone pybind/`ACLRT_LAUNCH_KERNEL` verify path (no op_host registration) — decompose the math into a manual raw-VEC chain.** Softmax backward → Mul(yGrad,y)→two-pass `WholeReduceSum`→`GetValue`→Adds(-dot)→Mul, all fp32. Anchor 内部路径（不公开）workspace/scaled_masked_softmax_grad_v2/kernel/scaled_masked_softmax_grad_v2_kernel.h:127,180-186. Evidence: scaled_masked_softmax_grad_v2 310p 2026-08-10, 9/9 fp32 PASS (device provenance UNVERIFIED, dav-m200 NOT_RUN; determinism not separately measured). Cross-ref [OL-160](ol_160_safety_net_file_name_coupling_is_an_architectural.md), [OL-159](ol_159_fa_class_forward_operators_with_softmax_attention.md), [P-P101](../../optimizations/p_p101_de_scalarize_flash_online_softmax_hand_rolled_me.md). backend=ascendc.

- Former A5Ops ID: `OL-414`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=softmax-backward (and any grad/reduction op whose CANN high-level API requires a host-side TilingFunc)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3; cann=9.0.0 (scaled_masked_softmax_grad_v2 310p 2026-08-10)`
`unverified_on: soc=Ascend950PR (A5), Ascend910_220x (A3) — the principle (the API's host-tiling dependency vs the pybind path's lack of host registration) is structural and arch-general, but no A5/A3 pybind-path evidence exists yet; re-confirm the specific high-level API cannot be instantiated before relying on the decomposition.`

### Principle

Some AscendC high-level operators bind their `__aicore__` `Compute` body to a **host-side tiling function** (`SoftMaxGradTilingFunc`, and the `REGISTER_TILING_FOR_TILINGKEY`-style macros) that the op_host / ops-nn registration pipeline is expected to populate. The standalone pybind / `ACLRT_LAUNCH_KERNEL` verify path (the P140 path — `build_ascendc.py` compiles only `workspace/<op>/kernel/` and launches with a hand-written raw tiling struct) **runs no op_host registration**, so those host-tiling dependencies are unresolved and the high-level API cannot be instantiated there. The correct response is NOT to stage a fake op_host shim — it is to **decompose the op's math into raw VEC primitives that carry no host-tiling dependency**, preserving the algorithm exactly.

This is the small, self-contained instance of the same structural reason FA-family arch35 kernels (which `#include` a multi-thousand-LOC shared-common library driven by host tiling) are infeasible on the pybind path (OL-159).

### Concrete anchor — softmax backward (the `SoftmaxGrad` case)

`SoftmaxGrad` requires `SoftMaxGradTilingFunc`. On the pybind path, replace it with the manual fp32 gradient chain — mathematically identical to `dL/dx_i = y_i * (dL/dy_i - Σ_j y_j·dL/dy_j)`:

```cpp
Mul(yGrad_y, yGrad, y, count);                              // y * dL/dy
WholeReduceSum<float>(dot, yGrad_y, MASK, rep, 1, 1, 8);    // pass 1: 1024 -> 16
WholeReduceSum<float>(dot, dot, rep, 1, 1, 1, 8);           // pass 2: 16 -> 1
PipeBarrier<PIPE_V>();
float s = dot.GetValue(0);                                  // Σ_j y_j·dL/dy_j
Adds(yGrad, yGrad, -s, count);                              // dL/dy - dot
Mul(out, yGrad_y, yGrad, count);                            // y * (dL/dy - dot)
// then optional Muls(scale) and mask Mul, both plain VEC
```

The reduction-to-scalar uses two `WholeReduceSum` passes + a scalar `GetValue(0)` (deliberately NOT `ReduceSum`/`GetAccVal`) — see `CAND-DAV-M200-REDUCESUM-RETURNS-ZERO` for why that primitive choice matters on arch 200x.

### Evidence

- scaled_masked_softmax_grad_v2 310p (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind path): the source has a `SoftmaxGrad`-API `norm_headdim` path AND a manual `large_headdim` path; only the manual Mul→WholeReduceSum→Adds→Mul chain above was shipped, because the `SoftmaxGrad` path's `SoftMaxGradTilingFunc` cannot be satisfied without op_host registration. Result: 9/9 fp32 PASS vs CPU fp64 truth (`workspace/scaled_masked_softmax_grad_v2/verification.json`, tier1_pass 9/9, all fp32). **Device provenance UNVERIFIED** — `back/310p-hw-spec-pages` declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN`; the 9/9 device PASS carries no confirmed device weight. The *principle* (`SoftmaxGrad`'s host-tiling dependency vs the pybind path's lack of it) is a CANN/CPU-side structural fact and holds regardless; the device PASS is the only device-dependent leg. (Determinism was not separately measured — the runner's determinism block is template-emitted, see the CPU-truth cohort audit; only the precision-vs-CPU-truth leg is cited here.)

### Other instances (predicted)

- log-softmax backward, cross-entropy / NLL backward (softmax-then-reduce grads) whose CANN reference uses a `*Grad` high-level API with a `TilingFunc`.
- Any grad op whose upstream `__aicore__` body is a single high-level call (`SoftmaxGrad`, a high-level `LayerNormGrad`, etc.) — port it as the decomposed arithmetic chain on the pybind path.
- Forward softmax that used the `Softmax`/`SoftmaxFlashV2` high-level APIs — same host-tiling constraint; P-P101 documents the de-scalarized manual online-softmax for the forward case.

**Cross-ref**: the P140 standalone pybind/`ACLRT_LAUNCH_KERNEL` verify path; OL-160 (canonical entry-point contract the pybind shim satisfies); OL-159 (FA-family arch35 infeasibility on the pybind path — the same `REGISTER_TILING_FOR_TILINGKEY` / host-tiling dependency, at scale); P-P101 (manual online-softmax forward — the forward-side analogue). backend=ascendc.
