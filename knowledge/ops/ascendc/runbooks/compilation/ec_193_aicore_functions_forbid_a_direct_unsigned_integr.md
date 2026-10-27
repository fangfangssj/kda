---
type: Runbook
title: "__aicore__ functions forbid a direct unsigned-integral→floating static_cast (\"cast between floating and unsign"
description: "`__aicore__` 函数禁止 unsigned→float 的直接 static_cast，要先经 int32  widening。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-193"]
---

# __aicore__ functions forbid a direct unsigned-integral→floating static_cast ("cast between floating and unsign

**EC-193** — `__aicore__` functions forbid a direct unsigned-integral→floating `static_cast` ("cast between floating and unsigned integer variable is not allowed in aicore function"); widen through `int32_t` first (`static_cast<float>(static_cast<int32_t>(u))`). COMPILE rejection of a plain C++ cast in device code — DISTINCT from [`PB-26`](pb_26_ascendc_tanh_fp32_primitive_bimodal_precision_floo.md) (the AscendC `Cast()` primitive's runtime dtype-pair silent-garbage table). Likely general aicore (the `__aicore__` attribute is on every arch), evidenced on dav-m200. Anchor: `代码片段 deep_norm_grad/deep_norm_grad_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`, 9/9 T1. Cross-ref [`PB-26`](pb_26_ascendc_tanh_fp32_primitive_bimodal_precision_floo.md), [`OL-18`](../precision/ol_18_every_code_change_must_be_followed_by_both_precisi.md), [`EC-191`](ec_191_duplicate_int8_t_has_no_dav_m200_overload.md)/91. backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a (aicore front-end rule); op_class=all (any __aicore__ function that converts a uint32_t/uint16_t to float)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (deep_norm_grad 310p iter-1 build, 2026-08-09)`
`unverified_on: soc=Ascend950PR (A5, arch 351x) — the rule is attributed to the aicore function attribute (present on every AscendC device function on every arch), so it is LIKELY general, not dav-m200-specific; re-confirm on a 351x kernel that needs a uint→float widening. Do not assume it is 220x-only.`

- **Category**: build / compile
- **Loaded by**: Generator/Worker when an `__aicore__` function converts an unsigned integer (loop bound, element count, dimension read as `uint32_t`) to `float` for reciprocal/scale math.
- **Symptom**: a line like `1.0f / static_cast<float>(D_)` where `D_` is `uint32_t` fails to compile with `error: cast between floating and unsigned integer variable is not allowed in aicore function`. The compiler accepts signed-integral→float and float→integral, but rejects unsigned-integral→float directly. (Distinct from PB-26, which is about the AscendC `Cast()` *primitive*'s dtype-pair table producing SILENT garbage for `uint8→float` at RUNTIME — this EC is a COMPILE rejection of a plain C++ `static_cast` in device code.)
- **Mechanism**: the bisheng/CCE aicore front-end disallows the unsigned-integral↔floating conversion in `__aicore__`-attributed functions. It is a codegen-rule restriction on the device-function attribute, not a header gap or a runtime primitive.
- **Fix**: cast through a signed `int32_t` intermediate: `static_cast<float>(static_cast<int32_t>(u))`. Safe when the unsigned value is in `INT32_MAX` range (element counts / dimensions always are; for true-uint32 magnitudes see the project int64 rule OL-18 — widen the intermediate instead of truncating).
- **Correct pattern** (anchor, `代码片段 deep_norm_grad/deep_norm_grad_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]` — `D_` is `uint32_t`, the inverse-count needs `float`):
  ```cpp
  // D_ is uint32_t (loop/dim bound). Direct static_cast<float>(D_) is rejected.
  invD_ = 1.0f / static_cast<float>(static_cast<int32_t>(D_));
  ```
- **Evidence**: deep_norm_grad 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 build rejected `static_cast<float>(uint32_t)`; the int32-intermediate fix is in the shipped kernel, which builds clean (object `auto_gen_deep_norm_grad_kernels.cpp.o` produced) and reports 9/9 T1 PASS. Recorded in `deep_norm_grad/failures_ledger.md` iter 1. Device provenance UNVERIFIED ([target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`) — but this is a COMPILE claim (build-host phenomenon), confirmed by the produced `.o`, NOT a device-runtime claim, so the provenance flag does not block it.
- **2nd-op confirmation + alternative mitigation (layer_norm_quant 310p, 2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5)**: hit the identical `cast between floating and unsigned integer variable is not allowed in aicore function` rejection for `1.0f / numCol` where `numCol` is `uint32_t`. Took a DIFFERENT mitigation than the int32-widen above — computed `aveNum = 1.0f / numCol` **on the host (`pybind11.cpp`)** and passed it into the kernel via the packed params buffer, avoiding the in-kernel unsigned→float cast entirely. Both mitigations are valid; the choice is a **decision rule** — int32-widen keeps the math in-kernel (preferred when the value is a simple dim/loop bound); host-side precompute is preferable when the scalar feeds a long in-kernel chain and you want it computed once in host fp64 precision. Confirms the rule on a 2nd dav-m200 op after deep_norm_grad. Anchor: `代码片段 layer_norm_quant/pybind11.core.snippet.cpp（见文末）` (host `aveNum` precompute → params buffer) → `代码片段 layer_norm_quant/layer_norm_quant_kernel.core.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]` (consumes `aveNum`, no in-kernel reciprocal). **Compile-fact** (build-host), confirmed by the produced `.o`; not a device-runtime claim.
- **3rd-op confirmation (rms_norm_grad 310p, 2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5)**: hit the identical unsigned→float rejection computing the RMSNorm mean-scaling `avg_factor` — `1.0f / static_cast<float>(static_cast<int32_t>(N_))` where `N_` is `uint32_t` (`代码片段 rms_norm_grad/rms_norm_grad_kernel.h.snippet.h（见文末）`). This is the same int32-widen mitigation as deep_norm_grad (the **decision-rule** choice above — value is a plain dim bound, so in-kernel int32-widen is preferred over layer_norm_quant's host-precompute). Confirms the rule on a 3rd dav-m200 op. Anchor: `代码片段 rms_norm_grad/rms_norm_grad_kernel.h.snippet.h（见文末）` (`N_ = static_cast<uint32_t>(N)` then the int32-intermediate reciprocal). **Compile-fact** (build-host), confirmed by the produced `.o`; not a device-runtime claim — holds regardless of the branch's `device.runtime: NOT_RUN` flag.
- **Cross-ref**: [`PB-26`](pb_26_ascendc_tanh_fp32_primitive_bimodal_precision_floo.md) (AscendC `Cast()` primitive dtype-pair table — the RUNTIME silent-garbage sibling; distinct mechanism), [`OL-18`](../precision/ol_18_every_code_change_must_be_followed_by_both_precisi.md) (int64/int32 value-range rule — when the int32 intermediate could overflow, widen instead), [`EC-191`](ec_191_duplicate_int8_t_has_no_dav_m200_overload.md)/[`EC-192`](ec_192_dav_m200_ascend310p3_use_the_full_using_namespac.md) (companion dav-m200 compile facts). backend=ascendc.

## 内嵌代码片段

### deep_norm_grad/deep_norm_grad_kernel.h.snippet.h

```cpp
 * adapted for __CCE_AICORE__==200 (310p): plain DataCopy (no DataCopyPad), fp32-only,
 * single-core (blockDim=1) to avoid the UNVERIFIED cross-core sync on dav-m200.
 *
 * Sync model (dav-m200 has no PIPE_MTE2_V):
 *   - GM→UB input sync via TQue<VECIN> EnQue/DeQue (handles MTE2→V).
 *   - UB→GM output sync via TQue<VECOUT> EnQue/DeQue (handles V→MTE3).
 *   - intra-VEC ordering via PipeBarrier<PIPE_V>.
 *   - UB→UB copy via Adds(dst, src, 0.0f, n) (PB-9: no DataCopy for LocalTensor).
 *
 * Math (canonical LN-grad):
        alpha_ = alpha;
        D_align_ = (D_ + DNG_BLK_ELEM_F32 - 1) / DNG_BLK_ELEM_F32 * DNG_BLK_ELEM_F32;
        // aicore forbids unsigned→float cast; go through signed int32.
        invD_ = 1.0f / static_cast<float>(static_cast<int32_t>(D_));

        dyGm_.SetGlobalBuffer((__gm__ float *)dy);
        xGm_.SetGlobalBuffer((__gm__ float *)x);
```

### layer_norm_quant/pybind11.core.snippet.cpp

```cpp
}

PYBIND11_MODULE(_fusion_attention_ext, m) {
    m.def("run_fusion_attention", &run_fusion_attention,
          "FusionAttention forward (row-tiled FlashAttention, Step 2)",
          pybind11::arg("query"),
          pybind11::arg("key"),
          pybind11::arg("value"),
          pybind11::arg("head_num"),
          pybind11::arg("input_layout"),
          pybind11::arg("scale") = 1.0,
          pybind11::arg("probe_workspace") = pybind11::none());
}
```

### layer_norm_quant/layer_norm_quant_kernel.core.snippet.h

```cpp
static constexpr uint32_t BLOCK_BYTES = 32;

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t RoundUp(uint32_t a, uint32_t b)
{
    return CeilDiv(a, b) * b;
}

class LayerNormQuantKernel {
public:
    __aicore__ inline LayerNormQuantKernel() {}

    __aicore__ inline void Init(
        __gm__ uint8_t* xGmRaw, __gm__ uint8_t* gammaGmRaw, __gm__ uint8_t* betaGmRaw,
        __gm__ uint8_t* scaleGmRaw, __gm__ uint8_t* offsetGmRaw,
        __gm__ uint8_t* zGmRaw, __gm__ uint8_t* scaleOutGmRaw,
        uint32_t numCol, float eps, float aveNum)
    {
        numCol_ = numCol;
        eps_ = eps;
        aveNum_ = aveNum;

        numPerBlockHalf_ = BLOCK_BYTES / sizeof(half);     // 16
        numPerBlockI8_   = BLOCK_BYTES / sizeof(int8_t);   // 32

        xGm_.SetGlobalBuffer((__gm__ half*)xGmRaw);
        gammaGm_.SetGlobalBuffer((__gm__ half*)gammaGmRaw);
        betaGm_.SetGlobalBuffer((__gm__ half*)betaGmRaw);
        zGm_.SetGlobalBuffer((__gm__ int8_t*)zGmRaw);
        scaleOutGm_.SetGlobalBuffer((__gm__ float*)scaleOutGmRaw);

        // UB allocation
        pipe.InitBuffer(xQue_,    BUFFER_NUM, TILE_SIZE * sizeof(half));
        pipe.InitBuffer(gbQue_,   BUFFER_NUM, TILE_SIZE * sizeof(half));
        pipe.InitBuffer(zQue_,    BUFFER_NUM, TILE_SIZE * sizeof(int8_t));
        pipe.InitBuffer(fp32Buf1_, TILE_SIZE * sizeof(float));
```

### rms_norm_grad/rms_norm_grad_kernel.h.snippet.h

```cpp
            int32_t tailAligned = Align8(tail);
            Add(src, src, src[body], tailAligned);
            PipeBarrier<PIPE_V>();
        }
        while (body > VEC_FP32_ELEMS) {
            body = body / 2;
            Add(src, src, src[body], body);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(VEC_FP32_ELEMS);
```

**孪生卡 ec-262 合并的独有内容**（ec-262-aicore-functions-forbid-a-direct-unsigned-integr.md 已删除）：

> - **Correct pattern** (anchor, `target/ascendc/examples/deep_norm_grad/snippets.md` — `D_` is `uint32_t`, the inverse-count needs `float`):

> // D_ is uint32_t (loop/dim bound). Direct static_cast<float>(D_) is rejected.

> - **3rd-op confirmation (rms_norm_grad 310p, 2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5)**: hit the identical unsigned→float rejection computing the RMSNorm mean-scaling `avg_factor` — `1.0f / static_cast<float>(static_cast<int32_t>(N_))` where `N_` is `uint32_t` (`target/ascendc/examples/rms_norm_grad/snippets.md`). This is the same int32-widen mitigation as deep_norm_grad (the **decision-rule** choice above — value is a plain dim bound, so in-kernel int32-widen is preferred over layer_norm_quant's host-precompute). Confirms the rule on a 3rd dav-m200 op. Anchor: `target/ascendc/examples/rms_norm_grad/snippets.md` (`N_ = static_cast<uint32_t>(N)` then the int32-intermediate reciprocal). **Compile-fact** (build-host), confirmed by the produced `.o`; not a device-runtime claim — holds regardless of the branch's `device.runtime: NOT_RUN` flag.
