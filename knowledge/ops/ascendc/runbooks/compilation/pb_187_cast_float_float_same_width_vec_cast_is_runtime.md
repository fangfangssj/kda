---
type: Runbook
title: "`Cast<float,float>` (same-width VEC cast) is runtime NOT_SUPPORTED on dav-m200 — the vconv dtype-assert list admits `Tuple<float, float>` but the intrinsics impl reports `ASCENDC_REPORT_NOT_SUPPORT`; stage fp32 TQue↔TBuf via the bit-exact `Add(dst, src, zeros)` identity instead"
description: "dav-m200 的 typed VEC `Cast` 不支持 fp32→fp32 同型搬运（NOT_SUPPORT）——TBuf 间搬运要用 DataCopy/Adds 桥。"
tags: [task.debugging, topic.implementation.synchronization, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.vector, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-187", "Cast", "Tuple", "ASCENDC_REPORT_NOT_SUPPORT"]
---

# `Cast<float,float>` (same-width VEC cast) is runtime NOT_SUPPORTED on dav-m200 — the vconv dtype-assert list admits `Tuple<float, float>` but the intrinsics impl reports `ASCENDC_REPORT_NOT_SUPPORT`; stage fp32 TQue↔TBuf via the bit-exact `Add(dst, src, zeros)` identity instead

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a (API-surface gap); op_class=all (any kernel staging fp32 between TQue and persistent TBuf)`
`verified_on: soc=Ascend310P3; cann=9.0.0 — SDK-header fact, line-checked at merge: CANN 安装树内 impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h:356 (NOT_SUPPORT overload) vs :429 (assert-list entry); substitute shipped in 30_TimeDecayExponentialStabilization (2026-09-06, lane 3, run records DEVICE-RUN; kw-2 re-grade on op-def-faithful inputs, superseding the kw-1 randn-surrogate grade) — pass_a 50/50 with ours==CANN identically on every output (identical_to_cann 50/50), det 50/50 ×3`
`unverified_on: soc=Ascend950PR, Ascend910 (same-width Cast support there — likely present, unprobed)`

- **Severity**: MEDIUM — a runtime surprise, not a compile-time catch: the dtype assert passes, the kernel builds, the failure surfaces only when the cast executes.
- **Symptom**: using the typed VEC `Cast` to move fp32 data (e.g. staging an fp32 param/state tile from `TQue<VECIN>` into a persistent `TBuf<VECCALC>`, or copying a TBuf result into the output queue) fails with `ASCENDC_REPORT_NOT_SUPPORT(false, "Cast from type float to float")` on dav-m200 — even though `Tuple<float, float>` IS in `CheckCastDatatype`'s assert list.
- **Mechanism**: the per-SoC vconv assert list is strictly WIDER than the intrinsics impl's support — `float→int16_t`, `int16_t→float`, `float→int64_t` are likewise listed-but-NOT_SUPPORT (CANN 安装树内 impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h:359-374). Do not read the assert list as a support matrix. Same narrowed-surface family as PB-105 (DataCopyPad absent), PB-108 (Duplicate dtype whitelist), PB-113 (ShiftLeft NOT_SUPPORT), PB-118/P-P125 (vconv matrix).
- **Mitigation**: bit-exact `Add(dst, raw, zeros, count)` identity — the canonical UB→UB copy idiom (PB-9 forbids UB→UB `DataCopy`): `Duplicate<float>(zeros, 0.0f, tileLen);` once, then `Add(tdeBuf, raw, zeros, count)` for the TQue→TBuf load and `Add(outL, src, zeros, count)` for TBuf→TQue staging (`代码片段 30_TimeDecayExponentialStabilization/tde_wkv_vec.h.snippet.h（见文末）`).
- **Evidence**: 30_TimeDecayExponentialStabilization 310p kw-1 (2026-09-06, benchmark) — all fp32 param/state loads into the kernel's 11 persistent `TBuf<VECCALC>` buffers use the Add-identity; no same-width `Cast` anywhere in the shipped kernel.
- **Evidence (2nd witness — 1_BatchMatmul 310p, 2026-09-06, benchmark, lane 3; run records label DEVICE-RUN, [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) still declares `device.runtime: NOT_RUN` — same owner-unresolved provenance flag as sibling 310p rows)**: the SILENT flavor, worse than the error string above — `Cast<float,float>` on a staged fp32 mid-buffer compiled clean and executed as a NO-OP, leaving PLAUSIBLE-MAGNITUDE STALE-UB data in the B image (iter-1 smoke: fp32/bf16 paths maxdiff 27.4 on |out|~15, all 512 elements wrong; the fp16 path was bit-exact because the cross-dtype `vconv_f162f32` cast DOES work — only the same-width pair no-ops). Corroborates PB-113's mechanism note that on this SDK `ASCENDC_REPORT_NOT_SUPPORT` runs as a silent no-op rather than erroring, and supplies the provable mechanism behind CAND-DAV-M200-CAST-IDENTITY's observed-garbage symptom. Fix per the Mitigation idiom: zero vector `Duplicate`'d once in `Init`, staged chunk moved with `Add(img, tmp, zeroBuf)`; fp32 image re-anchored at `alignDown(base)` so all VEC dst offsets stay CHUNK-aligned. Post-fix fp32 maxdiff 0.0 (18/18 bit-exact). Anchor: `代码片段 1_BatchMatmul/batch_matmul_kernel.h.snippet.h（见文末）` + `1_BatchMatmul/failures_ledger.md` iter-1 row.
- **Other instances (predicted)**: any dav-m200 kernel with TBuf-resident fp32 state (recurrence scans, persistent scratch staging) where the natural "no-op" fp32→fp32 Cast reload is reached for; kernels ported from 220x/351x that used same-width Cast as a copy.
- **Cross-ref**: [`PB-118`](pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md) (fp32 VEC arithmetic WORKS on dav-m200 — only the same-width Cast is the gap; ❌ row added to its matrix), P-P125 (verify the per-SoC vconv matrix at SDK header level pre-build — and never trust the assert list as the matrix), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md)/[`PB-108`](pb_108_duplicate_uint8_t_is_rejected_on_dav_m200_sdk_dt.md)/[`PB-113`](pb_113_shiftleft_shiftright_are_not_supported_on_dav_m2.md) (m200 narrowed-surface family), [`PB-9`](pb_9_ub_to_ub_datacopy_silent_data_corruption.md)/[`PB-4`](pb_4_bf16_scalar_cast_failure.md) (why `DataCopy`/`static_cast` are not alternatives), (the blocked "same-type Cast produces garbage" defect CLAIM — PB-187 is the different, SDK-provable fact: the API is NOT_SUPPORTED outright, which is WHY the Adds-identity workaround was shipped in those ops). backend=ascendc.

## 内嵌代码片段

### 30_TimeDecayExponentialStabilization/tde_wkv_vec.h.snippet.h

```cpp
        raw = qP_.DeQue<T>();
        if constexpr (kFp32) {
            LocalTensor<float> zeros = zeroBuf_.Get<float>();
            Add(dst, raw, zeros, count);
        } else {
            Cast(dst, raw, RoundMode::CAST_NONE, count);
        }
        const int32_t count = static_cast<int32_t>(len);
        LocalTensor<float> zeros = zeroBuf_.Get<float>();
        LocalTensor<float> outL = qOut_.AllocTensor<float>();
        Add(outL, src, zeros, count);
        PipeBarrier<PIPE_V>();
        qOut_.EnQue(outL);
        outL = qOut_.DeQue<float>();
```

### 1_BatchMatmul/batch_matmul_kernel.h.snippet.h

```cpp
constexpr int64_t BCACHE_ELEMS = 32768; // fp32 B-image capacity (>= max k*p of
                                        // every benchmark case: 256*128)
constexpr int64_t AROW_MAX = 1024;      // host TORCH_CHECKs k <= 1024
constexpr int64_t STAGE_ELEMS = 4096;   // T_OUT staging; >= lcm(P, Qout) for all
                                        // P <= 256 (lcm <= P*Q <= 4096), so an
                                        // aligned flush boundary always exists
constexpr int64_t CVT_STAGE = 2048;     // bf16->fp32 convert staging (fp32 elems)

__aicore__ inline float Bf16BitsToFp32(uint16_t bits)
{
    FloatBitsCvt c;
    c.u = static_cast<uint32_t>(bits) << 16;
    return c.f;
}

    // float mid: Cast<float,float> is ASCENDC_REPORT_NOT_SUPPORT on dav-m200
    // (silent no-op — probe 2026-09-06: fp32 path produced stale-UB data
    // while the half path was bit-exact), so the staged VECIN chunk is moved
    // with Add(img, tmp, zeroBuf) instead. half mid: vconv_f162f32 Cast
    // (CAST_NONE), bit-exact-verified.
    __aicore__ inline void BuildBImage(int64_t bi)
    {
            // chain: fp32 is bit-exact vs the golden already, bf16 has 2x ULP
            // margin. Products of fp16 upcasts are exact in fp32, so Muls+Add
            // rounds exactly once per term (single-round add, FMA-equivalent).
            const int64_t blk = (kDim_ < 32) ? kDim_ : 8;
            const int64_t nC = (kDim_ + blk - 1) / blk;
            LocalTensor<float> cs = csBuf_.Get<float>();
            LocalTensor<float> prod = prodBuf_.Get<float>();
```

## 适用场景

- **工程形态**：dav-m200 的 typed VEC `Cast` 不支持 fp32→fp32 同型搬运（NOT_SUPPORT）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
