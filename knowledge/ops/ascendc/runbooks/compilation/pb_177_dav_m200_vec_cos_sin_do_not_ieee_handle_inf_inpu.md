---
type: Runbook
title: "dav-m200 VEC `Cos`/`Sin` do NOT IEEE-handle ±inf input — `Cos(+inf)=1.0` / `Sin(+inf)=+inf` instead of NaN (fp64 golden is NaN); scalar-detect the non-finite position and write canonical qNaN"
description: "一张处置说明卡：OL-81 的 CAST_NONE 在 200x 上是上游忠实性选择而非\"CAST_RINT 坏\"，作为 hedged note 挂在 OL-81 上未升格。"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-177", "Cos", "Sin"]
---

# dav-m200 VEC `Cos`/`Sin` do NOT IEEE-handle ±inf input — `Cos(+inf)=1.0` / `Sin(+inf)=+inf` instead of NaN (fp64 golden is NaN); scalar-detect the non-finite position and write canonical qNaN

`applies_to: soc=Ascend310P3 (dav-m200, __NPU_ARCH__=2002); cann=9.0.0; bisheng=n/a; op_class=trig/transcendental elementwise with possibly-non-finite angle (sinusoidal PE, RoPE, any VEC Cos/Sin of a possibly-inf/nan angle)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (fp32_sinusoidal_embedding_1d 310p, 2026-08-27 — on-device dump: Cos(+inf)=1.0 / Sin(+inf)=+inf vs fp64 golden 512 NaN)`
`unverified_on: soc=Ascend950PR (351x) — PB-19 documents the 950PR VEC trig limit on LARGE finite |x|≥1e10, but inf-INPUT handling there is not separately characterized; probe before transfer`
- **Severity**: MEDIUM — silent wrong-value (1.0 / +inf) instead of IEEE NaN at non-finite positions; only fires when the input carries inf/nan, which a timestep-position contract never does, but the 商发 2.1 grader's `inf_nan_golden_special` / `sp_inf` special cases exercise it explicitly.
- **Symptom**: `sp_inf` special cases FAIL with rule `inf_nan_golden_special`. On-device: `Cos(+inf) = 1.0`, `Sin(+inf) = +inf` (256 finite + 256 inf elements) while the fp64 golden is 512 NaN (IEEE `cos(inf)=NaN`, `sin(inf)=NaN`).
- **Root cause**: the DAV_2002 VEC transcendental argument-reduction path does not emit the IEEE NaN result for an inf/nan input — the `Cos`/`Sin` doc only guarantees the input range `[-65504, 65504]`, and outside it (inf included) the result is unspecified and lands on 1.0 / +inf rather than NaN.
- **Fix (shipped recipe)**: scalar-detect the non-finite position BEFORE the VEC math and write canonical qNaN straight to the output row, skipping `Cos`/`Sin` for that row:
  if (pos != pos || pos > 3.0e38f || pos < -3.0e38f) {                 // inf/nan detect
      LocalTensor<uint32_t> out32 = out.template ReinterpretCast<uint32_t>();
      Duplicate(out32, 0x7FC00000u, static_cast<int32_t>(dim));        // canonical qNaN
      Muls(angle, freq, pos, half);  Cos(out, angle, half);  Sin(out[half], angle, half);
  Anchor: `target/ascendc/examples/fp32_sinusoidal_embedding_1d__opsmath/snippets.md`.
- **Evidence**: fp32_sinusoidal_embedding_1d 310p (2026-08-27, Ascend310P3/dav-m200, CANN 9.0.0, ops-math vendor-op flow): `sp_inf` special cases FAILed with `inf_nan_golden_special`; on-device dump confirmed `Cos(+inf)=1.0`/`Sin(+inf)=+inf` vs fp64 golden 512 NaN. The scalar-detect + `Duplicate` qNaN branch fixed them (for the op's real timestep contract the branch is dead — it exists purely to satisfy IEEE special-value grading). `fp32_sinusoidal_embedding_1d__opsmath/failures_ledger.md` precision iter 1; `verification.json` cases_v21 228/228 PASS incl. inf-nan §5.3.3.
- **Other instances (predicted)**: any dav-m200 kernel applying VEC `Cos`/`Sin` to an angle that can be non-finite — RoPE with inf position, sinusoidal PE with inf/nan timestep, trig of a possibly-overflowed product. The lesson generalizes: check a VEC transcendental's documented input domain before assuming IEEE special-value handling of inf/nan.
- **Cross-ref**: [`PB-19`](pb_19_ascendc_vec_sin_cos_lack_payne_hanek_argument_redu.md) (the 950PR sibling — VEC trig fails on LARGE finite |x|≥1e10 via absent Payne-Hanek reduction; this PB is the dav-m200 inf-INPUT defect — same primitive family, distinct symptom + SoC), CAND-FP16-BF16-FP32-IEEE-CONVERT-COMPLETE (the same "IEEE special-value completeness is graded by 商发 2.1 §5.3.3" pressure, on the dtype-conversion path). backend=ascendc.
  - rms_norm_grad 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): a single untyped entry point `rms_norm_grad_kernel` (`rms_norm_grad_kernels.cpp:15-27`) takes a host-passed `dtype_flag` scalar and internally dispatches `if (dtype_flag==1) KernelRmsNormGrad<half> else KernelRmsNormGrad<float>` — the legacy untyped-entry + cast-inside workaround, supporting both fp16 and fp32 from one launch symbol. 9/9 T1 PASS. (Compile/launch-symbol fact — `dtype_flag` is a host int64 packed into the launch args — independent of the branch's `device.runtime: NOT_RUN` provenance flag; the 9/9 PASS itself carries no confirmed device weight.)
  - 32_AdaptiveAvgPool3d 310p (2026-08-21, benchmark, Ascend310P3/dav-m200/arch-200x, CANN 9.0.0) — **workaround-pattern TRANSFER, 3rd SoC family (not a dav-m200 reproduction of the bisheng cast failure)**: arch-200 has no native `bfloat16_t` at all, so bf16 input rows are loaded via direct `__gm__` scalar reads + the PB-4 payload-widen helper (`bf16_to_f32(u) { c.u32 = (uint32_t)u << 16; }`, `代码片段 32_AdaptiveAvgPool3d/adaptiveavgpool3d_kernel.h.snippet.h（见文末）`; header comment cites PB-4 at `:49-50`) — widening bf16 through bit-shift payload, never `static_cast`. 168/168 Pass B (bf16 cases included) with fp32 lane accumulators. Confirms the bit-manipulation widen/narrow family as the portable cross-arch bf16 scalar route (A5 SIMT → dav-m200 compute kernels). (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`.)
  - rms_norm_grad 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): 2nd norm-backward op after deep_norm_grad to use the `Adds(dst, src, 0.0f, n)` identity for UB→UB copy — here the fp16→fp32 widen of `dy`/`x`/`gamma` (`Adds(dyFp32, dy, 0.0f, N_align_)` etc., `rms_norm_grad_kernel.h:110-111,137,139,187`), deliberately avoiding UB→UB `DataCopy`. 9/9 T1 PASS, PB-106-safe launch path (`.npu()`-only H2D, no launch-stream `.to(kFloat)`/`.float()` D2D cast in `pass_a_runner.py:152-155`; the `.float()` calls are post-launch output diffing). (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`; the Adds-identity PATTERN is arch-general, the 9/9 carries no confirmed device weight.)
  - 26_AvgPool3d 310p (2026-08-17, Ascend310P3/dav-m200/arch-200x, CANN 9.0.0, benchmark — `torch.nn.functional.avg_pool3d`): adds a distinct compile-fact to the UB→UB copy surface — the `VEC Copy(dst, src, count)` UB→UB intrinsic is `#if (__NPU_ARCH__ == 3510)`-only (unavailable on dav-m200/200x), so the portable bit-exact UB→UB substitute is `Adds(dst, src, 0.0f, count)` (+0.0f is identity for finite fp32); used for the fp32 TQue fast path. Confirms Adds-identity as the universal portable UB→UB bridge when neither `DataCopy` (this entry's corruption) nor `Copy` (3510-gated) is usable. (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`; the `Copy`-3510-gate is a compile/header-level fact.)
  - 32_AdaptiveAvgPool3d 310p (2026-08-21, benchmark, Ascend310P3/dav-m200/arch-200x, CANN 9.0.0) — **workaround-pattern TRANSFER to dav-m200 (prophylactic, not a reproduced no-op)**: bf16 output rows are narrowed by scalar RNE (`f32_to_bf16`) into a staged `TQue<VECOUT>` buffer and emitted via chunked `DataCopy(UB→GM)` (`代码片段 32_AdaptiveAvgPool3d/adaptiveavgpool3d_kernel.h.snippet.h（见文末）`) — zero scalar GM stores by construction, per this entry's pure-AIV-class prescription. Re-validates the stage-then-DataCopy emit structure on arch-200x at 168/168 Pass B. (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`.)
- inplace_add_layer_norm 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, fused inplace-Add+LayerNorm, ACLRT_LAUNCH_KERNEL+pybind verify path) — **15th op, norm/reduction class — TWO more re-proposed primitive-defect flavors (ReduceSum→GetAccVal "returns 0.0" = 5th re-proposal; CAST_RINT "unsupported/asserts-false" = cast-rounding-mode flavor) + one non-novel API note.** The verify runner does `.npu()` H2D of all inputs with **no NPU-side `.to()`/`.float()` D2D cast** on the launch stream (the embedding_bag no-race flavor — PB-106 race not in play). (i) **`ReduceSum<float>`+`GetAccVal<float>()` "returns 0.0 regardless of input"** → shipped `ManualReduceSum` (scalar `GetValue` loop, `target/ascendc/examples/inplace_add_layer_norm/snippets.md`); the **5th re-proposal** of `CAND-DAV-M200-REDUCESUM-RETURNS-ZERO` (add_layer_norm=1st, add_layer_norm_quant_v2=2nd, add_rms_norm=3rd, deep_norm=4th). Evidence cross-check (skill 2.7) fails identically: the shipped kernel **never calls `ReduceSum`/`GetAccVal`**, so 9/9 T1 PASS validates the scalar-loop workaround, NOT the accumulator-defect claim; the symptom is confounded with a missing V→S fence (OL-195/OL-25) and no fence-correct isolation test was run. Candidate STAYS blocked. (ii) **Cast round-mode claim: "`CAST_RINT` triggers `ASCENDC_ASSERT(false)` at runtime; only `CAST_NONE`/`CAST_ODD` supported for fp32↔fp16 on dav-m200"** → shipped `CAST_NONE` for every cast (`kernel.h:82,94,122,134,148,181,216,234,248`); evidence cross-check fails — the kernel **never exercises `CAST_RINT`**, so PASS validates `CAST_NONE`, NOT the "CAST_RINT broken" claim. The runtime assert/silently-garbage half is device-unverified; the supported-modes half cites an **external** CANN header (CANN 安装树内 dav_m200/kernel_operator_vec_vconv_impl.h:217-260, unauditable from this repo). This regresses/muddies **OL-81** — whose calibrated dav-m200 cast notes (ge_glu_grad_v2, mse_loss_v2) already frame `CAST_NONE` on arch 200x as an **upstream-fidelity** choice, NOT a "CAST_RINT is broken" constraint. NOT promoted; appended as a hedged UNVERIFIED note to OL-81. (iii) **TQue API names** (`QuePosition::VECIN`+`EnQue`/`DeQue`, not `TPosition`/`EnQueTensor`; `TBuf` uses `TPosition::VECCALC`) — already documented in `LANGUAGE_REFERENCE.md`; non-novel, no entry. Positive findings (two-pass variance — `Adds(xF,xF,-meanVal)` then square; `DataCopy(outL,meanF,8)` UB→UB via TQue) are instances of existing arch-general rules (P-P51/§5 fp32 promotion; OL-25/OL-94 sync), not new dav-m200 defects. **Provenance flag (unresolved, same as the 6th–14th ops)**: `verification.json` claims 9/9 device PASS + `observed_deterministic: true`, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` (SoC identity `UNVERIFIED`); the two runtime/defect claims (i, ii) carry no confirmed device weight. **Do NOT cite "ReduceSum/GetAccVal broken" or "CAST_RINT unsupported/asserts-false on dav-m200" as constraints** until owner re-verification. See `kb_scan/regression_risk_20260809T093135Z.md`.
- scaled_masked_softmax_grad_v2 310p (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, softmax-backward, ACLRT_LAUNCH_KERNEL+pybind verify path) — **17th op — a POSITIVE mechanism corroboration (race observed during dev), NOT a no-race witness.** The runner stages the mask on-NPU via `torch.ne(mask, 0).to(torch.float32)` (`代码片段 scaled_masked_softmax_grad_v2/model_new_ascendc.py.snippet.py（见文末）`) — PB-106's exact dtype-conversion `.to(at::kFloat)` D2D trigger. The worker reported the canonical symptom AND fell into PB-106's documented misdiagnosis trap: the `knowledge_update` attributed the stale-zero reads to a *kernel* "Mask DataCopy race condition" with "root cause unclear" rather than to the host-side `.to()`/launch ordering. Mitigation chosen = a **data-dependent skip**: branch on `(mask == 0).any().item()` (`model_new_ascendc.py:55`) and run the `.to(float32)` cast ONLY when the mask actually contains zeros; all 9 edge cases happen to have all-nonzero masks, so the cast (and thus the race surface) is never exercised in the passing run. This is NOT one of PB-106's recommended fixes (delete the staging / host-side cast `.cpu().to(kFloat).to(device)` / `aclrtMemcpyAsync`); the `torch.npu.synchronize()` present (`model_new_ascendc.py:62`) sits AFTER the launch (for CPU read-back), not in the between-D2D-and-launch placement PB-106 requires. **The masked-input path (actual zero mask values, which triggers the cast) is UNVERIFIED** — the op's 9/9 PASS covers only the all-nonzero subset that avoids the race. Corroborates PB-106's mechanism (the `.to()` cast is the trigger) and its misdiagnosis-trap warning. **Provenance flag (unresolved, same as the 6th–16th ops)**: `verification.json` claims 9/9 device PASS, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN`; the device PASS carries no confirmed weight. **Do NOT cite "kernel DataCopy reads stale mask data on dav-m200" as a kernel defect** — it is the host `.to()`/launch ordering race (this entry), and the masked path is untested.
- foreach_non_finite_check_and_unscale 310p (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, foreach optimizer-AMP unscale+finite-check, ACLRT_LAUNCH_KERNEL+pybind verify path) — **18th op — the fp16↔fp32 `Cast` "broken in kernel" RE-PROPOSAL (the swi_glu_grad / mse_loss_grad_v2 flavor), AND the proposed workaround itself IS this entry's race surface.** The worker reported "ANY Cast half↔float inside the kernel produces all-zero output" (7-iter bisection) and shipped the mitigation: move ALL fp16↔fp32 conversion to the pybind layer (`xC.to(at::kFloat)` at `代码片段 foreach_non_finite_check_and_unscale/pybind11.core.snippet.cpp（见文末）`, result cast back at `:42`). Two independent grounds this is NOT a `Cast` defect: (1) **evidence cross-check (skill 2.7) fails** — the shipped kernel operates **fp32-only** (no `Cast` instruction anywhere; `Muls`/`ReduceMax`/`GetValue` all on `float`, `target/ascendc/examples/foreach_non_finite_check_and_unscale/snippets.md,70,86,88`), so 9/9 T1 PASS validates the pybind-cast + fp32-kernel workaround, NOT the "Cast broken" claim — identical to the add_rms_norm_quant / group_norm_silu same-type-Cast non-promotions (CAND-DAV-M200-CAST-IDENTITY); (2) **the proposed workaround IS this entry's race surface** — `xC.to(at::kFloat)` is an NPU-side D2D cast on the SAME current stream as the raw launch (`pybind11.cpp:34-36`) with NO `torch.npu.synchronize()` between cast and launch, i.e. the swi_glu / mat_mul_v3 `.to()` flavor, NOT the embedding_bag no-race flavor. The op passed deterministically (9/9, run-twice bit-exact) despite the unsynchronized cast — consistent with this entry's "ragged-shape-triggered" qualifier (this op's fp32 copies are 256B-aligned by construction, so the race window plausibly did not open), NOT evidence the race is absent. The bisection's "Cast always zeros" is therefore most parsimoniously this race (or an un-isolated V↔MTE3 fence gap), mis-attributed to a `Cast` primitive defect — the precise re-proposal pattern this Evidence list exists to head off. Truth source (`pass_a_runner.py:81-89` `_model_fp64`) is a clean fp64 `(x*inv_scale)`+isfinite golden — no reference mutation, so PASS is not masking via the `dynamic_quant` cheat class; it masks via the workaround. **Do NOT cite "Cast half↔float broken on dav-m200" as a constraint** — it is the next flavor of this entry's refuted/blocked dav-m200 primitive-defect misdiagnosis chain. **Provenance flag (unresolved, same as the 6th–17th ops)**: `verification.json` claims 9/9 device PASS, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN`; the device PASS carries no confirmed weight. See `foreach_non_finite_check_and_unscale/kb_scan/regression_risk_20260810T025355Z.md`.
- **Evidence**: gelu 310p (2026-08-12, port_a3_to_a5 TARGET=310p) — failures_ledger precision iter 2: max_abs_diff_large from period-64 mask reuse; root-caused by the on-device stage-dump bisect (period-64 simulation match=1.0000), fixed by deleting Compare/Select entirely → 29/29 Tier-1 PASS, det 29/29 bit-identical. Kernel anchor `代码片段 gelu/gelu_kernel.h.snippet.h（见文末）` (workspace scratch, gitignored).
- **Evidence**: gelu 310p (2026-08-12) — failures_ledger precision iter 1: max_abs_diff_large traced to the un-shifted `2^k` reconstruction; the int-Muls fix shipped in the same chain → 29/29 Tier-1 PASS, det 29/29 bit-identical. Kernel anchor `代码片段 gelu/gelu_kernel.h.snippet.h（见文末）` (workspace scratch, gitignored).
- **Evidence (2nd op, independent probe)**: reduce_sum 310p (2026-08-17, CANN 9.1.0) — dedicated multi-core probe reproduced the drop at blockDim 4/8 (~50% of disjoint 32B-separated scalar stores lost, run-varying, silent rc==0) while blockDim ≤ 2 landed reliably in the same probe; multi-core scalar LOADS 100% reliable (store-direction-only, matching above). Additional data points: `SetFlag/WaitFlag<HardEvent::S_MTE3>` + trailing dummy MTE3 copy does NOT fix it, and `DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE)` is an UNDEFINED SYMBOL on m200 (declared, not implemented — PB-105/PB-110 missing-API class). Shipped fix: all output via UB + vector DataCopy + host quantum-multiple over-alloc/narrow; 37/37 tier-1 PASS.

## 内嵌代码片段

### 32_AdaptiveAvgPool3d/adaptiveavgpool3d_kernel.h.snippet.h

```cpp
 *     elementwise into a k*W fp32 chunkAcc (each k-row chunk is ONE Add —
 *     column t always lands on slot j*W+t because chunk length is a multiple
 *     of W), fold the k slots, then reduce [0,W) to OW lanes by either
 *     (a) in-place log2(wlen) halving per segment — safe even when adjacent
 *         floor/ceil windows overlap by one element: halving writes stay in
 *         [ws, ws+wlen/2) and the shared element is only ever read, or
 *     (b) for wlen<=2-ish shapes, the V1 gather sequence against the folded
 *         row (cheaper than OW narrow halving passes);
 *   scalar widen/narrow (bf16_to_f32 / f32_to_bf16 — PB-4: no static_cast
 *   on bf16), scalar __gm__ reads for the window fill.
 *
 * dav-m200 authoring constraints honored: EC-60 (blockDim >= 1 tolerated:
 * empty slices when blockDim > totalUnits), EC-90/91 (AscendC qualification),
 * EC-92 (no direct integral->float static_cast in __aicore__ — widen through
 * int32_t), EC-93 (no Subs — none used), EC-95 (no floorf — integer div only),
 * EC-96 (count-form VEC), EC-94/EC-105 (int64 strides/divmod).
 */
            const int64_t wl = we - ws;                          // in {p, p+1}
            int64_t lg = 0;
            while ((1 << lg) < wl) { lg++; }
            segHalvingInstr += lg;
        }

        // ---- strategy dispatch (instruction-count cost model) ----
        // rowsMax/hlenMax over all (od, oh) units.
        int64_t rowsMax = 0, hlenMax = 0;
        for (int64_t od = 0; od < OD_; od++) {
            const int64_t d0 = (od * D_) / Sd_;
            const int64_t d1 = ((od + 1) * D_ + Sd_ - 1) / Sd_;
            for (int64_t oh = 0; oh < OH_; oh++) {
                const int64_t h0 = (oh * H_) / Sh_;
            gangS1_ = hlenU * W_;
            gangUnitSpan_ = (dlenU - 1) * H_ * W_ + gangS1_;
            // od-run crossing adds an extra H*W*(dlen-1) elems (the d-plane
            // gap): stride(j) = j*S1 + (j/OH)*odCross — a staircase. Batches
            // therefore start at od-run boundaries (u % OH == 0) so the
            // table is a per-case constant.
            gangOdCross_ = H_ * W_ * (dlenU - 1) * 4;   // bytes
```

### scaled_masked_softmax_grad_v2/model_new_ascendc.py.snippet.py

```python

import _fusion_attention_ext as _ext  # noqa: E402

# UB budget per AIV core: 256 KB total, ~192 KB usable.
# Kernel static allocations:
#   Q_f, K_f, V_f, attn_f, out_f all fp32 [S, D]  ->  5 * S * D * 4 bytes
```

### foreach_non_finite_check_and_unscale/pybind11.core.snippet.cpp

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

### gelu/gelu_kernel.h.snippet.h

```cpp
            uint32_t chunks = chunksLeft > TILE_CHUNKS ? TILE_CHUNKS : static_cast<uint32_t>(chunksLeft);
            uint32_t num = chunks * CHUNK_ELEMS;
            CopyIn(offset, num);
            Compute(num);
            CopyOut(offset, num);
            offset += num;
            chunksLeft -= chunks;
        // R(z): Horner in u = min(z, ZMAX)*U_SCALE - 1 (domain-clamped so the
        // poly never extrapolates: R(ZMAX) is bounded and f below kills the tail)
        Mins(tC, tA, ZMAX_F, num);
        Muls(tC, tC, U_SCALE, num);
        Adds(tC, tC, -1.0f, num);                        // u
        Duplicate(tD, R_COEF[18], num);
        for (int k = 17; k >= 0; --k) {
```

## 适用场景

- **工程形态**：一张处置说明卡。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
