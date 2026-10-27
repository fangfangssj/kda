---
type: Runbook
title: "dav-m200 vconv accepts only a per-direction cast-mode whitelist"
description: "dav-m200 vconv 的 cast 模式按方向白名单：h→f 仅 CAST_NONE，f→h 仅 CAST_NONE/CAST_ODD——OL-81 的 CAST_RINT 默认值在此不适用"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-120"]
---

# dav-m200 vconv accepts only a per-direction cast-mode whitelist

**PB-120** — **dav-m200 vconv accepts only a per-direction cast-mode whitelist**: h→f `CAST_NONE` only; f→h `CAST_NONE`/`CAST_ODD`; `CAST_RINT` compiles but hits `ASCENDC_ASSERT(false)` → the cast NEVER EXECUTES → stale/uninit fp16 outputs (widened paths clean, no error raised — grep `RoundMode::CAST_RINT` to detect). f32→f16 `CAST_NONE` is RNE at normal magnitudes (0 diffs/1M vs CPU RNE) and SATURATES NaN/overflow→±65504 (== torch_npu `.to(fp16)`; canonical-grader-forgiven). PROMOTES [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md)'s 2026-08-09 hedge — its stated condition (device run exercising CAST_RINT on dav-m200) met by 17_AdamW 310p 2026-08-15: 18/18 OVERALL_T1, 13/18 bit-exact, det 21/21, perf 0.7408x. CAST_RINT default is VOID on arch-200x half↔float. Cross-ref [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md), [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md) (carries no dav-m200 caveat yet — flagged for owner), [`PB-107`](pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md), [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (17_AdamW device run, 2026-08-15)`
`unverified_on: soc=Ascend910_220x (a3), Ascend950PR (a5) — OL-81's CAST_RINT default remains the guidance there; do not import this restriction to those SoCs`

**Symptom**: fp16-path outputs read stale/uninitialized UB (narrowed outputs alternating `[0x0000, stale]`) while every WIDENED path (fp32/bf16) is clean; no error is raised — the narrowing cast silently did not happen. Grep a dav-m200 kernel for `RoundMode::CAST_RINT` on a half↔float cast: each hit is this bug.

**Mechanism**: the dav-m200 vconv implementation supports ONLY `CAST_NONE` for half→float and `CAST_NONE`/`CAST_ODD` for float→half (CANN 安装树内 asc/impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h:77-95,217-252). `CAST_RINT` is not in the whitelist; the impl reaches `ASCENDC_ASSERT(false)`, which in the release build skips the conversion rather than aborting — source and destination UB keep prior contents. Same dav-m200 SDK-surface-narrower-than-newer-CANN class as EC-197/PB-105.

**Numeric semantics of the supported modes** (probed on device, 17_AdamW):
- `CAST_NONE` f32→f16 behaves as RNE at normal magnitudes — 0 differing elements / 1M vs a CPU RNE cast; only overflow-boundary elements differ, because vconv SATURATES: NaN→65504, overflow→±65504. torch_npu's own `.to(fp16)` does the identical thing (probed); CPU torch casts overflow→±inf. Saturation-vs-inf is the canonical grader's documented forgiveness class (`compare.py:373`（a5_ops 路径，未迁移）).
- This RNE-at-normal-magnitudes probe PARTIALLY resolves the truncation-vs-RNE contest flagged in OL-81's layer_norm_quant row, for the fp32→fp16 pair; the fp16→int8 quant pair remains open.

**Fix**: use `CAST_NONE` for both directions on dav-m200 (`代码片段 17_AdamW/adamw_kernel.h.snippet.h（见文末）`, 6 sites). This entry PROMOTES the 2026-08-09 UNVERIFIED hedge in OL-81 to a confirmed constraint — its stated promotion condition ("a provenance-resolved run actually exercises `RoundMode::CAST_RINT` for fp32↔fp16 on dav-m200 and reproduces the assert/garbage") is now met: 17_AdamW shipped 6× CAST_RINT, device-observed the uninit-UB symptom, and the CAST_NONE switch cleaned the fp16 path.

**Evidence**: 17_AdamW 310p (2026-08-15, benchmark, `torch_npu.npu_apply_adam` ref): precision iter-1 fp16 outputs stale/uninit with fp32/bf16 clean (`17_AdamW/failures_ledger.md:12`); 6× CAST_RINT→CAST_NONE → pass_a 18/18 OVERALL_T1 (beats-CANN 18/18, 13/18 bit-exact), pass_b 645/645 T1 (605/645 bit-exact), det 21/21, perf 0.7408x mean device-event (`17_AdamW/pass_a_summary.json`, `pass_b_summary.json`, `perf_summary.json`). Device run — `device:1` recorded in all three summaries (note: [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) still carries the stale pre-device `device.runtime: NOT_RUN` flag; the run artifacts supersede it here). Prior art: inplace_add_layer_norm 2026-08-09 (the unverified hedge, OL-81); upstream-fidelity CAST_NONE choices — ge_glu_grad_v2 / mse_loss_v2 / rms_norm_grad 310p rows in OL-81.

**Cross-ref**: [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md) (CAST_RINT default — now per-SoC split; dav-m200 takes CAST_NONE), [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md) (fp32-internal + single CAST_RINT end-cast — the same per-SoC caveat applies on dav-m200; FLAGGED for owner: [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md) itself carries no dav-m200 caveat line yet), [`PB-107`](pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md) (fixpipe-flavor ±65504 saturation sibling), [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md) / [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (dav-m200 arch-surface gap class). backend=ascendc.

## 内嵌代码片段

### 17_AdamW/adamw_kernel.h.snippet.h

```cpp
        int32_t use_nesterov,
        int64_t total_elems,
        int64_t blk_idx, int64_t blk_cnt)
    {
        beta1_ = beta1;
        beta2_ = beta2;
        epsilon_ = epsilon;
        use_nesterov_ = use_nesterov;
        one_minus_beta1_ = one_minus_beta1;
        one_minus_beta2_ = one_minus_beta2;
        lr_corrected_ = lr_corrected;
        inv_one_minus_b2p_ = inv_one_minus_b2p;

        constexpr int32_t ALIGN = 16;
        int64_t elems_per_block = (total_elems + blk_cnt - 1) / blk_cnt;
        elems_per_block = ((elems_per_block + ALIGN - 1) / ALIGN) * ALIGN;

        int64_t start = blk_idx * elems_per_block;
        int64_t end = start + elems_per_block;
        if (end > total_elems) end = total_elems;
        if (start >= total_elems) { my_elems_ = 0; return; }
        my_elems_ = static_cast<int32_t>(end - start);

        gmVar_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(var_gm) + start, my_elems_);
        gmM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(m_gm) + start, my_elems_);
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v_gm) + start, my_elems_);
        gmGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(grad_gm) + start, my_elems_);

        pipe_.InitBuffer(varHInQ_, 1, ADAM_TILE * sizeof(half));
        pipe_.InitBuffer(gradHInQ_, 1, ADAM_TILE * sizeof(half));
        pipe_.InitBuffer(mInQ_, 1, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(vInQ_, 1, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(varHOutQ_, 1, ADAM_TILE * sizeof(half));
        pipe_.InitBuffer(mOutQ_, 1, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(vOutQ_, 1, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(varFBuf_, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(gradFBuf_, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(tmp1Buf_, ADAM_TILE * sizeof(float));
        pipe_.InitBuffer(tmp2Buf_, ADAM_TILE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (my_elems_ <= 0) return;
        int32_t loops = (my_elems_ + ADAM_TILE - 1) / ADAM_TILE;
        for (int32_t i = 0; i < loops; i++) {
            int32_t offset = i * ADAM_TILE;
            int32_t cur = ADAM_TILE;
            if (offset + cur > my_elems_) cur = my_elems_ - offset;
            int32_t aligned = ((cur + 15) / 16) * 16;
            CopyIn(offset, aligned);
            Compute(aligned);
            CopyOut(offset, aligned);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t count)
    {
        LocalTensor<half> varH = varHInQ_.AllocTensor<half>();
        LocalTensor<half> gradH = gradHInQ_.AllocTensor<half>();
        LocalTensor<float> mIn = mInQ_.AllocTensor<float>();
        LocalTensor<float> vIn = vInQ_.AllocTensor<float>();
        DataCopy(varH, gmVar_[offset], count);
        DataCopy(gradH, gmGrad_[offset], count);
        DataCopy(mIn, gmM_[offset], count);
        DataCopy(vIn, gmV_[offset], count);
        varHInQ_.EnQue(varH);
        gradHInQ_.EnQue(gradH);
        mInQ_.EnQue(mIn);
        vInQ_.EnQue(vIn);
    }

    __aicore__ inline void Compute(int32_t count)
    {
        LocalTensor<half> varH = varHInQ_.DeQue<half>();
        LocalTensor<half> gradH = gradHInQ_.DeQue<half>();
        LocalTensor<float> m = mInQ_.DeQue<float>();
        LocalTensor<float> v = vInQ_.DeQue<float>();

        LocalTensor<half> varHOut = varHOutQ_.AllocTensor<half>();
        LocalTensor<float> mOut = mOutQ_.AllocTensor<float>();
        LocalTensor<float> vOut = vOutQ_.AllocTensor<float>();
        LocalTensor<float> varF = varFBuf_.Get<float>();
        LocalTensor<float> gradF = gradFBuf_.Get<float>();
        LocalTensor<float> t1 = tmp1Buf_.Get<float>();
        LocalTensor<float> t2 = tmp2Buf_.Get<float>();

        Cast(varF, varH, RoundMode::CAST_NONE, count);
        Cast(gradF, gradH, RoundMode::CAST_NONE, count);

        // m_new
```

**孪生卡 pb-173 合并的独有内容**（pb-173-dav-m200-vconv-accepts-only-a-per-direction-cast.md 已删除）：

> **Evidence**: 17_AdamW 310p (2026-08-15, benchmark, `torch_npu.npu_apply_adam` ref): precision iter-1 fp16 outputs stale/uninit with fp32/bf16 clean (`17_AdamW/failures_ledger.md:12`); 6× CAST_RINT→CAST_NONE → pass_a 18/18 OVERALL_T1 (beats-CANN 18/18, 13/18 bit-exact), pass_b 645/645 T1 (605/645 bit-exact), det 21/21, perf 0.7408x mean device-event (`17_AdamW/pass_a_summary.json`, `pass_b_summary.json`, `perf_summary.json`). Device run — `device:1` recorded in all three summaries (note: [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) still carries the stale pre-device `device.runtime: NOT_RUN` flag; the run artifacts supersede it here). Prior art: inplace_add_layer_norm 2026-08-09 (the unverified hedge, OL-81); upstream-fidelity CAST_NONE choices — ge_glu_grad_v2 / mse_loss_v2 / rms_norm_grad 310p rows in OL-81.

> **Evidence (2nd op — overflow-semantics confirmation + NaN-cascade origin)**: instance_norm_grad 310p (2026-09-02, benchmark-acceptance lane, device micro-test `instance_norm_grad/failures_ledger.md` precision iter 1): std=0.01 case — the CPU dtype-native fp16 golden cascades inf→NaN (elementwise `pow(std,-3)` overflows fp16 to ±inf, then inf·finite / inf−inf) while the CAST_NONE kernel output stays finite (max 5424). Triage fact: the NaN cascade originates at the ELEMENTWISE overflow point, NOT the reductions — torch fp16 reductions accumulate in fp32 (acc_type); the pure-reduction outputs (grad_bias/grad_weight) were bit-exact on all 19 fp16 cases. When the grading golden IS IEEE-native, reproducing ±inf from AscendC requires explicit reconstruction: mask = |x| ≥ 65520 via `Mins`/`Maxs`, then `Div` by (1−mask); only worthwhile when the golden demands IEEE overflow semantics.

> **Evidence (3rd op — strike in a narrowing-cast chain; CORRECTED 2026-09-05 kw-2: the strike stands device-verified, the originally-recorded fix narrative described a worker-MODIFIED reference)**: 28_MultimodalRopePositionComputationWithGridBasedIndexing 310p (2026-09-05, benchmark, DEVICE-RUN device 3): Phase D iter-1 `Cast(f32→f16, CAST_RINT)` at a per-op round-trip site compiled clean but never executed — fp16 outputs were stale fp32 bits read as fp16 (all-ones probe: 0x3F800000 halves → 0.0/1.875 alternating), 3 benchmark + 9 edge cases corrupt while fp32/bf16 legs clean (`28_MultimodalRopePositionComputationWithGridBasedIndexing/failures_ledger.md` precision iter 1). Correction: kw-1 recorded the fix as "per-op fp16 c10-Half semantics" — that targeted a worker-modified model.py (sha e0d6c71d); the sha-pinned vendor def (f195e397) prescribes unified-fp32 accumulation, and the SHIPPED kernel consumes `CAST_NONE` exactly once, as the single final RNE round `Cast(ob, acc, RoundMode::CAST_NONE, hh)` (`代码片段 28_MultimodalRopePositionComputationWithGridBasedIndexing/mrope_kernel.h.snippet.h（见文末）`) — bit-exact on every dtype (pass_a 50/50 PASS_T1, ours_mere=0 on all 6 low-dtype cases). Either way the PB-120 rule is unchanged: f32→f16 narrowing on dav-m200 = `CAST_NONE` only. Provenance: OL-401 (revised).

> **Cross-ref**: [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md) (CAST_RINT default — now per-SoC split; dav-m200 takes CAST_NONE), [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md) (fp32-internal + single CAST_RINT end-cast — the same per-SoC caveat applies on dav-m200; FLAGGED for owner: [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md) itself carries no dav-m200 caveat line yet), OL-401 (revised — truth-source sha fidelity; this op's per-dtype narrative superseded), [`PB-107`](pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md) (fixpipe-flavor ±65504 saturation sibling), EC-96 / [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (dav-m200 arch-surface gap class). backend=ascendc.

## 适用场景

- **工程形态**：dav-m200 vconv 的 cast 模式按方向白名单。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
