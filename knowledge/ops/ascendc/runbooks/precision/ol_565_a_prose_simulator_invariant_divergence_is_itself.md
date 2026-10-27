---
type: Runbook
title: "A prose/simulator invariant divergence is itself a defect carrier — regenerate prose invariants from the simulator's formula"
description: "离线仿真与文字不变量发散时，仿真能跑所以被验证、文字没人测但被写 kernel 的人读——发散的文字不变量就是等着被实现的 bug，出现分歧默认文字错"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-565"]
---

# A prose/simulator invariant divergence is itself a defect carrier — regenerate prose invariants from the simulator's formula

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all (any op-gen flow that keeps an offline numerics simulation alongside analysis prose); workflow=verification`
`verified_on: soc=Ascend310P3; cann=9.0.0 (32_AdaptiveAvgPool3d Phase D iter-1, 2026-08-21 — the divergence WAS the bug)`
**Principle**: when an op carries BOTH an offline numerics simulation AND prose invariants (analysis.md, header comments, tiling notes), the two artifacts can silently disagree. The simulator is executable, so its formula is the one that gets validated; the prose is inert, so nothing ever tests it — and the kernel author reads the PROSE. A prose invariant that diverges from the sim's formula is therefore not a documentation nit: it is an **untested bug waiting to be implemented**, and it is invisible to every gate that checks the sim. Rule: every prose invariant must be re-derived from the same formula the simulator evaluates (or the sim extended to cover the prose claim); on any divergence, treat the PROSE as the defect carrier until proven otherwise — do not trust the prose because "the sim passed".
**Concrete anchor**: the offline numerics sim computed the adaptive-pool minimum window length as `wlen.min()` (mathematically `ceil(I/S)` — correct); the analysis prose and kernel both carried `p = floor(I/S)` (wrong, per `CAND-ADAPTIVE-POOL-MIN-WLEN-CEIL`). The sim's numerics proof stayed green the whole time — it never evaluated the prose's `floor`. The kernel (written from the prose) dropped the final element of every `q+2` lane; deterministic mid-scale fp32 error 0.22–0.39 on exactly those lanes. Fix `p = (W+Sw-1)/Sw` at `代码片段 32_AdaptiveAvgPool3d/adaptiveavgpool3d_kernel.h.snippet.h（见文末）` took fp32 to 1–2 ulp of the CPU-truth oracle; prose regenerated from the sim's formula in the same edit.
**Evidence**: 32_AdaptiveAvgPool3d 310p (2026-08-21, benchmark, Ascend310P3/dav-m200/arch-200x, CANN 9.0.0): the sim/prose divergence was the root cause of the only Phase D precision failure (failures_ledger iter 1 → resolution row); post-fix 100/100 + 168/168 + det 3-run bit-identical. (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`.)
**Other instances (predicted)**: any flow with paired artifacts where one is executable and one is narrative — tiling notes vs tiling emulators, precision-model spreadsheets vs actual accumulation order, API-contract comments vs interface tests. The general form: **the executable artifact is validated, the narrative artifact is consumed by the code author, and nothing checks their equality**. backend=all.

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
