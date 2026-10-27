---
type: Runbook
title: "310P data-movement kernels MUST ship MULTI-CORE (DataCopy-only + K-aligned row groups) — single-core is NOT an acceptable fallback; the multi-core scalar-GM-store drop (PB-78) is a reason to eliminate scalar stores, not a reason to retreat to single-core"
description: "310P 纯搬运类 kernel 必须交付多核版本——单核正确但慢不是终点（单核慢约 7 倍）；标量写被丢弃是写成 DataCopy-only 的理由，不是退回单核的理由"
tags: [task.optimization, topic.data_movement.data_copy, topic.programming_model.execution_unit.scalar, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-683"]
---

# 310P data-movement kernels MUST ship MULTI-CORE (DataCopy-only + K-aligned row groups) — single-core is NOT an acceptable fallback; the multi-core scalar-GM-store drop (PB-78) is a reason to eliminate scalar stores, not a reason to retreat to single-core

## 优化点

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; op_class=data-movement / data-rearrangement (pixel_shuffle/unshuffle, permute, transpose, gather, scatter — pure data movement, zero arithmetic)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (pixel_shuffle + pixel_unshuffle 2026-08-29 — 8/8 contract each + large maxerr=0)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the K-aligned group math is perBlock-parameterized; multi-core scalar-store drop is dav-m200-specific (PB-78), re-probe before relying`
**Principle**: a pure data-movement op on dav-m200 ships **multi-core**. Do NOT force `coreNum=1` as a blanket, and do NOT accept "single-core is correct-but-slow" as the endpoint. The multi-core scalar-GM-store drop (PB-78 — scalar `SetValue` to GM is silently dropped under concurrency, no fence fixes it) is the reason to write the kernel **DataCopy-only** (single-writer-per-32B-block ownership), NOT the reason to retreat to single-core. Single-core leaves ~7× on the table (pixel_shuffle aligned 4.19M: 4785µs single-core vs 648µs multi-core). The recipe is P-P130 (K-aligned row groups + DataCopy-only stores) layered on P-P129 (DataCopy load + Gather reorder + K-group store).
**Recipe (tiling side)**: `K = perBlock/gcd(wout, perBlock)`; `coreNum = min(GetCoreNumAiv(), totalRows/K)`; distribute **K-row groups** (former/tail computed in GROUP units, then × K to get row counts) so every core owns whole 32B blocks. `totalRows % K != 0` residual (< K rows) still triggers the kernel's scalar fallback → for that rare shape keep `coreNum=1` (correctness over parallelism); the DataCopy-only residual is the P-P123 overlap patch (not yet needed by the contract battery).
**Evidence**: pixel_shuffle + pixel_unshuffle 310p (2026-08-29): 8/8 contract each + large-shape maxerr=0 (4.19M el / 512 rows); shuffle aligned 4.19M 648µs vs 4785µs single-core = **7.4×**; the pre-fix failure (single-row tail cores → scalar fallback) reproduced the exact PB-78 signature (racy cores' output stayed NaN-prefill, non-deterministic which core dropped). ⚠ **The earlier "revert to single-core" decision was a STALE-JIT-CACHE misdiagnosis** — always `rm -f /root/atc_data/kernel_cache/Ascend310P3/te_<op>*` before re-testing after a kernel change (note: unshuffle's cache file is `te_pixelunshuffle_*`, NO underscore between `pixel` and `unshuffle`).
**Cross-ref**: P-P130 (multi-core partitioning recipe), P-P129 (vectorization), PB-78 (multi-core scalar-store drop — the root cause; NOT PB-73, which is the fp32→fp16 Cast issue), PB-77 (same-core 64B-sector scalar-store fence — distinct from PB-78), P-P123 (ragged tail + overlap patch). backend=ascendc.
- **masked_softmax_with_rel_pos_bias 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — fp16 masked-softmax (3-pass), TQue-only I/O corroborating this entry; the "MTE2_V/V_MTE3 unsupported" framing is the refutation-trap, NOT the lesson**: a 3-pass softmax (FindGlobalMax → ComputeExpSum → NormalizeOutput, each re-reading the GM inputs — no GM workspace, so no cross-pass MTE3→MTE2 round-trip) driven entirely by `TQue<VECIN,1>` for the three GM→UB inputs and `TQue<VECOUT,1>` for the UB→GM output, with `TBuf<VECCALC>` retained ONLY for fp32 compute scratch (`target/ascendc/examples/masked_softmax_with_rel_pos_bias/snippets.md`). Input tensors are `FreeTensor`'d AFTER the reduction's `V_S` sync guarantees all V reads complete (`:156-159`) — confirming the FreeTensor-timing rule. A VEC tree reduction (`Max`/`Add` + `FloorPow2` + tail fold + `SetFlag/WaitFlag<HardEvent::V_S>` before `GetValue`, `:13-52`) produced correct per-tile max/sum on dav-m200 — i.e. the VEC-reduce→scalar-read path works when fenced (consistent with the add_layer_norm_quant_v2 / layer_norm_v4 dav-m200 evidence and the confound theory in `CAND-DAV-M200-REDUCESUM-RETURNS-ZERO`). **Regression-risk note (do NOT over-read)**: the worker framed the TQue choice as "manual `SetFlag<HardEvent::MTE2_V>` is BROKEN / unsupported on dav-m200" — that framing is REGRESSION-RISK BLOCKED as `CAND-DAV-M200-MTE2V-VMTE3-UNSUPPORTED` (contradicts CAND-DAV-M200-PIPE-MTE2-V / OL-94; the shipped kernel never exercises explicit `MTE2_V` flags, so the PASS validates TQue, not the defect claim). The lesson is the OL-25 prescription — use TQue (or explicit `HardEvent::MTE2_V`) for GM↔UB; do NOT cite "explicit MTE2_V flags broken on dav-m200". **Provenance**: CPU-truth (`verification.json` `truth_source=cpu_canonical_via_synthetic_edge_dataset` — A3 device ref unreachable), 9/9 T1 PASS fp16, `observed_deterministic: true`, device-runtime `NOT_RUN` (same flag as sibling dav-m200 ports). PB-60-safe launch path (`.npu()`-only H2D, no launch-stream D2D cast, `pass_a_runner.py:74-76`). Anchor: `target/ascendc/examples/masked_softmax_with_rel_pos_bias/snippets.md`.
  - 26_AvgPool3d 310p (2026-08-17, benchmark, Ascend310P3/dav-m200/arch-200x, CANN 9.0.0): **Case A sub-case extended — the vendor op runs but is MATERIALLY LESS ACCURATE than CPU fp32 truth (a precision FLOOR, not corruption — distinct from the 22_Nonzero wrong-content flavor)**: `torch.nn.functional.avg_pool3d` carries ~1e-3 fp32 relative error vs CPU fp32 truth, and bf16 large-magnitude windows SATURATE to 65536.0 (= fp16 max) via internal fp16 accumulation. Ours is bit-exact vs CPU truth (mere=0.0, mare=0.0 on all 198 edge cases) — strictly MORE accurate than CANN. Grading topology (same as the 22_Nonzero row): `model.py` vendor-verbatim for perf baseline, `model_cpu_truth.py` (OL-68 Case A widen→fp32→avg_pool3d→RNE-narrow) as precision oracle via `precision_eval_two_tier.py` + `pass_b_runner.py` two-tier ladder → 72/72 Pass A + 198/198 Pass B OVERALL_T1. Direct ours-vs-CANN MERE/MARE grading is INVALID here (would fail 50 fp32 + 4 bf16 cases where ours is MORE correct than CANN). (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`.)
- **CAST_NONE fp16↔fp32, norm-BACKWARD instance (rms_norm_grad 310p, 2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5)**: the fp16 path casts `dy`/`x`/`gamma` fp16→fp32 and the result fp32→fp16 with `RoundMode::CAST_NONE` (`rms_norm_grad_kernel.h:113,141,143,189`), mirroring the upstream 220x cast mode. 9/9 T1 PASS (5 fp16 + 4 fp32). 6th dav-m200 instance of the "mirror upstream's per-dtype Cast mode (CAST_NONE for fp16↔fp32 on arch 200x)" rule after celu/ge_glu_grad_v2/fatrelu_mul/mse_loss_v2/layer_norm_quant — confirms it for a norm-**backward** (per-row reduction + cross-row dgamma accumulation) class on a PB-60-safe launch path. No new constraint: CAST_NONE is the upstream-fidelity choice here, not evidence that CAST_RINT is broken. **Device provenance UNVERIFIED** ([target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`) — the 9/9 PASS carries no confirmed device weight; the cast-mode-fidelity observation is a compile/algorithm-level fact (the build links) and holds regardless.
- **33_CTClossV3 310p (2026-08-22, Ascend310P3/dav-m200/arch-200x, CANN 9.1.0, benchmark, DEVICE-RUN — verification.json perf A/B + det on NPU device 0)**: **full software-transcendental Tier-1 confirmation on dav-m200** — per-cell log-sum-exp built entirely from the Tier-1 palette because the dav-m200 hw transcendental floor (~2⁻⁹) sits far above the T1 requirement (2⁻¹³): software exp via expm1 8th-order Horner + the `2^k` exponent-bit assembly, software log via atanh series range-normalized to s∈[1,3] **by construction** (max term e=1 exactly, so normalization is a single Compare/Select — no exponent-bit extraction), select-form −inf/nan/+inf poison handling. fp32-internal DP for ALL dtypes (pybind exact-widen fp32 staging — no legal in-kernel widen on this chip, CAND-DAV-M200-INT-VEC-OP-MATRIX). Result: Pass A 100/100 (fp32 37/37, fp16 32/32, bf16 31/31) + Pass B 236/236 at mere ≤9.2e-8 vs fp64 golden (ULP-scale) + det 100/100 ×2 runs — the Tier-1 software path delivers fp32-grade CTC loss on dav-m200 exactly as the decision tree prescribes for CPU-libm references. **Palette caveat for dav-m200**: the Tier-1 allowed-ops list includes VEC `ShiftLeft`/`ShiftRight` — they do NOT exist on dav-m200 (tensor-arg overloads unimplemented); the `2^k` trick must use the SCALAR `(k+127)<<23` form (`代码片段 33_CTClossV3/ctclossv3_kernel.h.snippet.h（见文末）`). Perf: 0.9698 mean / 0.7684 median ratio vs CANN (large-T cases 2.2–3.2× faster, small-T 1.5–2× slower on launch/staging overhead; sum-wall 1.13× faster) — the software-transcendental tax is amortized in a DP-shaped kernel, in contrast to op#2 SwiGLU's ~2× elementwise tax.
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ **EMPIRICAL-FINDING (downgraded from PENDING-VALIDATION)** — a3 agent inspected the archive on 2026-04-29 and found the dispatcher is NOT a generic shape-regime detector; it's a 6-entry hand-tagged scenario lookup table that ValueErrors on shapes outside its hand-curated set. Reviewed via `opencode` (Minimax M2.5 local) AND `codex` (gpt-5.5) — both confirmed amendment accurate, recommend EMPIRICAL-FINDING severity (not retraction). flagged: 内部路径（不公开）model_new_tilelang.py:30-32 mirrors the same archive-specific shape gate, so the issue is not AscendC-only — it's an archive-wide adapter pattern.
1. **内部路径（不公开）model.py:6-72** defines `SCENARIOS` — a Python list of **exactly 6 dicts**, each with a hard-coded `shape` tuple plus optional `tilelang_mode` tag. Only 3 of the 6 (multi_w, split_c, split_w) have a `tilelang_mode` field; the other 3 (big_kernel, normal, reduce_d) fall through to default `"auto"`.
2. **内部路径（不公开）model.py:73** builds `SCENARIO_BY_SHAPE = {s["shape"]: s for s in SCENARIOS}` — a 6-key lookup table.
   (内部路径（不公开）model_new_tilelang.py:30-32) — so this isn't an AscendC-only artifact;
**Evidence (2nd instance — import-order half of the same rule)**: sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 2 runtime): `model_new_ascendc.py` had to `import torch_npu` BEFORE importing the freshly built `sort_ext` extension — otherwise the extension's `libtorch_npu.so` dependency was not yet on the loader path at dlopen and the import failed. On the acceptance lane the CANN env is sourced by the harness script (OL-663); the torch_npu import order is the remaining in-process half.
- regression test for the marker-gated accept/reject — `test_a5_verify_path_fraud_gate.py:202`（a5_ops 路径，未迁移）
  - **ascend_quant_v2 310p (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — 2nd-op confirmation, SOURCE-GUARD flavor**: this quantize-to-int8 op is a NPUKernelBench raw-launch pybind kernel (no `op_host` AICoreConfig), so its BF16 exclusion is at the KERNEL-SOURCE level, not the config-registration level — the 220x upstream guards the BF16 path behind `#if __CCE_AICORE__ == 220` (`kernel/ascend_quant_v2_kernels.cpp:8`: "Supported dtypes: FP16, FP32 — BF16 NOT supported on 310P"), which is dead on arch 200x, so only FP16 + FP32 compile. SAME arch-boundary mechanism as adaptive_avg_pool3d's config-level drop (the `>= 220` BF16 gate), observed at a different layer — corroborating BF16 is gated out across the 310p SoC surface, not just one op's config. 9/9 fp16 int8-bit-exact vs CPU fp64 truth. Device provenance UNVERIFIED (`back/310p-hw-spec-pages` `device.runtime: NOT_RUN` / FORBIDDEN; `verification.json` `method` claims a device run the branch hardware doc contests).
- 26_AvgPool3d 310p kw-1 (2026-08-17, benchmark, Ascend310P3/dav-m200/arch-200x, CANN 9.0.0, `torch.nn.functional.avg_pool3d`): 2nd independent dav-m200 witness, and it extends the byte-offset fact — **BOTH `srcOffsetLocal[i]` AND `srcBaseOffset` are byte-granular** (the intf `offsetAddr = (uint64_t)srcLocal + srcBaseOffset` asserts `srcBaseOffset % sizeof(PrimType) == 0`, so the base must be pre-scaled by `sizeof(T)` too, not just the per-lane offsets). Count-mode Gather is 3002/3102/3510-only; dav-m200 (200x) decomposes to mask-mode (works). Sliding-window box-sum: `off_[ow]=(ow*sw)*4`, `srcBaseOffset=(extra+dw)*4` (fp32, ×4 byte scale), one Gather call per output row. 72/72 Pass A + 198/198 Pass B bit-exact vs CPU truth. (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`; the byte-offset semantics are a compile/header-level fact, the PASS carries no confirmed device weight.) Anchor: `代码片段 26_AvgPool3d/avgpool3d_kernel.core.snippet.h（见文末）`.

## 内嵌代码片段

### 33_CTClossV3/ctclossv3_kernel.h.snippet.h

```cpp
 * e=1.0 bit-exactly (d=0 path), so walk is ~sqrt(T)*delta ~ 5e-5 absolute on
 * loss magnitude >= ~1 -> rel ~5e-5 << 1.22e-4 (MERE) with ~2.4x margin.
 */

namespace ctclossv3_ops {

// Capacity (dataset maxima: L <= 251 -> S' = 2L+1 <= 503; C <= 500 -> C8 <= 512)
static constexpr int32_t S2_MAX = 512;
static constexpr int32_t C8_MAX = 512;

// Multi-sample lane batching (ko-3 2026-08-22): each block processes a GROUP
// of samples per t-step instead of one sample at a time. Per-sample s-vectors
// occupy 64-lane-aligned SEGMENTS of the shared S2_MAX-lane vectors; the DP
// math is lane-wise identical (bit-identical outputs), but the per-step
// dependency-stall cost (Opt3 falsification: latency-bound serial recurrence,
// lanes mostly idle at S2 << 512) is amortized over the group.
static constexpr int32_t LANE_UNIT = 64;              // VSEL/Compare granularity
static constexpr int32_t LANE_UNITS = S2_MAX / LANE_UNIT;  // 8 units of 64
static constexpr int32_t GRP_MAX = LANE_UNITS;        // >=1 segment per sample
__aicore__ inline float BitsToF32(uint32_t u) { F32Bits c; c.u = u; return c.f; }
__aicore__ inline uint32_t F32ToBits(float f) { F32Bits c; c.f = f; return c.u; }

// Scalar fp32 exp for the final 2-element combine — same math as the VEC path.
__aicore__ inline float ScalarExp(float x) {
    float kf = x * EXP_LOG2E;
    // floor(kf + 0.5) == rint for our domain kf in [-125.7, 0]
        // lengths): target min(N, NB) groups so N <= NB degenerates to the
        // original one-sample-per-block shape; N > NB packs ~N/NB samples per
        // group, closing early when the 512-lane budget would overflow.
        const int32_t nTarget = static_cast<int32_t>(
            nb < N_ ? (nSort < nb ? nSort : static_cast<int32_t>(nb)) : nSort);
        const int32_t per = nTarget > 0 ? (nSort + nTarget - 1) / nTarget : 1;
        int32_t gFirst[N_SORT_MAX + 1];
        int32_t nGroups = 0;
        {
            int32_t i = 0;
            while (i < nSort) {
                int32_t units = (2 * llenArr[order[i]] + 1 + 63) >> 6;
                int32_t cnt = 1;
                gFirst[nGroups++] = i;
        auto idx = bufIdx_.Get<uint32_t>();
        auto logp = bufLogp_.Get<float>();
        Gather<float>(logp, row, idx, 0, un);
        rowQue_.FreeTensor(row);
    }

    // element-wise fp32 copy helper (count elements)
        Mul(p, p, u, un); Adds(p, p, EXP_C5, un);
        Mul(p, p, u, un); Adds(p, p, EXP_C4, un);
        Mul(p, p, u, un); Adds(p, p, EXP_C3, un);
        Mul(p, p, u, un); Adds(p, p, EXP_C2, un);
        Mul(p, p, u, un); Adds(p, p, 1.0f, un);      // bracket 1 + u/2 + ... + u^8/8!
        Mul(p, p, u, un);                            // p = expm1(u)
        Adds(p, p, 1.0f, un);                        // p = e^u
        // 2^k via the celu-proven float bit trick
        Adds(k, k, EXP_K_BIAS, un);
```

### 26_AvgPool3d/avgpool3d_kernel.core.snippet.h

```cpp
 * 100/100 T1 + 168/168 Pass B on this box): EC-60 (blockDim >= 1 tolerated —
 * empty slices), EC-90/91 (AscendC qualification), EC-92 (no direct
 * integral->float static_cast in __aicore__ — widen through int32_t), EC-95
 * (no floorf — integer div only), EC-94/105 (int64 strides/divmod), EC-96
 * (count-form VEC), OL-178..181 (TBuf InitBuffer, output row padding).
 *
 * NO TQue anywhere: dav-m200 TQue EnQue/DeQue does not order MTE2<->V or
 * V->MTE3 (cousin probe 2026-08-24: a SINGLE SetFlag/WaitFlag(MTE2_V) pair
 * after DataCopy fails ~60-70% of launches; TWO back-to-back pairs after the
 * copy are deterministic 90/90 across 16..4096 elems). Every cross-pipe edge
 * is an explicit fetch/set/wait/release with a FRESH event id per use.
 */

namespace avgpool3d_ops {

// =====================================================================
// bf16 <-> fp32 scalar helpers (PB-4 payload path, cousin-proven).
// =====================================================================
__aicore__ inline float bf16_to_f32(uint16_t u) {
    union { uint32_t u32; float f32; } c;
    c.u32 = static_cast<uint32_t>(u) << 16;
    return c.f32;
}

__aicore__ inline uint16_t f32_to_bf16(float f) {
    union { uint32_t u32; float f32; } c;
    c.f32 = f;
    uint32_t b = c.u32;
    b += 0x7FFFu + ((b >> 16) & 1u);   // RNE round to nearest bf16, ties-to-even
    return static_cast<uint16_t>(b >> 16);
}

// fp16/fp32 scalar widen (direct static_cast on half is fine on this core —
// only bf16 carries PB-4; uint16_t bf16 payload uses bf16_to_f32 above).
template <typename T>
__aicore__ inline float ap_to_f32(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, half>) {
        return static_cast<float>(v);
```

## 适用场景

- **工程形态**：310P 纯搬运类 kernel 必须交付多核版本。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
