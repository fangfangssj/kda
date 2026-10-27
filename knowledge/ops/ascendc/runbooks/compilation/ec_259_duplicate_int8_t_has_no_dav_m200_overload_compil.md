---
type: Runbook
title: "`Duplicate<int8_t>` has no dav-m200 overload — compile-time `SupportType` static_assert; duplicate through an int16 view (the vendor's own idiom)"
description: "（孪生家族之一）dav-m200 `Duplicate<int8_t>` 编译失败；另含自定义 tiling 结构体命名不能与内置撞名的修复。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-259", "Duplicate", "SupportType"]
---

# `Duplicate<int8_t>` has no dav-m200 overload — compile-time `SupportType` static_assert; duplicate through an int16 view (the vendor's own idiom)

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=dav-m200 toolchain; op_class=all (any kernel zero-filling / broadcast-filling an 8-bit buffer)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (diag_v2 310p, 2026-08-09, compile-caught at worker iter 1)`
- **Symptom**: `Duplicate<int8_t>(...)` fails at COMPILE time — the dav-m200 `SupportType`
  static_assert accepts only half / int16 / uint16 / int32 / uint32 / fp32. Same
  missing-dtype-pair class as EC-88 (`Cast<float,int8_t>` has no arch-220x overload), one
  primitive over. Other 8-bit dtypes (uint8) are outside the same list — expected identical,
  unverified.
- **Fix — reinterpret to a supported width**: view the int8 buffer as int16
  (`buf.Get<int16_t>()`) and `Duplicate<int16_t>` through the view (element count halves;
  handle an odd tail by scalar `SetValue` or a 32B-aligned pad). This is not an invented
  workaround: upstream ops-math 仓（公开）/conversion/diag_flat/op_kernel/diag_flat_nd_to_2d.h:279,327
  ALWAYS Duplicates through an int16 view — the reinterpret idiom exists in the source
  precisely because the 200x VEC unit lacks 8-bit Duplicate. **Copy the idiom, not the
  dtype** (OL-336: the source's own 200x-compatibility choices are the design reference).
- **Detection**: any `Duplicate<int8_t>` / 8-bit `Duplicate` in a 200x-target TU → expect
  the static_assert. Read the SDK `SupportType` list for the TARGET arch before choosing a
  fill dtype (OL-130 header-lookup chain); do NOT shop dtypes by trial compile.
- **Evidence**: diag_v2 310p (2026-08-09, first instance) — iter-1 build failure on
  `Duplicate<int8_t>` zero-fill of the assist buffers; fixed via `Get<int16_t>()` view +
  `Duplicate<int16_t>`; then 8/8 bit-exact T1, det 8/8
  (`diag_v2/failures_ledger.md` iter 1; workspace scratch, gitignored).
- **Cross-ref**: EC-88 (the Cast float←int8 no-overload sibling on 220x), PB-59
  (DataCopyPad absent on dav-m200 — same SDK-capability-gap class, data-movement twin),
  P-P125 (dav-m200 cast/predicate support matrix — verify at SDK header level pre-build),
  OL-336, OL-130. backend=ascendc.
- **Companion case (entry-point signature, not internal cast)**: the auto-generated host wrapper that `build_ascendc.py` emits dereferences each kernel entry-point parameter with `*param` to forward it. That compiles when the param is declared `GM_ADDR` (so `*param` is a scalar address the wrapper can pass on), but FAILS to compile when the param is declared raw `__gm__ uint8_t*` — the wrapper declares its own copy as `__attribute__((cce_global)) uint8_t *param`, and `*param` then yields a scalar of the wrong shape for the launch stub. **Rule**: declare ALL global-memory pointer parameters on the kernel entry point as `GM_ADDR`, never as the raw `__gm__ uint8_t*`. This is the entry-point analogue of EC-2's inside-the-kernel cast rule — same `GM_ADDR` token, two distinct failure sites. Evidence: moe_init_routing_v2 (2026-08-10, Ascend310P3, CANN 9.0.0) Phase C iter 2 — changing every entry-point arg from `__gm__ uint8_t*` to `GM_ADDR` cleared the auto-gen wrapper compile. Failures-ledger: `iter 2 | compile | CAND-310P-AUTOGEN-GMADDR`.
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK — kernel launch pattern); OL-127 (220x build checklist — same "use `GM_ADDR`" rule at grep-item granularity).
- **Evidence (2nd — arch-200/dav-m200, multi-core)**: 12_Permute kw-1 (2026-08-14, Ascend310P3 / dav-m200 / `__CCE_AICORE__=200`, CANN 9.0.0, benchmark). Symptom matched exactly: a fp32 `DataCopy(gm, ub, 12)` (48 B, non-multiple of ALIGN=8) left positions 8–11 zero (only the first 32 B-aligned block landed); rounding every DataCopy count UP to ALIGN fixed the per-copy corruption. **Multi-core refinement (new dimension)**: when the moved rows are partitioned across the 7 vector cores, the naive "pad the tail up to ALIGN" (the root-cause above) over-writes past the row boundary into the NEXT core's partition — 6 core-boundaries × padded tail = cross-core corruption. The overlap-tail fix is cross-core-safe precisely because the re-copied `[cnt-ALIGN, cnt)` window ends exactly at the row boundary and never leaves the current core's row; rows with `cnt < ALIGN` (whole row below one alignment block) use a scalar `GetValue`/`SetValue` tail instead (`代码片段 12_Permute/permute_kernel.h.snippet.h（见文末）`). Reported result: 149/149 bit-exact (`torch.equal`), fp16 141/141 + bf16 5/5, det 20/20. **Provenance**: device UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) marks `device.runtime: NOT_RUN`, `compile_evidence_device_execution: FORBIDDEN` on this branch (no 310P device involved per policy); the alignment/overlap-tail PATTERN is arch-general, the reported device PASS carries no confirmed device weight. Cross-ref [`PB-22`](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md) (the 32 B alignment *quantum*, of which this is the alignment-*requirement* face), PB-59 (`DataCopyPad` absent on dav-m200 → forces plain `DataCopy` + tail), P-P123 (dav-m200 in-kernel ragged-tail recipe).
- **Generalization (this is NOT PadTiling-specific)**: `kernel_tiling.h` pre-defines a whole family of tiling structs in the `AscendC::tiling` namespace and pulls them into scope via `using AscendC::tiling::*` (`PadTiling`, `RmsNormTiling`, `LayerNormTiling`, …). Defining an outer/global struct whose name matches ANY of them raises an ambiguous-reference error. Convention: name every custom tiling/kernel struct `<OpName>KernelTiling` (or `<OpName>OpTiling`) so it cannot collide with the built-in `<X>Tiling` set. Compile-fact (C++ namespace pollution) — arch-agnostic; reproduces on every SoC that includes `kernel_tiling.h`.
- **Error pattern**: `error: reference to '<Name>Tiling' is ambiguous` (observed: `PadTiling`, `RmsNormTiling`).
- **Root cause**: CANN `kernel_tiling.h` defines the struct in `AscendC::tiling` and imports it via `using`; a custom struct with the same name at outer scope is ambiguous.
- **Fix**: Rename the custom struct to a name that cannot collide — `<OpName>KernelTiling` (e.g. `RmsNormKernelTiling`, not bare `RmsNormTiling`).
- **Correct pattern** (anchor, `target/ascendc/examples/rms_norm/snippets.md`):
  // NOTE: named RmsNormKernelTiling to avoid conflict with AscendC::tiling::RmsNormTiling
  struct RmsNormKernelTiling { uint32_t numCol; ... };
  - Pad V2 first build (2026-04-09) — `PadTiling` conflict; renamed to a unique struct.
  - rms_norm 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — `RmsNormTiling` conflicted with `AscendC::tiling::RmsNormTiling` (内部路径（不公开）kernel_tiling.h:565); renamed to `RmsNormKernelTiling`. 2nd instance, broadening the class from `PadTiling` to the whole `AscendC::tiling::*` family. Compile-fact (build-host), confirmed by the produced object; not a device-runtime claim — holds regardless of the branch's `device.runtime: NOT_RUN` flag. Anchor: `target/ascendc/examples/rms_norm/snippets.md`.
- **dav-m200 (Ascend310P3, arch 200x) — DataCopyPad AVOIDED; plain `DataCopy` with caller-guaranteed 32-byte alignment (multi_scale_deformable_attn_function 310p port, 2026-08-10, CANN 9.0.0, port_a3_to_a5)**: completes the SoC-dependency spectrum — 220x supports DataCopyPad UB→GM (per the CANN `#if __CCE_AICORE__==220` shim above), V351/arch35 works BOTH directions (group_norm_silu / modulate / top_k_top_p_sample above), and on dav-m200 the worker did NOT use DataCopyPad, porting instead to plain `DataCopy` with 32-byte-aligned counts. This was a DESIGN CHOICE, not a measured crash: the `(B,S,H,C)` value layout makes every channel vector start at a 32-byte-aligned GM offset because `embedDims % 8 == 0` (fp32) so `DataCopy` works directly. I.e. "DataCopyPad unsupported on dav-m200" is the worker's availability rationale, NOT a device-verified 507035 — a future dav-m200 op that genuinely needs sub-32B GM↔UB copies should PROBE DataCopyPad availability (compile + one-case device run) rather than assume the avoidance generalizes. Anchor: `target/ascendc/examples/multi_scale_deformable_attn_function/snippets.md` (layout comment) + `target/ascendc/examples/multi_scale_deformable_attn_function/snippets.md` (`DataCopy(valLocal, valueGm_[gmOffset], embedDims_)` scatter read). **Provenance**: CPU-truth (`truth_source=cpu_canonical_via_synthetic_edge_dataset`), 9/9 T1 PASS; `device.runtime: NOT_RUN` in [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) — the alignment/layout reasoning is compile-verifiable, but "DataCopyPad unsupported on dav-m200" is NOT device-confirmed.
- **dav-m200 (Ascend310P3, arch200) — the SIMT escape hatch is ABSENT** (added 2026-08-11, incre_flash_attention): dav-m200 has NO SIMT (SIMT is an arch35 (351x) feature), so the OL-242 `__simt_callee__` scalar-transcendental path does not exist here. On dav-m200 `expf` is rejected already at COMPILE time (`error: call to [host] function "expf" from [aicore] function`), not only at link. The only routes are (a) a SIMD high-level `Exp()` pass via this entry's SIMT+SIMD split, or (b) a hand-rolled polynomial `FastExp` in pure scalar S-pipe code. incre_flash_attention chose (b) for an all-scalar UB-free kernel (CAND-DAV-M200-UB-FREE-SCALAR-KERNEL); the resulting 1-ULP fp16 drift vs CANN is the no-FMA / polynomial-exp consequence catalogued in EC-201, NOT a libm-link failure. Failures-ledger: `iter 2 | compile | novel` (`expf` host-only).
- **dav-m200 (Ascend310P3, arch 200) scope — the fallback is CORRECT here, not a bug**: the SAME `#define bfloat16_t int16_t` mechanism fires on `__CCE_AICORE__ != 220`, and on dav-m200 bf16 is genuinely NOT a supported dtype (the SoC is fp16+fp32 only — `config310P`/def.cpp confirm; every dav-m200 310p port in this repo targets fp16/fp32, never bf16). So unlike A5 (where EC-65 says author a local copy with the fallback REMOVED), on dav-m200 the fallback is the truthful type alias and must NOT be "fixed" away — there is no native bf16 to restore. Scope discriminator: before applying EC-65's workaround, check `__CCE_AICORE__` / SoC — remove the `#define` only on a bf16-capable SoC (A5/351x, A3/220x); leave it on dav-m200/arch-200. (Evidence: `inplace_add_rms_norm/knowledge_update.md` Finding #5 + `inplace_add_rms_norm/PROGRESS.md` Phase A "No BF16 on 310P"; compile/def-level fact, holds regardless of the branch's `device.runtime: NOT_RUN` provenance flag.)
- **Cross-ref**: KB_INDEX [`EC-1`](ec_1_missing_aicore_on_helper_function.md)..[`EC-65`](ec_65_v220_op_kernel_base_h_define_bfloat16_t_int16_t_fa.md) row; OL-185 220x→V351 calibration anchor; [`PB-4`](pb_4_bf16_scalar_cast_failure.md) (bf16 scalar cast — a bisheng codegen bug on bf16-capable SoCs, distinct from this type-absence-on-arch-200 fact).
- **Evidence (13_Cat kw-1, 2026-08-13 — 2nd independent dav-m200 op, broadens to data-movement class)**: pure-copy op that moves bf16 *data* (not just fp16/fp32); the dav-m200 bisheng front-end rejects `bfloat16_t` outright (`unknown type name 'bfloat16_t'; did you mean 'float16_t'?`), independently re-confirming "no native bf16 on arch 200". NEW positive technique surfaced here (vs EC-65's "avoid/leave-the-fallback"): because Cat never interprets values, the kernel is templated on `int16_t` (2-byte == bf16 width) and `Adds(int16,0)` / `DataCopy` carry the bf16 bit pattern bit-exactly — no fp32 upcast needed. This is the data-movement-class companion to EC-65's compute-class guidance: for pure copy/transpose/reshape, int16_t reinterpret is preferable to host fp32 roundtrip. See candidate `CAND-DAV-M200-INT16-BF16-DATAMOVE`. 51/51 T1 bit-exact (MERE=0 MARE=0), bf16 13/13, edge 1098/1098, 2.64× vs torch.cat. Anchor: `代码片段 13_Cat/kernel.cpp.snippet.cpp（见文末）`.
  - add_layer_norm_grad cast_probe (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0): TQue InitBuffer called with 2 args (que, depth) → "too few arguments" compile error; added the per-buffer byte length → compile passed. Same TBuf-2-arg vs TQue-3-arg confusion on the OTHER CANN major version — the 3-arg requirement holds on dav-m200 CANN 9.0.0 too (scope extension beyond this entry's A5 9.1.T500 evidence base). (`add_layer_norm_grad/failures_ledger.md`, cast_probe compile row)
- **dav-m200 (Ascend310P3, arch 200x) variant — COMPILE-LOUD, not silent (layer_norm_quant 310p, 2026-08-09, CANN 9.0.0, port_a3_to_a5)**: on dav-m200 the same raw-`__gm__`-pointer-to-`DataCopy` mistake does NOT silently return zeros as on Ascend950PR — it is a hard **compile error** `no matching function for call to 'DataCopy(...)'` (the template overload set has no raw-pointer match on the old arch). Same root class (raw pointer vs GlobalTensor), same fix (declare `GlobalTensor<half>` members, `SetGlobalBuffer`, subscript `xGm_[offset]`); just caught at compile time instead of silently at runtime — easier to diagnose, but the same GlobalTensor discipline is mandatory. Distinct from OL-167/EC-23 (dav-m200 lacks `DataCopyPad` → use plain aligned `DataCopy`): that is about the *tail-pad primitive*, this is about the *pointer type* of the DataCopy arguments. Anchor: `代码片段 layer_norm_quant/layer_norm_quant_kernel.core.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]` (rewritten iter-1 to GlobalTensor members after the `no matching function for call to 'DataCopy'` rejection). **Compile-fact** (build-host phenomenon, confirmed by the produced `.o`) — holds regardless of the branch's `device.runtime: NOT_RUN` flag.

## 内嵌代码片段

### 12_Permute/permute_kernel.h.snippet.h

```cpp
        bool merged = false;
        int64_t SPd = 0, dstGapU = 0, nrc = 0;
        if (d >= 0 && R % Wv == 0 && R <= TILE_ELEMS) {
            SPd = 1;
            for (int32_t k = d + 1; k < nm1; ++k) SPd *= t_.shape[k];  // o-space stride of dim d (outer dims only)
            dstGapU = (SPd * R - R) / Wv;
            nrc = TILE_ELEMS / R;
            // input slab base + output base must be 32B-aligned for every slow
            // combo: every slow outer stride (input and output side) must be a
            // Wv multiple (R side already is).
            bool ok = dstGapU >= 0 && dstGapU <= 65535 && nrc >= 1;
            for (int32_t k = 0; ok && k < nm1; ++k) {
                if (k == d) continue;
                if (t_.in_stride[t_.inv_perm[k]] % Wv != 0) ok = false;
            }
            merged = ok;
        }
        // Opt19: keyed routing (structural class validated against the host
        // key; demote-only — kernel derivation is the correctness authority).
        const uint64_t key_cls = t_.tiling_key & TK_CLASS_MASK;
        const uint64_t struct_cls = merged ? TK_CLASS_MERGED : TK_CLASS_FALLBACK;
        const uint64_t eff_cls =
            (key_cls == struct_cls) ? struct_cls : TkWeaken(key_cls, struct_cls);
        if (eff_cls == TK_CLASS_MERGED) { ProcessMerged(d, SPd, dstGapU, nrc); return; }
        for (int64_t o = outer_start_; o < outer_end_; ++o) {
            int64_t in_off = 0, rem = o;
    __aicore__ inline void ProcessMerged(int32_t d, int64_t SPd, int64_t dstGapU, int64_t nrc) {
        constexpr int32_t TILE_ELEMS = TILE_BYTES / static_cast<int32_t>(sizeof(T));
        constexpr int32_t Wv = 32 / static_cast<int32_t>(sizeof(T));
        const int32_t nm1 = static_cast<int32_t>(t_.ndim) - 1;
        const int64_t R = t_.R;
        int32_t slow[MAX_RANK]; int32_t ns = 0;
        int64_t SPk[MAX_RANK];
        for (int32_t k = 0; k < nm1; ++k)
            if (k != d) slow[ns++] = k;
        for (int32_t i = 0; i < ns; ++i) {
            SPk[i] = 1;
            for (int32_t j = slow[i] + 1; j < nm1; ++j) SPk[i] *= t_.shape[j];
        }
        const int64_t L = t_.shape[d];
        const int64_t slow_cnt = t_.outer_size / L;
        int64_t per = (slow_cnt + blk_cnt_ - 1) / blk_cnt_;
        int64_t s0 = blk_idx_ * per;
        int64_t s1 = s0 + per;
            // Odometer tables (Opt7): per-output-dim input stride and output
            // extent, hoisted out of the per-element loop so the hot path is
            // adds only — replaces the per-element int64 div/mod chain of
            // OutLinearToInOff (dav-m200 has no hw int64 divider; the chain
            // cost ~most of the measured 147 cycles/elem).
            s_[k] = t_.in_stride[t_.inv_perm[k]];
            oshape_[k] = t_.shape[k];
        }
        int64_t total = t_.outer_size * t_.R;
        int64_t per_blk = (total + blk_cnt - 1) / blk_cnt;
        per_blk = ((per_blk + ALIGN - 1) / ALIGN) * ALIGN;
        j_start_ = blk_idx * per_blk;
        j_end_ = j_start_ + per_blk;
        if (j_end_ > total) j_end_ = total;
        if (j_start_ >= total) { j_start_ = total; j_end_ = total; }
        in_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x_gm), total);
        out_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y_gm), total);
        constexpr int32_t TILE_ELEMS = TILE_BYTES / static_cast<int32_t>(sizeof(T));
        // Opt8: gather-class detection + bigger buffers when the class holds.
        // Class (all dtype-agnostic, element units; Wv = 32B/elems, ROWv = 2*Wv):
        //   R % Wv == 0                     (32B-aligned output rows / counts)
        //   s_last % Wv == 0 && s_last >= 2*Wv  (aligned, non-overlapping gather rows)
        //   srcGap units (s_last/Wv - 2) <= 65535
        //   dstGap units (S_c/Wv - 1)  <= 65535  (S_c = out_stride of cdim)
```

### 13_Cat/kernel.cpp.snippet.cpp

```cpp
// NOTE: cited line 41 is 3 past EOF — clamped (citation drift)
    GM_ADDR src, GM_ADDR dst,
    int64_t total_elems, int64_t outer_size,
    int64_t cat_dim_size, int64_t inner_size,
    int64_t cat_offset, int64_t total_cat_dim)
{
    CatCopyBf16 op;
    op.Init(src, dst, total_elems, outer_size, cat_dim_size, inner_size,
            cat_offset, total_cat_dim, GetBlockIdx(), GetBlockNum());
    op.Process();
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

## 适用场景

- **工程形态**：（孪生家族之一）dav-m200 `Duplicate<int8_t>` 编译失败；另含自定义 tiling 结构体命名。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
