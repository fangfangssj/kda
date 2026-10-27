---
type: Runbook
title: "dav-m200 composed-function VEC kernel faults deterministically — aicore 507015 / error 0x100000 / vec error info 0x1038 (stable) in the FIRST tile — while every sub-mechanism passes individually in the same op (15-rung in-kernel bisect); the failure is composition-shaped, NOT primitive-shaped [dav-m200/Ascend310P3, UNRESOLVED]"
description: "kernel 首 tile 即 aicore exception 507015（vec error info 0x1038 稳定）——与输入规模无关的首 tile 崩溃特征。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-185"]
---

# dav-m200 composed-function VEC kernel faults deterministically — aicore 507015 / error 0x100000 / vec error info 0x1038 (stable) in the FIRST tile — while every sub-mechanism passes individually in the same op (15-rung in-kernel bisect); the failure is composition-shaped, NOT primitive-shaped [dav-m200/Ascend310P3, UNRESOLVED]

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=dav-m200 toolchain; op_class=all (any single-function multi-stage VEC kernel: bf16 widen + long VEC chain + fold + multiple output paths)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (29_TanhGatedResidualAddBackward 310p spawn 3, 2026-09-06, DEVICE-RUN — deterministic across n=64..16384, 4 structural variants, ~10 rebuilds)`
`unverified_on: soc=Ascend950PR (351x), Ascend910 (220x) — composition-level codegen fault not observed there; do not import`
- **Symptom**: kernel K1 (`tgra_bwd_main_1`) faults in its first tile — `aicore exception 507015`, error `0x100000`, **vec error info 0x1038 stable across every run**, mte error info varies (0x56/0x3b/0xf/0x87), core 0. Identical ~1.5µs duration from n=64 (single tile) to n=16384 → first-tile fault, shape-independent.
- **What it is NOT (rung-proven inside the same op)**: scalar GM stores (rung 214), V-written f32 TBuf + DataCopy out (215), S-written u16 TBuf + fences + DataCopy out (216), UB roundtrip Duplicate+GetValue+RNE (218), TQue MTE2 loads (231/241/242), 4-tensor widen + passthrough (251), widen + partial ws write (252), exact while-loop scaffold (253), TanhVec chain (246) — every piece passes in isolation. Fence-pairing fix (iter 6) and a clean stage-knob-stripped rewrite (iter 7) changed nothing. NOT OL-189 contention (identical stable signature across an idle device and ~10 rebuilds); NOT the PB-21 PipeBarrier class (rung-216 fences work); NOT a single-primitive defect (CAND-DAV-M200-TBUF-VEC-OPS stays blocked — do NOT re-propose "primitive X broken on dav-m200" from this fault).
- **Four structural compositions, all fail (2 silent, 2 loud)**: (a) separate ProcessTile fn + runtime `if (stage_ >= k)` stage branches → SILENT zero-write (see CAND-DAV-M200-NO-INLOOP-STAGING); (b) inline tile body + stage branches → silent zero-write; (c) inline unbranched single pipeline → loud 0x1038 fault; (d) swi_glu per-input-queue dataflow → same loud fault. Removing ALL in-loop runtime branches converted the silent death into the loud deterministic fault.
- **Status: OPEN — unresolved, evidence-complete.** Recommended next steps (from the op's verification.json runtime_debug): decompose into 3 stream-ordered rung-sized launches (K1a loads+widen+partial fold / K1b tanh path / K2 fold+grad_gate — every piece individually rung-proven), or a bisheng `-O0` test (codegen-bug hypothesis), or disassemble faulting PC 0x124600200000 against `device_aic.o`.
- **Evidence**: 29_TanhGatedResidualAddBackward 310p spawn 3 (2026-09-06, benchmark, DEVICE-RUN): pass_a 0/50 — "not reached — kernel faults (aicore 507015, vec 0x1038) on every case before any output is produced" (`29_TanhGatedResidualAddBackward/verification.json`); 15-rung bisect ladder + 8 novel ledger iterations (`29_TanhGatedResidualAddBackward/failures_ledger.md`); anchors `代码片段 29_TanhGatedResidualAddBackward/tgra_bwd_kernel.core.snippet.h（见文末）`, `29_TanhGatedResidualAddBackward/analysis.md`.
- **Other instances (predicted)**: any dav-m200 op that fuses widen + long VEC chain + reduction fold + multi-output in ONE function — fused backward activations, gated residuals, fused norm-backward.
- **Cross-ref**: [`OL-189`](ol_189_507015_aicore_exception_npu_contention.md) (contention ruled out by deterministic repro), [`PB-21`](pb_21_pipebarrier_pipe_all_between_tbuf_resident_mte2_lo.md) (fence class ruled out), (primitive-defect framing blocked; this entry IS the composition-level residue), CAND-DAV-M200-NO-INLOOP-STAGING (the silent variant), CAND-DAV-M200-BULK-LOAD-MTE2, P-P131 (the probe triad that localized it). backend=ascendc.

## 内嵌代码片段

### 29_TanhGatedResidualAddBackward/tgra_bwd_kernel.core.snippet.h

```cpp

// ---- bf16 <-> fp32 scalar conversion (dav-m200 has no native bf16 VEC path) --
__aicore__ inline float bf16_to_f32(uint16_t u)
{
    union { uint32_t i; float f; } c;
    c.i = static_cast<uint32_t>(u) << 16;
    return c.f;
}

// RNE fp32 -> bf16, ties-to-even — matches PyTorch .to(torch.bfloat16).
__aicore__ inline uint16_t f32_to_bf16_rne(float f)
{
    union { uint32_t i; float f; } c;
    c.f = f;
    uint32_t b = c.i;
    b += 0x7FFFu + ((b >> 16) & 1u);
    return static_cast<uint16_t>(b >> 16);
}

// ---- cross-pipe fences on CACHED event IDs (swi_glu svId_/vsId_ form: the id
// is fetched ONCE in Init and reused; NO per-call FetchEventID — OL-94 anchor +
// multi_scale pool-exhaustion row; NO ReleaseEventID, 7_Sum ships this form) --
__aicore__ inline void SyncSV(TEventID id)
{
    SetFlag<HardEvent::S_V>(id);
    WaitFlag<HardEvent::S_V>(id);
}
__aicore__ inline void SyncVS(TEventID id)
{
    SetFlag<HardEvent::V_S>(id);
    WaitFlag<HardEvent::V_S>(id);
}
__aicore__ inline void SyncVM3(TEventID id)
{
    SetFlag<HardEvent::V_MTE3>(id);
    WaitFlag<HardEvent::V_MTE3>(id);
}
__aicore__ inline void SyncM2V(TEventID id)
{
    SetFlag<HardEvent::MTE2_V>(id);
```
