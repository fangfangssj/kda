---
type: Runbook
title: "fixpipe L0C(fp32)→UB(fp16) on dav-m200 saturates at ±65504, never inf — CPU references must clamp-then-cast"
description: "fixpipe L0C(fp32)→UB(fp16) on dav-m200 saturates at ±65504, never inf — CPU references must clamp-then-cast"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-107 — PB-107: A5Ops additive extension

> 关联：同一 PB-107 主题已由既有卡 [pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md](pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=任何 CPU reference 模拟硬件 fixpipe cast 链的 fp16 cube kernel。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（add_lora 310p，2026-08-08 — 按 OL-82 的 on-device 最小复现）。

**PB-107** — **fixpipe L0C(fp32)→UB(fp16) on dav-m200 SATURATES at ±65504, never inf** — on-device repro (`torch.mm` fp16 chain: 1.44e6 → 65504 hw vs inf CPU `.half()`); a CPU reference emulating fp16 matmul chains MUST clamp-then-cast (`t.clamp(-65504,65504).half()`) at every fixpipe boundary. Caveat: torch_npu library kernels (`mul` saturates / `add` infs) ≠ raw-primitive semantics; raw VEC fp16 overflow UNVERIFIED. add_lora 310p 2026-08-08 (large_mag case → bit-exact). Cross-ref [PB-42](pb_42_v220_cube_aic_gemm_must_not_fixpipe_directly_to_th.md) (distinct: routing rule vs value semantics), OL-82. [dav-m200/Ascend310P3, fixpipe, precision-emulation]

- Former A5Ops ID: `PB-107`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=any fp16 cube kernel whose CPU reference emulates the hardware fixpipe cast chain`
`verified_on: soc=Ascend310P3; cann=9.0.0 (add_lora 310p, 2026-08-08 — on-device minimal repro per OL-82)`
`unverified_on: soc=Ascend950PR, Ascend910 (saturation-vs-inf is a fixpipe implementation choice — probe per arch before assuming)`

- **Severity**: MEDIUM — CPU-reference/harness mismatch, not a kernel bug; the kernel is correct and the REFERENCE is wrong, which inverts triage (the hardware truth gets "fixed" toward the wrong reference).
- **Symptom**: an fp16 cube kernel is bit-exact vs its CPU reference except on large-magnitude cases, where the CPU reference reports `inf` and hardware reports a finite `±65504` (fp16 max).
- **Mechanism**: on dav-m200 the fixpipe's fp32→fp16 downcast SATURATES to the fp16 max instead of overflowing to inf. A CPU reference built naively (`t.half()`) overflows to inf and mismatches. On-device minimal repro: a `torch.mm` fp16 chain producing an intermediate of 1.44e6 → hardware 65504, CPU torch `.half()` → inf.
- **Fix (CPU-reference side)**: emulate the fixpipe boundary with clamp-then-cast at EVERY fp32→fp16 cast that models a fixpipe transfer:
  ```python
  def _fixpipe_f16(t: torch.Tensor) -> torch.Tensor:
      return t.clamp(-65504.0, 65504.0).half()   # dav-m200 fixpipe saturates, never inf
  ```
  After the fix, the add_lora large_mag saturation case went bit-exact.
- **Caveat — library-kernel ≠ raw primitive**: a torch_npu probe showed `mul` saturating but `add` producing inf — torch_npu library-kernel behavior is NOT raw-primitive semantics. Raw AscendC `Muls`/`Add` fp16 overflow behavior on dav-m200 remains UNVERIFIED (add_lora's envelope cannot overflow there, so it was not probed). Do not extend the saturation rule to VEC-pipe fp16 arithmetic without a probe.
- **Evidence**: add_lora port_a3 310p (2026-08-08, Ascend310P3, CANN 9.0.0) — pre-fix case 19 (large_mag) CPU-reference inf vs hardware 65504; post-fix 45/46 bit-exact vs hardware-semantics CPU fp16 chain.
- **Related boundary (input-side dtype conversion is part of the ALGORITHM)**: the same reference-emulates-hardware rule applies when the KERNEL's contract includes a dtype conversion of the inputs. lin_space_d 310p (2026-08-10): upstream's `static_cast<float>` (RNE rounding of int32 start/end/step) is part of the algorithm — an fp64 golden anchored at the exact ints fails MARE near the zero crossing for ANY fp32 implementation. Golden = fp64 lift of the fp32-anchored start/end/step. General shape: anchor the fp64 golden at the CONVERTED values, never at the pre-conversion exact values.
- **Other instances (predicted)**: every fp16 cube op's CPU-truth / deferred-CPU-truth reference on 310P (LoRA variants, GEMM epilogues, fused matmul+bias). Likely transfers to other archs' fixpipe paths — probe per arch (OL-82) rather than assume. Distinct from PB-42 (220x fixpipe→GM fp16-output wrong values — a routing rule; PB-107 is the VALUE semantics of the L0C→UB downcast).
