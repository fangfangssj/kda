---
type: Runbook
title: "fixpipe L0C(fp32)→UB(fp16) on dav-m200 SATURATES at ±65504, never inf"
description: "dav-m200 上 fixpipe L0C(fp32)→UB(fp16) 在 ±65504 处饱和而非 inf——与 CPU .half() 的 inf 行为不同，比对时注意"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.cube]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-107"]
---

# fixpipe L0C(fp32)→UB(fp16) on dav-m200 SATURATES at ±65504, never inf

**PB-107** — **fixpipe L0C(fp32)→UB(fp16) on dav-m200 SATURATES at ±65504, never inf** — on-device repro (`torch.mm` fp16 chain: 1.44e6 → 65504 hw vs inf CPU `.half()`); a CPU reference emulating fp16 matmul chains MUST clamp-then-cast (`t.clamp(-65504,65504).half()`) at every fixpipe boundary. Caveat: torch_npu library kernels (`mul` saturates / `add` infs) ≠ raw-primitive semantics; raw VEC fp16 overflow UNVERIFIED. add_lora 310p 2026-08-08 (large_mag case → bit-exact). Cross-ref [`PB-42`](pb_42_v220_cube_aic_gemm_must_not_fixpipe_directly_to_th.md) (distinct: routing rule vs value semantics), [`OL-82`](../performance/ol_82_msprof_scalar_ratio_0_9_scalar_pipe_saturated_repl.md). [dav-m200/Ascend310P3, fixpipe, precision-emulation]

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

**孪生卡 pb-160 合并的独有内容**（pb-160-fixpipe-l0c-fp32-ub-fp16-on-dav-m200-saturates-a.md 已删除）：

> - **Other instances (predicted)**: every fp16 cube op's CPU-truth / deferred-CPU-truth reference on 310P (LoRA variants, GEMM epilogues, fused matmul+bias). Likely transfers to other archs' fixpipe paths — probe per arch (OL-82) rather than assume. Distinct from PB-42 (V220 fixpipe→GM fp16-output wrong values — a routing rule; PB-107 is the VALUE semantics of the L0C→UB downcast).
