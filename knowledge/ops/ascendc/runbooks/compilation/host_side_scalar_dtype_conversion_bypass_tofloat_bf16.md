---
type: Runbook
title: "Host-side scalar dtype-conversion to bypass W11 `ToFloat<bf16>` restriction in pybind/ACLRT_LAUNCH_KERNEL ports"
description: "Host-side scalar dtype-conversion to bypass W11 `ToFloat<bf16>` restriction in pybind/ACLRT_LAUNCH_KERNEL ports"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-A3A5-23"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-A3A5-23 — Host-side scalar dtype-conversion to bypass W11 `ToFloat<bf16>` restriction in pybind/ACLRT_LAUNCH_KERNEL ports

## 适用场景

- 适用平台与范围：soc=Ascend950PR；cann=9.0.0；op_class=port_a3_to_a5 / elementwise-with-scalar-param（pybind/ACLRT_LAUNCH_KERNEL 路径）。
- 验证来源：soc=Ascend950PR cann=9.0.0 上 1 个算子证据（fatrelu_mul，8/8 T1 PASS）；A3 未验证（V220 无 W11 限制，该 pattern 为 A5 移植特有，unverified_on）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=port_a3_to_a5,elementwise-with-scalar-param`
`verified_on: soc=Ascend950PR; cann=9.0.0 (1-op evidence: fatrelu_mul)`
`unverified_on: A3 (V220 has no W11 restriction; pattern is A5-port-specific)`

**Predicted rule** (forward-looking, 1-op evidence):
For ops that take a small set (1-3) of scalar parameters of fp16/bf16/fp32 dtype, **extracting the scalar(s) to fp32 on host** in the pybind11 launcher via `tensor.to(at::kFloat).cpu().data_ptr<float>()[0]` and **passing them as fp32 kernel launch arguments** avoids the W11 `ToFloat<bfloat16_t>` restriction entirely. The kernel itself never reads a bf16 scalar from GM, never calls `ToFloat<bfloat16_t>`, never invokes any restricted intrinsic — yielding a uniform kernel template across fp16/bf16/fp32 input dtypes with no W11 conflict.

**Contrast with upstream V220 pattern**: V220 reads scalar from a 1-elem GM tensor via `inScalarGM.GetValue(0)` then casts via `ToFloat(threshold)`, which requires `__aicore__` specialization for bfloat16_t. On A5 (V351) this is restricted per W11. Host-side conversion is cleaner and side-steps the restriction at the kernel boundary.

**Pybind purity preservation**: `tensor.to(at::kFloat)` is a dtype conversion (managed by torch's tensor library), not a math operation. The pybind layer remains compute-free per the project's "no PyTorch/CANN delegation" rule — it only reshapes scalar parameters into a launch-arg-compatible form.

**Concrete anchor (fatrelu_mul, 2026-05-17)**:
```cpp
// workspace/fatrelu_mul/kernel/pybind11.cpp::run_fatrelu_mul
float threshold = threshold_tensor.to(at::kFloat).cpu().data_ptr<float>()[0];
// All three kernel entry points take `float threshold`:
//   fatrelu_mul_kernels.cpp::fatrelu_mul_fp32(..., float threshold, ...)
//   fatrelu_mul_kernels.cpp::fatrelu_mul_fp16(..., float threshold, ...)
//   fatrelu_mul_kernels.cpp::fatrelu_mul_bf16(..., float threshold, ...)
```
Result: 8/8 T1 PASS bit-exact across fp16/bf16/fp32. No W11 errors, no per-dtype `__aicore__` specialization needed for the scalar param.

**Byte-identity proof for each IEEE dtype** (validated on fatrelu_mul 2026-05-17 — explains why the host-side conversion does NOT introduce a precision delta vs upstream V220's kernel-side `GetValue(0) + ToFloat` chain):

- **fp32**: `.item<float>()` is identity host→device (single fp32 word read; no conversion needed).
- **fp16**: `.item<float>()` performs IEEE half→fp32 widening (zero-extend mantissa, re-bias exponent — single instruction, deterministic, bit-identical to AscendC's `(float)half_value` widening on device).
- **bf16**: `.item<float>()` performs bf16→fp32 by zero-padding the low 16 mantissa bits — bit-identical to AscendC's `ToFloat<bfloat16_t>(v)` on device. The widening conversion is exact (no rounding occurs) for both directions because bf16 is a strict prefix of fp32's bit layout.

This dtype-by-dtype identity proof generalizes to ANY 1-element scalar tensor input in the IEEE float family — same pattern transfers to other thresholds/alphas/limits without per-op identity re-derivation.

**Applicability** (predicted next ports): `clamp_min` (1 scalar), `add_scalar` (1 scalar), `topk_threshold` (1 scalar), `leaky_relu` (1 scalar `negative_slope`), `hardshrink` (1 scalar `lambd`), `softplus` (1 scalar `beta`), or any other A3→A5 port where the upstream V220 reads a scalar param from a tiny GM tensor and casts via `ToFloat`.

**Promotion gate**: needs validation on 2+ additional ports with scalar parameters (different op classes, e.g. clamp_min + leaky_relu) to confirm the pattern transfers across activation/threshold variants. If a second case shows the host-side conversion introduces a precision-mismatch vs CPU truth (e.g. due to host fp32 quantization differing from device-side cast), revisit.

**Cross-ref**:
- W11 (restricted `ToFloat<bfloat16_t>` intrinsic on A5 — see KB W11 entry if present, or W-restriction sweep notes)
- OL-143 (L1 mechanical port for port_a3_to_a5 — this pattern fits cleanly inside L1)
- OL-81 (CAST_RINT for bf16 — the per-dtype cast convention this pattern preserves at output, despite consolidating input scalar to fp32)
- P140 (pybind/ACLRT_LAUNCH_KERNEL path — host conversion is only meaningful in that mode; ops-nn/op_host/op_kernel/arch35 binary-registration path follows a different scalar-passing convention)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-A3A5-23，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
