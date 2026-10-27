---
type: Runbook
title: "`__aicore__` functions forbid a direct unsigned-integral→floating `static_cast` — widen through `int32_t` first"
description: "`__aicore__` functions forbid a direct unsigned-integral→floating `static_cast` — widen through `int32_t` first"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-193 — EC-193: A5Ops additive extension

> 关联：同一 EC-193 主题已由既有卡 [ec_193_aicore_functions_forbid_a_direct_unsigned_integr.md](ec_193_aicore_functions_forbid_a_direct_unsigned_integr.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，arch 200x）；CANN 9.0.0；op_class=all（任何将 uint32_t/uint16_t 转 float 的 __aicore__ 函数；规则归因于 __aicore__ 属性本身，可能全 arch 通用，A5 未验证）。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0（deep_norm_grad 310p iter-1 build，2026-08-09；layer_norm_quant、rms_norm_grad 复验见正文）。

**EC-193** — `__aicore__` functions forbid a direct unsigned-integral→floating `static_cast` ("cast between floating and unsigned integer variable is not allowed in aicore function"); widen through `int32_t` first (`static_cast<float>(static_cast<int32_t>(u))`). COMPILE rejection of a plain C++ cast in device code — DISTINCT from PB-26 (the AscendC `Cast()` primitive's runtime dtype-pair silent-garbage table). Likely general aicore (the `__aicore__` attribute is on every arch), evidenced on dav-m200. Anchor: 内部路径（不公开）workspace/deep_norm_grad/kernel/deep_norm_grad_kernel.h:43, 9/9 T1. Cross-ref [PB-26](pb_26_ascendc_tanh_fp32_primitive_bimodal_precision_floo.md), [OL-18](../precision/ol_18_every_code_change_must_be_followed_by_both_precisi.md), [EC-191](duplicate_int8_no_dav_m200_overload.md)/91. backend=ascendc.

- Former A5Ops ID: `EC-193`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a (aicore front-end rule); op_class=all (any __aicore__ function that converts a uint32_t/uint16_t to float)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (deep_norm_grad 310p iter-1 build, 2026-08-09)`
`unverified_on: soc=Ascend950PR (A5, arch 351x) — the rule is attributed to the aicore function attribute (present on every AscendC device function on every arch), so it is LIKELY general, not dav-m200-specific; re-confirm on a 351x kernel that needs a uint→float widening. Do not assume it is 220x-only.`

- **Category**: build / compile
- **Loaded by**: Generator/Worker when an `__aicore__` function converts an unsigned integer (loop bound, element count, dimension read as `uint32_t`) to `float` for reciprocal/scale math.
- **Symptom**: a line like `1.0f / static_cast<float>(D_)` where `D_` is `uint32_t` fails to compile with `error: cast between floating and unsigned integer variable is not allowed in aicore function`. The compiler accepts signed-integral→float and float→integral, but rejects unsigned-integral→float directly. (Distinct from PB-26, which is about the AscendC `Cast()` *primitive*'s dtype-pair table producing SILENT garbage for `uint8→float` at RUNTIME — this EC is a COMPILE rejection of a plain C++ `static_cast` in device code.)
- **Mechanism**: the bisheng/CCE aicore front-end disallows the unsigned-integral↔floating conversion in `__aicore__`-attributed functions. It is a codegen-rule restriction on the device-function attribute, not a header gap or a runtime primitive.
- **Fix**: cast through a signed `int32_t` intermediate: `static_cast<float>(static_cast<int32_t>(u))`. Safe when the unsigned value is in `INT32_MAX` range (element counts / dimensions always are; for true-uint32 magnitudes see the project int64 rule OL-18 — widen the intermediate instead of truncating).
- **Correct pattern** (anchor, 内部路径（不公开）workspace/deep_norm_grad/kernel/deep_norm_grad_kernel.h:43 — `D_` is `uint32_t`, the inverse-count needs `float`):
  ```cpp
  // D_ is uint32_t (loop/dim bound). Direct static_cast<float>(D_) is rejected.
  invD_ = 1.0f / static_cast<float>(static_cast<int32_t>(D_));
  ```
- **Evidence**: deep_norm_grad 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 build rejected `static_cast<float>(uint32_t)`; the int32-intermediate fix is in the shipped kernel, which builds clean (object `auto_gen_deep_norm_grad_kernels.cpp.o` produced) and reports 9/9 T1 PASS. Recorded in `workspace/deep_norm_grad/failures_ledger.md` iter 1. Device provenance UNVERIFIED (`hardware/target/ascend310p.md` `device.runtime: NOT_RUN`) — but this is a COMPILE claim (build-host phenomenon), confirmed by the produced `.o`, NOT a device-runtime claim, so the provenance flag does not block it.
- **2nd-op confirmation + alternative mitigation (layer_norm_quant 310p, 2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5)**: hit the identical `cast between floating and unsigned integer variable is not allowed in aicore function` rejection for `1.0f / numCol` where `numCol` is `uint32_t`. Took a DIFFERENT mitigation than the int32-widen above — computed `aveNum = 1.0f / numCol` **on the host (`pybind11.cpp`)** and passed it into the kernel via the packed params buffer, avoiding the in-kernel unsigned→float cast entirely. Both mitigations are valid; the choice is a **decision rule** — int32-widen keeps the math in-kernel (preferred when the value is a simple dim/loop bound); host-side precompute is preferable when the scalar feeds a long in-kernel chain and you want it computed once in host fp64 precision. Confirms the rule on a 2nd dav-m200 op after deep_norm_grad. Anchor: `workspace/layer_norm_quant/kernel/pybind11.cpp` (host `aveNum` precompute → params buffer) → `workspace/layer_norm_quant/kernel/layer_norm_quant_kernel.h` (consumes `aveNum`, no in-kernel reciprocal). **Compile-fact** (build-host), confirmed by the produced `.o`; not a device-runtime claim.
- **3rd-op confirmation (rms_norm_grad 310p, 2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5)**: hit the identical unsigned→float rejection computing the RMSNorm mean-scaling `avg_factor` — `1.0f / static_cast<float>(static_cast<int32_t>(N_))` where `N_` is `uint32_t` (内部路径（不公开）workspace/rms_norm_grad/kernel/rms_norm_grad_kernel.h:47). This is the same int32-widen mitigation as deep_norm_grad (the **decision-rule** choice above — value is a plain dim bound, so in-kernel int32-widen is preferred over layer_norm_quant's host-precompute). Confirms the rule on a 3rd dav-m200 op. Anchor: 内部路径（不公开）workspace/rms_norm_grad/kernel/rms_norm_grad_kernel.h:44-47 (`N_ = static_cast<uint32_t>(N)` then the int32-intermediate reciprocal). **Compile-fact** (build-host), confirmed by the produced `.o`; not a device-runtime claim — holds regardless of the branch's `device.runtime: NOT_RUN` flag.
- **Cross-ref**: PB-26 (AscendC `Cast()` primitive dtype-pair table — the RUNTIME silent-garbage sibling; distinct mechanism), OL-18 (int64/int32 value-range rule — when the int32 intermediate could overflow, widen instead), EC-191/EC-192 (companion dav-m200 compile facts). backend=ascendc.
