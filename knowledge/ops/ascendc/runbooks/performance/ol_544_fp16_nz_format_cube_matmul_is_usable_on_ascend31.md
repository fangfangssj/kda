---
type: Runbook
title: "FP16 NZ-format Cube matmul IS usable on Ascend310P3/dav-m200 even though only INT8/FP4-quantized cube attentio"
description: "FP16 NZ 格式 cube matmul 在 310P3/dav-m200 上可用——可复用的是 ND↔NZ 数据搬运组合（NZ→NZ UB→L1、NZ SoftMax、enVecND2NZ 输出），附 dav-m200 特有编译坑"
tags: [task.optimization, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-544"]
---

# FP16 NZ-format Cube matmul IS usable on Ascend310P3/dav-m200 even though only INT8/FP4-quantized cube attentio

## 优化点

**OL-544** — **FP16 NZ-format Cube matmul IS usable on Ascend310P3/dav-m200 even though only INT8/FP4-quantized cube attention paths are documented — the ND↔NZ data-movement composition (ND→NZ on-the-fly, NZ→NZ UB→L1, NZ SoftMax, enVecND2NZ output) is the reusable cube+vec template.** FP16 Cube comes from the matmul lib with NZ operands (`MatmulType<TSCM,CubeFormat::NZ,half>`); the work is the data-movement chain, not the cube unit. Compile gotchas on dav-m200: `GetNormalConfig` is GLOBAL-ns (not `matmul::`), `GetBlockIdx`→`AscendC::`, L1 TQue has no `.Get<T>()` (pass handles). Evidence: prompt_flash_attention 2026-08-11 dav-m200 CANN 9.0.0 (27/27 fp16 PASS). Cross-ref [`OL-405`](ol_405_on_a_220x_dav_m200_port_the_upstream_if_cce_aico.md), [`OL-268`](../compilation/ol_268_tilingdata_two_registration_styles_mismatch.md), [`OL-545`](../compilation/ol_545_a_scalar_vec_op_muls_adds_on_an_nz_format_buffer.md), [`EC-203`](../compilation/ec_203_aclrt_launch_kernel_arg_count_is_arch_dependent.md), [`OL-186`](../precision/ol_186_v351_fa_cube_matmul_pv_required.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=attention-fwd / any FP16 cube+vec (CUBE_MIX) kernel on dav-m200`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — prompt_flash_attention, single op (27/27 fp16 PASS)`
`unverified_on: the composition is expected to transfer to other dav-m200 FP16 cube kernels, but is cross-validated only on prompt_flash_attention; the compile API gotchas are CANN-9.0.0/dav-m200 facts that may differ on other installs`

- **Category**: algorithm_selection / arch_capability
- **Loaded by**: Generator/Analyzer when authoring an FP16 cube+vec fused kernel on Ascend310P3/dav-m200
- **Principle**: dav-m200 ships no FP16 attention cube path — the documented 310P cube attention route is INT8/FP4-quantized (e.g. the swin quantized-score op). FP16 Cube matmul IS available on this arch via the matmul library with NZ-format operands (`matmul::MatmulType<TPosition::TSCM, CubeFormat::NZ, half, ...>`); the load-bearing work is the data-movement chain around the cube unit, not the cube unit itself. The reusable composition for a full FP16 cube+vec attention kernel:
  1. **GM ND → L1 NZ**: `CopyND2NZOnTheFly` (adapted from the swin INT8 pattern; works for FP16 too)
  2. **UB NZ → L1 NZ**: `CopyNZ2NZ` (softmax output → BMM2 A operand)
  3. **SoftMax on NZ-format input**: `SoftMax<half, true, false, true>` over the `[roundM, roundSkv]` NZ block (NOT an ND buffer)
  4. **BMM2 output → GM ND**: `GetNormalConfig(false, false, true)` (enVecND2NZ) so the final P@V result lands in GM as ND
- **Compile gotchas hit when authoring this on dav-m200** (small anchors — see EC-203 for the launch-macro one):
  - `GetNormalConfig` lives in the GLOBAL namespace, not `matmul::` — `matmul::GetNormalConfig(...)` is a compile error (内部路径（不公开）pfa_kernel.h:129).
  - `GetBlockIdx()` needs explicit `AscendC::GetBlockIdx()` qualification in this TU layout.
  - An L1 `TQue` has no `.Get<T>()` member — pass tensor handles between cube stages (the swin pattern); do not try to materialize an L1 TQue view.
- **Concrete anchor** (内部路径（不公开）pfa_kernel.h):
  ```cpp
  using a1T = matmul::MatmulType<TPosition::TSCM, CubeFormat::NZ, half, false>; // Q operand, NZ
  static constexpr auto CFG_ENVECND2NZ = GetNormalConfig(false, false, true);    // global ns; BMM2 NZ→GM ND
  ```
- **Action**: when an FP16 cube+vec kernel is needed on dav-m200, do NOT fall back to a quantized/INT8 path or a vec-only hack — compose the four-stage ND↔NZ chain above. Mirror the swin `CopyND2NZ`/`CopyNZ2NZ` helpers; run SoftMax on the NZ block directly.
- **Evidence**: prompt_flash_attention (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0) — first FP16 NZ Cube attention kernel on dav-m200; 27/27 fp16 cases PASS (`max_rel_err < 0.002`, verification.json). Worker-reported perf geomean ~2.07× vs the A3 910B2C aclnn baseline — NOT independently re-verified in this merge (raw-launch perf on dav-m200 needs device-event measurement, see OL-163). Source candidate: `CAND-310P-FP16-CUBE-ATTENTION`. Failures-ledger: iter-1 compile fixes (GetNormalConfig ns, GetBlockIdx ns, L1 TQue.Get).
- **Other instances (predicted)**: any FP16 cube+vec fused kernel on dav-m200 (cross-attention, self-attention, score-matmul + scoring); the ND↔NZ composition also transfers to any arch whose cube operands physically live in NZ (220x/arch22, V351/arch35) — but the compile gotchas are dav-m200/CANN-9.0.0-specific.
- **Cross-ref**: [`OL-405`](ol_405_on_a_220x_dav_m200_port_the_upstream_if_cce_aico.md) (310P port is not a 220x drop-in — this entry is the constructive complement: a from-scratch FP16 cube composition that works on `==200`), [`OL-268`](../compilation/ol_268_tilingdata_two_registration_styles_mismatch.md) (host tiling serialization for the same cube-MIX launch path), [`OL-545`](../compilation/ol_545_a_scalar_vec_op_muls_adds_on_an_nz_format_buffer.md) (the NZ scalar-op count precision rule found in this same kernel), [`EC-203`](../compilation/ec_203_aclrt_launch_kernel_arg_count_is_arch_dependent.md) (the launch-macro tilingKey arg-count gotcha on this arch), [`OL-186`](../precision/ol_186_v351_fa_cube_matmul_pv_required.md) (V351 forward FA needs cube-MatmulImpl P@V — the A5 sibling of "vec-only is a hack"), CAND-310P-MOE-TILING (sibling dav-m200 cube-MIX op). backend=ascendc.

## 补充证据（2026-10-10 合并自 `fp16_nz_cube_matmul_usable_on_dav_m200.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-544 — OL-544: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=attention-fwd / any FP16 cube+vec (CUBE_MIX) kernel on dav-m200。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0 — prompt_flash_attention, single op (27/27 fp16 PASS)。

**OL-544** — **FP16 NZ-format Cube matmul IS usable on Ascend310P3/dav-m200 even though only INT8/FP4-quantized cube attention paths are documented — the ND↔NZ data-movement composition (ND→NZ on-the-fly, NZ→NZ UB→L1, NZ SoftMax, enVecND2NZ output) is the reusable cube+vec template.** FP16 Cube comes from the matmul lib with NZ operands (`MatmulType<TSCM,CubeFormat::NZ,half>`); the work is the data-movement chain, not the cube unit. Compile gotchas on dav-m200: `GetNormalConfig` is GLOBAL-ns (not `matmul::`), `GetBlockIdx`→`AscendC::`, L1 TQue has no `.Get<T>()` (pass handles). Evidence: prompt_flash_attention 2026-08-11 dav-m200 CANN 9.0.0 (27/27 fp16 PASS). Cross-ref [OL-405](../compilation/upstream_aicore_lt_220_branches_are_vendor_200x_design.md), [OL-268](../compilation/ol_268_tilingdata_two_registration_styles_mismatch.md), [OL-545](../precision/nz_buffer_vec_op_must_span_full_padded_extent.md), [EC-203](../compilation/aclrt_launch_kernel_no_tilingkey_dav_m200.md), [OL-186](../precision/ol_186_v351_fa_cube_matmul_pv_required.md). backend=ascendc.

- Former A5Ops ID: `OL-544`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=attention-fwd / any FP16 cube+vec (CUBE_MIX) kernel on dav-m200`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — prompt_flash_attention, single op (27/27 fp16 PASS)`
`unverified_on: the composition is expected to transfer to other dav-m200 FP16 cube kernels, but is cross-validated only on prompt_flash_attention; the compile API gotchas are CANN-9.0.0/dav-m200 facts that may differ on other installs`

- **Category**: algorithm_selection / arch_capability
- **Loaded by**: Generator/Analyzer when authoring an FP16 cube+vec fused kernel on Ascend310P3/dav-m200
- **Principle**: dav-m200 ships no FP16 attention cube path — the documented 310P cube attention route is INT8/FP4-quantized (e.g. the swin quantized-score op). FP16 Cube matmul IS available on this arch via the matmul library with NZ-format operands (`matmul::MatmulType<TPosition::TSCM, CubeFormat::NZ, half, ...>`); the load-bearing work is the data-movement chain around the cube unit, not the cube unit itself. The reusable composition for a full FP16 cube+vec attention kernel:
  1. **GM ND → L1 NZ**: `CopyND2NZOnTheFly` (adapted from the swin INT8 pattern; works for FP16 too)
  2. **UB NZ → L1 NZ**: `CopyNZ2NZ` (softmax output → BMM2 A operand)
  3. **SoftMax on NZ-format input**: `SoftMax<half, true, false, true>` over the `[roundM, roundSkv]` NZ block (NOT an ND buffer)
  4. **BMM2 output → GM ND**: `GetNormalConfig(false, false, true)` (enVecND2NZ) so the final P@V result lands in GM as ND
- **Compile gotchas hit when authoring this on dav-m200** (small anchors — see EC-203 for the launch-macro one):
  - `GetNormalConfig` lives in the GLOBAL namespace, not `matmul::` — `matmul::GetNormalConfig(...)` is a compile error (内部路径（不公开）pfa_kernel.h:129).
  - `GetBlockIdx()` needs explicit `AscendC::GetBlockIdx()` qualification in this TU layout.
  - An L1 `TQue` has no `.Get<T>()` member — pass tensor handles between cube stages (the swin pattern); do not try to materialize an L1 TQue view.
- **Concrete anchor** (内部路径（不公开）pfa_kernel.h):
  ```cpp
  using a1T = matmul::MatmulType<TPosition::TSCM, CubeFormat::NZ, half, false>; // Q operand, NZ
  static constexpr auto CFG_ENVECND2NZ = GetNormalConfig(false, false, true);    // global ns; BMM2 NZ→GM ND
  ```
- **Action**: when an FP16 cube+vec kernel is needed on dav-m200, do NOT fall back to a quantized/INT8 path or a vec-only hack — compose the four-stage ND↔NZ chain above. Mirror the swin `CopyND2NZ`/`CopyNZ2NZ` helpers; run SoftMax on the NZ block directly.
- **Evidence**: prompt_flash_attention (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0) — first FP16 NZ Cube attention kernel on dav-m200; 27/27 fp16 cases PASS (`max_rel_err < 0.002`, verification.json). Worker-reported perf geomean ~2.07× vs the A3 910B2C aclnn baseline — NOT independently re-verified in this merge (raw-launch perf on dav-m200 needs device-event measurement, see OL-163). Source candidate: `CAND-310P-FP16-CUBE-ATTENTION`. Failures-ledger: iter-1 compile fixes (GetNormalConfig ns, GetBlockIdx ns, L1 TQue.Get).
- **Other instances (predicted)**: any FP16 cube+vec fused kernel on dav-m200 (cross-attention, self-attention, score-matmul + scoring); the ND↔NZ composition also transfers to any arch whose cube operands physically live in NZ (220x/arch22, V351/arch35) — but the compile gotchas are dav-m200/CANN-9.0.0-specific.
- **Cross-ref**: OL-405 (310P port is not a 220x drop-in — this entry is the constructive complement: a from-scratch FP16 cube composition that works on `==200`), OL-268 (host tiling serialization for the same cube-MIX launch path), OL-545 (the NZ scalar-op count precision rule found in this same kernel), EC-203 (the launch-macro tilingKey arg-count gotcha on this arch), OL-186 (V351 forward FA needs cube-MatmulImpl P@V — the A5 sibling of "vec-only is a hack"), CAND-310P-MOE-TILING (sibling dav-m200 cube-MIX op). backend=ascendc.

---
