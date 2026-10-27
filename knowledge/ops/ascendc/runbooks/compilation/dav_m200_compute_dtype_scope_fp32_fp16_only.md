---
type: Runbook
title: "dav-m200 (Ascend310P3) compute-dtype scope is fp32/fp16 only — the ascend310p AICoreConfig registers {DT_FLOAT, DT_FLOAT16}; drop BF16"
description: "dav-m200 (Ascend310P3) compute-dtype scope is fp32/fp16 only — the ascend310p AICoreConfig registers {DT_FLOAT, DT_FLOAT16}; drop BF16"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["AICoreConfig"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-412 — OL-412: A5Ops additive extension

> 关联：同一 OL-412 主题已由既有卡 [ol_412_dav_m200_ascend310p3_compute_dtype_scope_is_fp32.md](ol_412_dav_m200_ascend310p3_compute_dtype_scope_is_fp32.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any op ported to the ascend310p SoC config whose upstream supports BF16 on newer arches)。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0 (adaptive_avg_pool3d 310p, 2026-08-10 — op_host config-level evidence)。

**OL-412** — dav-m200 (Ascend310P3) compute-dtype scope is fp32/fp16 ONLY: the `ascend310p` `AICoreConfig` registers `{DT_FLOAT, DT_FLOAT16}` (drop BF16 — the upstream `__CCE_AICORE__ >= 220` BF16 branch is dead on this SoC). CONFIG/registration-level fact; arch < 220 also lacks DataCopyPad (→ pybind pad per P-P66/EC-23). NOTE: the `__CCE_AICORE__==200 ↔ dav-m200` identity tensions the arch-value table (pre-existing KB inconsistency, NOT resolved here). Anchor 内部路径（不公开）workspace/adaptive_avg_pool3d/op_host/adaptive_avg_pool3d_def.cpp:58-74, 12/12 fp32/fp16. Cross-ref [P-P66](../../optimizations/p_p66_tquebind_vecin_vecout_1_for_in_place_ub_buffer_r.md), [EC-23](ec_23_datacopypad_ub_gm_crashes_on_ascend950pr_507035.md), [EC-88](cast_float_int8_no_arch_220x_overload.md)/93. backend=ascendc.

- Former A5Ops ID: `OL-412`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any op ported to the ascend310p SoC config whose upstream supports BF16 on newer arches)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (adaptive_avg_pool3d 310p, 2026-08-10 — op_host config-level evidence)`
`unverified_on: the STRONGER claim "BF16 is hardware-unsupported on dav-m200" is device-unverified (`hardware/target/ascend310p.md` `device.runtime: NOT_RUN`); only the config-registration fact (BF16 absent from the 310p config) is solid here.`

> [kb-manager] NOTE (skill step 2.7 / arch-table contradiction): the related "`__CCE_AICORE__ == 200` maps to dav-m200" identity is in tension with the arch-value table elsewhere in this file (which lists `100`→Ascend310/310P and `200`→Ascend910-original). That mapping is a PRE-EXISTING internal KB inconsistency NOT resolved by this entry — see the Mode 1 merge report for `adaptive_avg_pool3d`. OL-412 states the OBSERVABLE config-level fact (BF16 not registered for the ascend310p SoC) and deliberately avoids re-asserting the contested arch-number identity.

**Principle**: when porting an op to the dav-m200 / Ascend310P3 (`ascend310p`) SoC, the compute-dtype surface is narrower than on A5/220x: BF16 is not part of the registered dtype set for this SoC. The op's `op_host/<op>_def.cpp` `AICoreConfig` for `ascend310p` must therefore be declared `.DataType({ge::DT_FLOAT, ge::DT_FLOAT16})` only — NOT the `{DT_FLOAT, DT_FLOAT16, DT_BF16}` the generic (non-310p) configs carry. Carrying BF16 into the 310p config mismatches the SoC's compute surface; upstream ops-nn gates the BF16 branch behind `__CCE_AICORE__ >= 220`, which is dead on this SoC, so a faithful port drops BF16 from the 310p registration rather than shipping a branch that can never execute.

**Concrete anchor** (adaptive_avg_pool3d 310p, 内部路径（不公开）workspace/adaptive_avg_pool3d/op_host/adaptive_avg_pool3d_def.cpp:58-74): the `ascend310p_config` registers input `x` and output `y` as `.DataType({ge::DT_FLOAT, ge::DT_FLOAT16})` (lines 61, 66) — NO `DT_BF16` — while the non-310p configs in the same file (lines 25, 30, 40, 45) DO carry `{DT_FLOAT, DT_FLOAT16, DT_BF16}`. Line 74 `AddConfig("ascend310p", …)` (+ kirinx90 / kirin9030 aliases, L75-76) binds the fp32/fp16-only surface to this SoC family.
  ```cpp
  OpAICoreConfig ascend310p_config;
  ascend310p_config.Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}); // no DT_BF16
  // ...
  this->AICore().AddConfig("ascend310p", ascend310p_config);
  ```

**Scope / how to recognize the class**: fires for any op ported to the `ascend310p` / dav-m200 SoC config whose upstream source offers BF16 on newer arches (the `__CCE_AICORE__ >= 220` gate). Action: trim the 310p config dtype list to `{fp32, fp16}`; keep BF16 only in the non-310p configs. This is a CONFIG/registration-level fact (the `op_host` `.DataType` declaration), independent of device runtime.

**Corollary — no DataCopyPad on this arch**: the same arch boundary that drops BF16 also lacks `DataCopyPad` (arch < 220), so non-block-aligned GM↔UB transfers must use the pybind-side channel/axis padding convention (P-P66 / EC-23), not an in-kernel `DataCopyPad`. adaptive_avg_pool3d pads the innermost C dimension to `numPerBlock` in pybind (内部路径（不公开）workspace/adaptive_avg_pool3d/kernel/pybind11.cpp:57-101) — a 3rd-SoC-family instance of P-P66 (after V351/220x).

**Evidence**: adaptive_avg_pool3d 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — 12/12 PASS across 9 fp32 + 3 fp16 cases (CPU-truth; A3 device ref unreachable). The 310p config ships fp32/fp16-only; BF16 was never exercised. `[first evidence]` — now corroborated by a 2nd dav-m200 op (below); the config-level fact is 2-op-supported across two flavors (config-registration + kernel-source-guard).
  - **ascend_quant_v2 310p (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — 2nd-op confirmation, SOURCE-GUARD flavor**: this quantize-to-int8 op is a NPUKernelBench raw-launch pybind kernel (no `op_host` AICoreConfig), so its BF16 exclusion is at the KERNEL-SOURCE level, not the config-registration level — the 220x upstream guards the BF16 path behind `#if __CCE_AICORE__ == 220` (内部路径（不公开）kernel/ascend_quant_v2_kernels.cpp:8: "Supported dtypes: FP16, FP32 — BF16 NOT supported on 310P"), which is dead on arch 200x, so only FP16 + FP32 compile. SAME arch-boundary mechanism as adaptive_avg_pool3d's config-level drop (the `>= 220` BF16 gate), observed at a different layer — corroborating BF16 is gated out across the 310p SoC surface, not just one op's config. 9/9 fp16 int8-bit-exact vs CPU fp64 truth. Device provenance UNVERIFIED (`back/310p-hw-spec-pages` `device.runtime: NOT_RUN` / FORBIDDEN; `verification.json` `method` claims a device run the branch hardware doc contests).

**Other instances (predicted)**: any pooling / elementwise / normalization op ported to `ascend310p` whose upstream carries a BF16 branch — adaptive_avg_pool2d, max_poolNd, layer-family ops on dav-m200. Each should drop BF16 from the 310p config the same way.

**Cross-ref**: P-P66 / EC-23 (pybind padding when `DataCopyPad` is absent on arch < 220 — the alignment corollary), the adv_api `SupportType<T,…>` caveat (CONTRAST: bf16 IS supported on Ascend950PR/A5 primitives — different chip family, no conflict; verify dtype support via the `c310_impl.h` static_assert, not the header docstring), EC-88 / EC-194 (companion arch-surface-reduction entries for dav-m200: missing primitives). backend=ascendc.
