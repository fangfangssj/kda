---
type: Runbook
title: "dav-m200 (Ascend310P3) compute-dtype scope is fp32/fp16 ONLY: the ascend310p AICoreConfig registers {DT_FLOAT"
description: "dav-m200 的计算 dtype 范围只有 fp32/fp16（AICoreConfig 不注册 BF16），上游 >=220 的 BF16 分支在这颗料上是死代码"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-412"]
---

# dav-m200 (Ascend310P3) compute-dtype scope is fp32/fp16 ONLY: the ascend310p AICoreConfig registers {DT_FLOAT

**OL-412** — dav-m200 (Ascend310P3) compute-dtype scope is fp32/fp16 ONLY: the `ascend310p` `AICoreConfig` registers `{DT_FLOAT, DT_FLOAT16}` (drop BF16 — the upstream `__CCE_AICORE__ >= 220` BF16 branch is dead on this SoC). CONFIG/registration-level fact; arch < 220 also lacks DataCopyPad (→ pybind pad per P-P66/[`EC-23`](ec_23_datacopypad_ub_gm_crashes_on_ascend950pr_507035.md)). NOTE: the `__CCE_AICORE__==200 ↔ dav-m200` identity tensions the arch-value table (pre-existing KB inconsistency, NOT resolved here). Anchor `代码片段 adaptive_avg_pool3d/adaptive_avg_pool3d_def.cpp.snippet.cpp（见文末）`, 12/12 fp32/fp16. Cross-ref P-P66, [`EC-23`](ec_23_datacopypad_ub_gm_crashes_on_ascend950pr_507035.md), EC-88/93. backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any op ported to the ascend310p SoC config whose upstream supports BF16 on newer arches)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (adaptive_avg_pool3d 310p, 2026-08-10 — op_host config-level evidence)`
`unverified_on: the STRONGER claim "BF16 is hardware-unsupported on dav-m200" is device-unverified ([target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`); only the config-registration fact (BF16 absent from the 310p config) is solid here.`

> [kb-manager] NOTE (skill step 2.7 / arch-table contradiction): the related "`__CCE_AICORE__ == 200` maps to dav-m200" identity is in tension with the arch-value table elsewhere in this file (which lists `100`→Ascend310/310P and `200`→Ascend910-original). That mapping is a PRE-EXISTING internal KB inconsistency NOT resolved by this entry — see the Mode 1 merge report for `adaptive_avg_pool3d`. OL-412 states the OBSERVABLE config-level fact (BF16 not registered for the ascend310p SoC) and deliberately avoids re-asserting the contested arch-number identity.

**Principle**: when porting an op to the dav-m200 / Ascend310P3 (`ascend310p`) SoC, the compute-dtype surface is narrower than on A5/220x: BF16 is not part of the registered dtype set for this SoC. The op's `op_host/<op>_def.cpp` `AICoreConfig` for `ascend310p` must therefore be declared `.DataType({ge::DT_FLOAT, ge::DT_FLOAT16})` only — NOT the `{DT_FLOAT, DT_FLOAT16, DT_BF16}` the generic (non-310p) configs carry. Carrying BF16 into the 310p config mismatches the SoC's compute surface; upstream ops-nn gates the BF16 branch behind `__CCE_AICORE__ >= 220`, which is dead on this SoC, so a faithful port drops BF16 from the 310p registration rather than shipping a branch that can never execute.

**Concrete anchor** (adaptive_avg_pool3d 310p, `代码片段 adaptive_avg_pool3d/adaptive_avg_pool3d_def.cpp.snippet.cpp（见文末）`): the `ascend310p_config` registers input `x` and output `y` as `.DataType({ge::DT_FLOAT, ge::DT_FLOAT16})` (lines 61, 66) — NO `DT_BF16` — while the non-310p configs in the same file (lines 25, 30, 40, 45) DO carry `{DT_FLOAT, DT_FLOAT16, DT_BF16}`. Line 74 `AddConfig("ascend310p", …)` (+ kirinx90 / kirin9030 aliases, L75-76) binds the fp32/fp16-only surface to this SoC family.
  ```cpp
  OpAICoreConfig ascend310p_config;
  ascend310p_config.Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}); // no DT_BF16
  // ...
  this->AICore().AddConfig("ascend310p", ascend310p_config);
  ```

**Scope / how to recognize the class**: fires for any op ported to the `ascend310p` / dav-m200 SoC config whose upstream source offers BF16 on newer arches (the `__CCE_AICORE__ >= 220` gate). Action: trim the 310p config dtype list to `{fp32, fp16}`; keep BF16 only in the non-310p configs. This is a CONFIG/registration-level fact (the `op_host` `.DataType` declaration), independent of device runtime.

**Corollary — no DataCopyPad on this arch**: the same arch boundary that drops BF16 also lacks `DataCopyPad` (arch < 220), so non-block-aligned GM↔UB transfers must use the pybind-side channel/axis padding convention (P-P66 / EC-23), not an in-kernel `DataCopyPad`. adaptive_avg_pool3d pads the innermost C dimension to `numPerBlock` in pybind (`代码片段 adaptive_avg_pool3d/pybind11.cpp.snippet.cpp（见文末）`) — a 3rd-SoC-family instance of P-P66 (after V351/220x).

**Evidence**: adaptive_avg_pool3d 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — 12/12 PASS across 9 fp32 + 3 fp16 cases (CPU-truth; A3 device ref unreachable). The 310p config ships fp32/fp16-only; BF16 was never exercised. `[first evidence]` — now corroborated by a 2nd dav-m200 op (below); the config-level fact is 2-op-supported across two flavors (config-registration + kernel-source-guard).
  - **ascend_quant_v2 310p (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — 2nd-op confirmation, SOURCE-GUARD flavor**: this quantize-to-int8 op is a NPUKernelBench raw-launch pybind kernel (no `op_host` AICoreConfig), so its BF16 exclusion is at the KERNEL-SOURCE level, not the config-registration level — the 220x upstream guards the BF16 path behind `#if __CCE_AICORE__ == 220` (内部路径（不公开）kernel/ascend_quant_v2_kernels.cpp:8: "Supported dtypes: FP16, FP32 — BF16 NOT supported on 310P"), which is dead on arch 200x, so only FP16 + FP32 compile. SAME arch-boundary mechanism as adaptive_avg_pool3d's config-level drop (the `>= 220` BF16 gate), observed at a different layer — corroborating BF16 is gated out across the 310p SoC surface, not just one op's config. 9/9 fp16 int8-bit-exact vs CPU fp64 truth. Device provenance UNVERIFIED (`back/310p-hw-spec-pages` `device.runtime: NOT_RUN` / FORBIDDEN; `verification.json` `method` claims a device run the branch hardware doc contests).

**Other instances (predicted)**: any pooling / elementwise / normalization op ported to `ascend310p` whose upstream carries a BF16 branch — adaptive_avg_pool2d, max_poolNd, layer-family ops on dav-m200. Each should drop BF16 from the 310p config the same way.

**Cross-ref**: P-P66 / [`EC-23`](ec_23_datacopypad_ub_gm_crashes_on_ascend950pr_507035.md) (pybind padding when `DataCopyPad` is absent on arch < 220 — the alignment corollary), the adv_api `SupportType<T,…>` caveat (CONTRAST: bf16 IS supported on Ascend950PR/A5 primitives — different chip family, no conflict; verify dtype support via the `c310_impl.h` static_assert, not the header docstring), EC-88 / [`EC-194`](ec_194_dav_m200_ascend310p3_arch_200x_the_scalar_subtra.md) (companion arch-surface-reduction entries for dav-m200: missing primitives). backend=ascendc.

## 内嵌代码片段

### adaptive_avg_pool3d/adaptive_avg_pool3d_def.cpp.snippet.cpp

```cpp
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "adaptive_avg_pool3d_apt");
        this->AICore().AddConfig("ascend950", aicore_config_950);

        OpAICoreConfig ascend310p_config;
        ascend310p_config.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        ascend310p_config.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        ascend310p_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend310p", ascend310p_config);
        this->AICore().AddConfig("kirinx90", ascend310p_config);
        this->AICore().AddConfig("kirin9030", ascend310p_config);
```

### adaptive_avg_pool3d/pybind11.cpp.snippet.cpp

```cpp
}  // namespace

torch::Tensor run_adaptive_avg_pool3d(torch::Tensor x,
                                       const std::vector<int64_t> &output_size)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "x must be on NPU");
    TORCH_CHECK(output_size.size() == 3,
                "output_size must have 3 elements (D_out, H_out, W_out), got ",
                output_size.size());

    auto x_c = x.contiguous();
    auto sizes = x_c.sizes().vec();

    // Normalize to 5D NCDHW. If input is 4D CDHW, prepend N=1.
    bool was_4d = false;
    if (sizes.size() == 4) {
        was_4d = true;
        x_c = x_c.unsqueeze(0);
        sizes = x_c.sizes().vec();
    }
    TORCH_CHECK(sizes.size() == 5,
                "x must be 4D (CDHW) or 5D (NCDHW), got ", x.dim(), "D");

    int64_t N    = sizes[0];
    int64_t C    = sizes[1];
    int64_t D_in = sizes[2];
    int64_t H_in = sizes[3];
    int64_t W_in = sizes[4];
    int64_t D_out = output_size[0];
    int64_t H_out = output_size[1];
    int64_t W_out = output_size[2];

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0,
                "output_size must be positive, got (",
                D_out, ",", H_out, ",", W_out, ")");
    TORCH_CHECK(D_in > 0 && H_in > 0 && W_in > 0,
                "input spatial dims must be positive");

    int64_t total_out = N * C * D_out * H_out * W_out;

    // Allocate the exact output shape — the kernel uses DataCopyPad on the
    // unaligned tail of the last AIV block, so no padding/narrow is needed.
    // (P149: pybind is dispatch+alloc only; no host-side alignment cleanup.)
    auto out_shape = std::vector<int64_t>{N, C, D_out, H_out, W_out};
    auto out_5d = torch::empty(out_shape, x_c.options());

    if (total_out == 0) {
        if (was_4d) return out_5d.view({C, D_out, H_out, W_out});
        return out_5d;
```

**孪生卡 ol-639 合并的独有内容**（ol-639-on-dav-m200-ascend310p3-the-registered-compute-d.md 已删除）：

> **Concrete anchor** (adaptive_avg_pool3d 310p, `target/ascendc/examples/adaptive_avg_pool3d/snippets.md`): the `ascend310p_config` registers input `x` and output `y` as `.DataType({ge::DT_FLOAT, ge::DT_FLOAT16})` (lines 61, 66) — NO `DT_BF16` — while the non-310p configs in the same file (lines 25, 30, 40, 45) DO carry `{DT_FLOAT, DT_FLOAT16, DT_BF16}`. Line 74 `AddConfig("ascend310p", …)` (+ kirinx90 / kirin9030 aliases, L75-76) binds the fp32/fp16-only surface to this SoC family.

> ascend310p_config.Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}); // no DT_BF16

> **Corollary — no DataCopyPad on this arch**: the same arch boundary that drops BF16 also lacks `DataCopyPad` (arch < 220), so non-block-aligned GM↔UB transfers must use the pybind-side channel/axis padding convention (P-P66 / EC-23), not an in-kernel `DataCopyPad`. adaptive_avg_pool3d pads the innermost C dimension to `numPerBlock` in pybind (`target/ascendc/examples/adaptive_avg_pool3d/snippets.md`) — a 3rd-SoC-family instance of P-P66 (after V351/220x).

## 适用场景

- **工程形态**：dav-m200 的计算 dtype 范围只有 fp32/fp16（AICoreConfig 不注册 BF16），上游 。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
