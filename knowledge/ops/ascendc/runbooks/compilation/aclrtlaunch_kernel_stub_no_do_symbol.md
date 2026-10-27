---
type: Runbook
title: "CANN 9.1.0 ascendc cmake generates `aclrtlaunch_<kernel>` host stubs returning `uint32_t` — no `<kernel>_do` symbol exists"
description: "CANN 9.1.0 ascendc cmake generates `aclrtlaunch_<kernel>` host stubs returning `uint32_t` — no `<kernel>_do` symbol exists"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["aclrtlaunch"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-214 — EC-214: A5Ops additive extension

> 关联：同一 EC-214 主题已由既有卡 [ec_214_on_the_cann_9_1_0_ascendc_cmake_flow_the_generat.md](ec_214_on_the_cann_9_1_0_ascendc_cmake_flow_the_generat.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.1.0；op_class=all（standalone-pybind / 验收通道中 launch kernel 的宿主 TU）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（sort 310p，2026-08-18，build iter 3）。

**EC-214** — **on the CANN 9.1.0 ascendc cmake flow the generated host launch stub is `aclrtlaunch_<kernel>` returning `uint32_t` — no `<kernel>_do` symbol exists.** Symptom is a LINK failure on the launch symbol, not compile; prototype lives at `build/include/kernels/aclrtlaunch_<kernel>.h`; rc is TORCH_CHECK-able. sort 310p 2026-08-18 (build iter 3). Cross-ref [EC-10](ec_10_aclrtlaunch_undefined_reference_linker.md), [PB-112](pb_112_select_with_selmode_vsel_cmpmask_spr_applies_the.md), CAND-BENCH-STANDALONE-KERNEL-PACKAGING. backend=ascendc.

- Former A5Ops ID: `EC-214`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; bisheng=n/a; op_class=all (standalone-pybind / acceptance-lane host TUs that launch the kernel)`
`unverified_on: soc=Ascend950PR, Ascend910 on cann=9.1.0 (stub naming is a per-cmake-flow generated artifact — grep the generated header before assuming either convention)`

- **Symptom**: link failure (`undefined reference to <kernel>_do`) or a missing-prototype compile error when the pybind/host TU calls the `<kernel>_do` launch convention carried by templates from other CANN versions/flows.
- **Fix**: on this flow the ascendc cmake generates `aclrtlaunch_<kernel>` host stubs returning a `uint32_t` return code — declare `extern "C" uint32_t aclrtlaunch_<kernel>(uint32_t blockDim, void* stream, ...)` (the generated prototype lives at `build/include/kernels/aclrtlaunch_<kernel>.h` under the kernel build tree) and `TORCH_CHECK(rc == 0)` the result (the PB-112 pre-launch-check discipline applies to the launch rc too).
- **Detection**: a host TU that built against another CANN flow fails at LINK, not compile, on the launch symbol → generated-stub naming drift, not a code bug; read the generated `build/include/kernels/` header for the actual symbol.
- **Evidence**: sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 3): `pybind11.cpp` written against the `<kernel>_do` convention failed at link; switching to `aclrtlaunch_<kernel>` + rc `TORCH_CHECK` fixed; kernel shipped 26/26 bit-exact.
- **Cross-ref**: EC-10 (aclrtlaunch undefined reference — the declaration-form half), PB-112 (pre-launch device sync + rc checks), CAND-BENCH-STANDALONE-KERNEL-PACKAGING (the standalone cmake flow this stub comes from). backend=ascendc.
