---
type: Runbook
title: "on the CANN 9.1.0 ascendc cmake flow the generated host launch stub is aclrtlaunch_<kernel> returning uint32_t"
description: "CANN 9.1.0 cmake 流程生成的 host launch stub 叫 `aclrtlaunch_<kernel>` 返回 uint32_t，没有 `<kernel>_do` 符号——症状是链接失败而非编译失败。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-214"]
---

# on the CANN 9.1.0 ascendc cmake flow the generated host launch stub is aclrtlaunch_<kernel> returning uint32_t

**EC-214** — **on the CANN 9.1.0 ascendc cmake flow the generated host launch stub is `aclrtlaunch_<kernel>` returning `uint32_t` — no `<kernel>_do` symbol exists.** Symptom is a LINK failure on the launch symbol, not compile; prototype lives at `build/include/kernels/aclrtlaunch_<kernel>.h`; rc is TORCH_CHECK-able. sort 310p 2026-08-18 (build iter 3). Cross-ref [`EC-10`](ec_10_aclrtlaunch_undefined_reference_linker.md), [`PB-112`](pb_112_select_with_selmode_vsel_cmpmask_spr_applies_the.md),. backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; bisheng=n/a; op_class=all (standalone-pybind / acceptance-lane host TUs that launch the kernel)`
`unverified_on: soc=Ascend950PR, Ascend910 on cann=9.1.0 (stub naming is a per-cmake-flow generated artifact — grep the generated header before assuming either convention)`

- **Symptom**: link failure (`undefined reference to <kernel>_do`) or a missing-prototype compile error when the pybind/host TU calls the `<kernel>_do` launch convention carried by templates from other CANN versions/flows.
- **Fix**: on this flow the ascendc cmake generates `aclrtlaunch_<kernel>` host stubs returning a `uint32_t` return code — declare `extern "C" uint32_t aclrtlaunch_<kernel>(uint32_t blockDim, void* stream, ...)` (the generated prototype lives at `build/include/kernels/aclrtlaunch_<kernel>.h` under the kernel build tree) and `TORCH_CHECK(rc == 0)` the result (the PB-112 pre-launch-check discipline applies to the launch rc too).
- **Detection**: a host TU that built against another CANN flow fails at LINK, not compile, on the launch symbol → generated-stub naming drift, not a code bug; read the generated `build/include/kernels/` header for the actual symbol.
- **Evidence**: sort 310p (2026-08-18, benchmark-acceptance lane, failures_ledger build iter 3): `pybind11.cpp` written against the `<kernel>_do` convention failed at link; switching to `aclrtlaunch_<kernel>` + rc `TORCH_CHECK` fixed; kernel shipped 26/26 bit-exact.
- **Cross-ref**: [`EC-10`](ec_10_aclrtlaunch_undefined_reference_linker.md) (aclrtlaunch undefined reference — the declaration-form half), [`PB-112`](pb_112_select_with_selmode_vsel_cmpmask_spr_applies_the.md) (pre-launch device sync + rc checks), (the standalone cmake flow this stub comes from). backend=ascendc.
