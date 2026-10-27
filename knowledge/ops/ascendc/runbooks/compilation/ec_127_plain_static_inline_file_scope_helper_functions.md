---
type: Runbook
title: "plain `static inline` file-scope helper functions called from `__aicore__` context are dropped by the device link (ld.lld) — use SDK-form `__aicore__ inline` (non-static), with a host-side empty-`#define` fallback when the same header is also host-compiled"
description: "kernel 头文件里的 `static inline` 文件作用域函数在设备侧 ld.lld 链接时丢失——设备共享逻辑要用 `__aicore__ inline` 成员函数或模板。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-127"]
---

# plain `static inline` file-scope helper functions called from `__aicore__` context are dropped by the device link (ld.lld) — use SDK-form `__aicore__ inline` (non-static), with a host-side empty-`#define` fallback when the same header is also host-compiled

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=ccec (CANN 9.0.0 device toolchain); op_class=all (any kernel TU carrying a file-scope C++ helper library header)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (26_MoeGroupScoreAggregationAndMasking build iter 1, 2026-09)`
`unverified_on: soc=Ascend950PR / Ascend910 bisheng (the `__aicore__ inline` form is the portable SDK shape regardless; the silent-drop at link is a ccec/ld.lld observation on dav-m200)`

- **Symptom**: a kernel header factorizes shared device logic into plain `static inline` file-scope functions (the libstdc++/host-C++ idiom) and calls them from `__aicore__` code; the host compile is clean but the DEVICE link (`ld.lld`) drops/fails to resolve those functions — build failure at the device-object link step.
- **Fix**: declare the helpers in SDK form — `__aicore__ inline` (NON-static). When the same header is also compiled by host g++ (host-side battery/test harness), guard the qualifier with an empty define on the host side, e.g. host prelude `#define __aicore__` (empty) so the header compiles in both worlds.
- **Evidence**: 26_MoeGroupScoreAggregationAndMasking 310p kw (2026-09, failures_ledger build iter 1): `kernel/moe_group_lib.h` selection-algorithm helpers (`move_median_to_first`, partition/insertion scans) initially `static inline`; device ld.lld dropped them → converted to `__aicore__ inline` with the host-side empty-`#define` fallback (the header is shared with `tests/test_moe_lib_host.cpp`) → build PASS at iter 2 (SOC=Ascend310P3). Workspace scratch, gitignored: `26_MoeGroupScoreAggregationAndMasking/kernel/moe_group_lib.h [snippet-unavailable: source workspace not on this box]`.
- **Other instances (predicted)**: any op that factorizes device code into a shared C++ helper header (selection/sort libraries, fixed-point helpers, bit-manipulation routines) — especially ports from host C++ where `static inline` is the default idiom.
- **Cross-ref**: [`EC-35`](ec_35_aiv_device_binary_does_not_link_libm_transcendenta.md) (the sibling device-link trap — libm names undefined at the AIV-vector `ld.lld` step; same "host-clean, device-link-fails" shape, different symbol class). backend=ascendc.
