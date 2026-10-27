---
type: Runbook
title: "half is a GLOBAL typedef, NOT AscendC::half"
description: "`half` 是全局 typedef 而非 `AscendC::half`；入口 .cpp 全局作用域写 `AscendC::half` 会编译失败。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-212"]
---

# half is a GLOBAL typedef, NOT AscendC::half

**EC-212** — **`half` is a GLOBAL typedef, NOT `AscendC::half`** — qualifying it at global scope in the entry .cpp (template alias / extern-C signature) is a compile error (`'half' is not a member of 'AscendC'`); kernel .h files never notice because they `using namespace AscendC` and write unqualified `half`. Fix: unqualified `half` at global scope; grep the SDK header ([`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) chain) before assuming any name is `AscendC::`. Same namespace-surface family as [`OL-544`](../performance/ol_544_fp16_nz_format_cube_matmul_is_usable_on_ascend31.md)'s global-ns `GetNormalConfig`. layer_norm_v3 310p 2026-08-17 (build iter 1 → green; 54/54 Pass A). Cross-ref [`OL-544`](../performance/ol_544_fp16_nz_format_cube_matmul_is_usable_on_ascend31.md), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=CANN 9.0.0 toolchain; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (layer_norm_v3 310p, 2026-08-17 — build iter 1)`
`unverified_on: soc=Ascend910 (220x) / Ascend950PR (351x) — SDK-header namespace fact, expected arch-general (kernel code everywhere uses unqualified `half`), but confirmed only on the dav-m200 CANN 9.0.0 install`

- **Symptom**: compile error of the form `error: 'half' is not a member of 'AscendC'` when a kernel entry .cpp / pybind-adjacent TU qualifies the type at GLOBAL scope — typically in a template alias, an extern-C wrapper signature, or a `using` declaration.
- **Root cause**: the AscendC SDK headers define `half` as a GLOBAL typedef (like `int64_t`), not inside `namespace AscendC`. Kernel .h files never notice because they `using namespace AscendC;` and write unqualified `half`; the failure only appears in a TU that tries to be explicit. Same namespace-surface family as OL-544's `GetNormalConfig` being global-ns (not `matmul::`) on the same install.
- **Fix**: write unqualified `half` at global scope; qualify only types that actually live in `AscendC::` (`AscendC::LocalTensor<T>`, etc.). When in doubt about which namespace a name lives in, grep the SDK header (OL-80 lookup chain) rather than assuming everything is `AscendC::`.
- **Evidence**: layer_norm_v3 310p (2026-08-17, benchmark-acceptance lane, failures_ledger build iter 1): single qualification fix → clean build; downstream 54/54 Pass A on-device.
- **Cross-ref**: [`OL-544`](../performance/ol_544_fp16_nz_format_cube_matmul_is_usable_on_ascend31.md) (dav-m200 namespace gotchas), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (SDK-header lookup chain). backend=ascendc.
