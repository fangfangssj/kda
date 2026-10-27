---
type: Runbook
title: "`half` is a global typedef, not `AscendC::half` — qualifying it at global scope is a compile error"
description: "`half` is a global typedef, not `AscendC::half` — qualifying it at global scope is a compile error"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["error: 'half' is not a member of 'AscendC'", "AscendC::half"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-212 — EC-212: A5Ops additive extension

> 关联：同一 EC-212 主题已由既有卡 [ec_212_half_is_a_global_typedef_not_ascendc_half.md](ec_212_half_is_a_global_typedef_not_ascendc_half.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；bisheng=CANN 9.0.0 toolchain；op_class=all。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（layer_norm_v3 310p，2026-08-17，build iter 1；修复后 54/54 Pass A）。

**EC-212** — **`half` is a GLOBAL typedef, NOT `AscendC::half`** — qualifying it at global scope in the entry .cpp (template alias / extern-C signature) is a compile error (`'half' is not a member of 'AscendC'`); kernel .h files never notice because they `using namespace AscendC` and write unqualified `half`. Fix: unqualified `half` at global scope; grep the SDK header (OL-80 chain) before assuming any name is `AscendC::`. Same namespace-surface family as OL-544's global-ns `GetNormalConfig`. layer_norm_v3 310p 2026-08-17 (build iter 1 → green; 54/54 Pass A). Cross-ref OL-544（A5Ops 绑定外部引用；本语料未收录目标）, [OL-80](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). backend=ascendc.

- Former A5Ops ID: `EC-212`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=CANN 9.0.0 toolchain; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (layer_norm_v3 310p, 2026-08-17 — build iter 1)`
`unverified_on: soc=Ascend910 (220x) / Ascend950PR (351x) — SDK-header namespace fact, expected arch-general (kernel code everywhere uses unqualified `half`), but confirmed only on the dav-m200 CANN 9.0.0 install`

- **Symptom**: compile error of the form `error: 'half' is not a member of 'AscendC'` when a kernel entry .cpp / pybind-adjacent TU qualifies the type at GLOBAL scope — typically in a template alias, an extern-C wrapper signature, or a `using` declaration.
- **Root cause**: the AscendC SDK headers define `half` as a GLOBAL typedef (like `int64_t`), not inside `namespace AscendC`. Kernel .h files never notice because they `using namespace AscendC;` and write unqualified `half`; the failure only appears in a TU that tries to be explicit. Same namespace-surface family as OL-544's `GetNormalConfig` being global-ns (not `matmul::`) on the same install.
- **Fix**: write unqualified `half` at global scope; qualify only types that actually live in `AscendC::` (`AscendC::LocalTensor<T>`, etc.). When in doubt about which namespace a name lives in, grep the SDK header (OL-80 lookup chain) rather than assuming everything is `AscendC::`.
- **Evidence**: layer_norm_v3 310p (2026-08-17, benchmark-acceptance lane, failures_ledger build iter 1): single qualification fix → clean build; downstream 54/54 Pass A on-device.
- **Cross-ref**: OL-544 (dav-m200 namespace gotchas), OL-80 (SDK-header lookup chain). backend=ascendc.
