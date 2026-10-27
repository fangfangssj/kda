---
type: Runbook
title: "`Adds` dtype set is NARROWER than `Duplicate`'s on dav-m200 — no uint32 (compile-time static_assert); per-API dtype whitelists differ WITHIN one SoC, so byte-identity bridges on u32-carried patterns must use the int32 view"
description: "dav-m200 `Adds` 只支持 half/float/int16/int32（同 SoC 的 Duplicate 却支持 uint32）——每个原语的 dtype 白名单独立，别类推。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-126", "Adds", "Duplicate"]
---

# `Adds` dtype set is NARROWER than `Duplicate`'s on dav-m200 — no uint32 (compile-time static_assert); per-API dtype whitelists differ WITHIN one SoC, so byte-identity bridges on u32-carried patterns must use the int32 view

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a (API-surface whitelist); op_class=all (any VEC identity-bridge / constant-fill on packed-byte buffers)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (22_HybridAttentionMaskPreparation kw-1 iter 4, 2026-09-05 — compile iter hit the static_assert; int32-view bridge shipped, 50/50 bit-exact device run)`
`unverified_on: soc=Ascend950PR, Ascend910_V220, Ascend310B (check each API's per-SoC impl header separately — see Mechanism)`

- **Severity**: LOW — compile-time static_assert (fails loud, not silent); costs one build iter if not known in advance.
- **Symptom**: `Adds(LocalTensor<uint32_t>, ..., count)` fails to compile on dav-m200 — the m200 `Adds` SupportType is **half / float / int16 / int32 ONLY**, while `Duplicate` on the SAME SoC supports uint32 (PB-62's whitelist: half/int16/uint16/int32/uint32/float).
- **Mechanism**: per-SoC dtype whitelists live in each API's OWN impl header — they differ PER API on the same SoC, not just per SoC. You cannot infer one VEC API's dtype surface from a sibling's: PB-62 documents `Duplicate<uint8_t>` rejected; this entry documents `Adds<uint32_t>` rejected while `Duplicate<uint32_t>` is the canonical u32-lane filler. Before first use of a VEC op on an untried dtype, grep that op's SupportType table in the per-SoC impl headers under CANN 安装树内 impl/basic_api/dav_m200/ (the OL-80 lookup chain's step-2.5 extension).
- **Fix**: byte-identity bridges on u32-carried byte patterns use the int32 view — `Adds(outT.ReinterpretCast<int32_t>(), inD.ReinterpretCast<int32_t>(), 0, n>>2)` (integer add-with-zero preserves every bit pattern; immune to the float-domain bridge's `-0.0` flip per the vision_attention 310p caveat in PB-9's evidence). Anchor: `22_HybridAttentionMaskPreparation/kernel/hybrid_mask_kernel.h:169-172 [snippet-unavailable: source workspace not on this box]`.
- **Evidence**: 22_HybridAttentionMaskPreparation kw-1 iter 4 (2026-09-05, Ascend310P3/dav-m200, benchmark): compile failure logged as "Adds has no uint32 on dav-m200 (SupportType: half/float/int16/int32); fix = int32-view identity bridge" (`22_HybridAttentionMaskPreparation/failures_ledger.md:10`); shipped kernel 50/50 pass_a/pass_b bit-exact, det 50/50 (`22_HybridAttentionMaskPreparation/verification.json`). Device run; PB-74 precedent on the stale `device.runtime: NOT_RUN` hardware-page flag (the dtype whitelist itself is a compile/header-level fact and holds regardless).
- **Cross-ref**: PB-62 (Duplicate<uint8_t> — the sibling per-API whitelist gap + the `v|(v<<8)` fill pattern), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md)/PB-69/`PB-70`（310B 卡，本批未迁移）/PB-74 (dav-m200 SDK-surface-narrower class), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (API lookup chain). backend=ascendc.

## 适用场景

- **工程形态**：dav-m200 `Adds` 只支持 half/float/int16/int32（同 SoC 的 Duplicate 却支持 uint32）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
