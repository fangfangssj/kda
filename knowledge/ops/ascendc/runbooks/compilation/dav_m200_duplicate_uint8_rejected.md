---
type: Runbook
title: "`Duplicate<uint8_t>` is rejected on dav-m200 (SDK dtype whitelist) — zero-fill via a `ReinterpretCast<int16_t>` view"
description: "`Duplicate<uint8_t>` is rejected on dav-m200 (SDK dtype whitelist) — zero-fill via a `ReinterpretCast<int16_t>` view"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-108 — PB-108: A5Ops additive extension

> 关联：同一 PB-108 主题已由既有卡 [pb_108_duplicate_uint8_t_is_rejected_on_dav_m200_sdk_dt.md](pb_108_duplicate_uint8_t_is_rejected_on_dav_m200_sdk_dt.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__CCE_AICORE__=200）；CANN 9.0.0；op_class=all（API 面白名单）。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（logical_not 310p，2026-08-09 — 编译期 SDK static_assert）。

**PB-108** — `Duplicate<uint8_t>` is REJECTED on dav-m200 (SDK dtype whitelist: half/int16/uint16/int32/uint32/float only) — zero-fill a u8 buffer via its `ReinterpretCast<int16_t>` view (0x0000 = two zero bytes; constant `v` fills as `v

- Former A5Ops ID: `PB-108`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a (API-surface whitelist); op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (logical_not 310p, 2026-08-09 — SDK static_assert at compile time)`
`unverified_on: soc=Ascend310B, Ascend910, Ascend950PR (Duplicate dtype matrices differ per SoC — check the per-SoC impl header before relying)`

- **Severity**: LOW — compile-time static_assert (fails loud, not silent); costs one build iter if not known in advance.
- **Symptom**: `Duplicate(LocalTensor<uint8_t>, ...)` fails to compile on dav-m200 with an SDK static_assert — the m200 `Duplicate` implementation supports **half / int16 / uint16 / int32 / uint32 / float only**; uint8 (and int8) are not in the whitelist.
- **Mechanism**: per-SoC dtype whitelists in the SDK's vector-compute impl headers; same gap class as PB-105 (DataCopyPad absent) and the CAND-DAV-M200-VCONV matrix (P-P203) — dav-m200 narrows several VEC API dtype surfaces relative to 220x/351x.
- **Fix**: zero-fill a uint8 buffer through its 2-byte view — `buf.ReinterpretCast<int16_t>()` then `Duplicate<int16_t>(view, 0, count/2)` (0x0000 = two zero bytes). Any constant-fill of a u8 buffer whose per-byte value is `v` fills the int16 view with `v | (v<<8)`.
- **Evidence**: logical_not 310p (2026-08-09, port_a3_to_a5 TARGET=310p) — compile iter 1 hit the static_assert at 内部路径（不公开）workspace/logical_not/kernel/logical_not_kernel.h:134 (workspace scratch, gitignored); the int16-view zero-fill shipped in iter 2 → 34/34 bit-exact vs A3 capture, det 34/34.
- **Other instances (predicted)**: any dav-m200 kernel zero-initializing or constant-filling a u8/int8 UB staging buffer (predicate masks, packed bool lanes, byte-domain scatter staging). Cross-ref [PB-105](dav_m200_datacopypad_unsupported_both_directions.md) (m200 missing-API class), P-P203 (per-SoC vconv matrix — same lookup-before-coding discipline), OL-80/OL-130 (grep the SDK headers first).
