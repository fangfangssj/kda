---
type: Runbook
title: "220x Level-2 `And`/`Or`/`Xor` on `LocalTensor<int32_t>` compile clean but process `count/2` elements (silent corruption)"
description: "220x Level-2 `And`/`Or`/`Xor` on `LocalTensor<int32_t>` compile clean but process `count/2` elements (silent corruption)"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-90"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-90 — PB-90: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend910 (220x)；CANN 9.0.0；bisheng=a3 toolchain；op_class=任何在 int32（或更宽整数）数据上使用 Level-2 bitwise ops 的 220x kernel。
- 验证来源：6_QuantMatmul a3 移植（2026-08-01，编译干净、静默 count/2 损坏）。

**PB-90** — 220x Level-2 `And`/`Or`/`Xor` on `LocalTensor<int32_t>` compile clean but process `count/2` elements (dav_c220 bitwise is int16/uint16-lane ONLY; Level-2 lacks the Level-0 static_assert) — SILENT corruption; fix = `ReinterpretCast<int16_t>` + `2×count` or the Level-0 form.

- Former A5Ops ID: `PB-90`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend910 (220x); cann=9.0.0; bisheng=a3 toolchain; op_class=any 220x kernel using Level-2 bitwise ops on int32 (or wider integer) data`
`verified_on: 6_QuantMatmul a3 port (2026-08-01, compile-clean, silent count/2 corruption)`

- **Symptom**: Level-2 bitwise ops on `LocalTensor<int32_t>` COMPILE CLEAN but process only `count/2` elements — silent precision corruption, no error anywhere. The dav_c220 bitwise instruction is int16/uint16-lane ONLY; the Level-2 form lacks the Level-0 form's `static_assert` that would reject the wider type at compile time.
- **Fix**: `ReinterpretCast<int16_t>` + `2×count` (process the int32 payload as two int16 lanes), or use the Level-0 form whose `static_assert` rejects non-int16 at compile time.
- **Evidence**: 6_QuantMatmul a3 port (2026-08-01, worker compile-clean + silent halved-element count; the same class as PB-37's silent uint8 garbage but in the bitwise lane).
- **Transfer**: any Level-2 bitwise op on integer data wider than 16 bits on 220x.

---<a id="pb-100"></a>
