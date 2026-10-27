---
type: Runbook
title: "`ShiftLeft` / `ShiftRight` are not supported on dav-m200 and fail silently (compile clean, run as a no-op)"
description: "`ShiftLeft` / `ShiftRight` are not supported on dav-m200 and fail silently (compile clean, run as a no-op)"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["ShiftLeft", "ShiftRight"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-113 — PB-113: A5Ops additive extension

> 关联：同一 PB-113 主题已由既有卡 [pb_113_shiftleft_shiftright_are_not_supported_on_dav_m2.md](pb_113_shiftleft_shiftright_are_not_supported_on_dav_m2.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__CCE_AICORE__=200）；CANN 9.0.0；op_class=all（SDK 实现缺口）。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（gelu 310p，2026-08-12 — on-device 观测）。

**PB-113** — `ShiftLeft` / `ShiftRight` are NOT SUPPORTED on dav-m200 and fail SILENTLY (`ASCENDC_REPORT_NOT_SUPPORT` in CANN 安装树内 impl/dav_m200/kernel_operator_vec_binary_scalar_impl.h compiles clean, runs as a no-op) — a `(k+127)<<23` 2^k reconstruction produced the UN-shifted value. Fix: int `Muls(t, t, 1<<s)` (vmuls s32 supported; overflow-check

- Former A5Ops ID: `PB-113`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a (SDK impl gap); op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12 — on-device observation)`
`unverified_on: soc=Ascend310B, Ascend910, Ascend950PR (ShiftLeft works on 220x/351x — the gap is dav-m200-SDK-specific; on m200, distrust ANY op whose impl is `ASCENDC_REPORT_NOT_SUPPORT`)`

- **Severity**: HIGH — the worst failure mode: compiles clean, no runtime error, the shift simply does not happen.
- **Symptom**: `ShiftLeft(dst, src, shiftCount, count)` / `ShiftRight` leave `dst` holding the UN-shifted values. Observed: a `2^k` bit reconstruction `(k+127)<<23` produced `(k+127)` unshifted (127 interpreted as fp32 bits ≈ 1.78e-43) — downstream math collapsed (in gelu: erfc≡0, output w∈{0,2}).
- **Mechanism**: the dav-m200 SDK binary-scalar impl header marks the shift path `ASCENDC_REPORT_NOT_SUPPORT(false, "ShiftLeft")` (CANN 安装树内 impl/dav_m200/kernel_operator_vec_binary_scalar_impl.h) — on this SDK that marker compiles and runs as a SILENT no-op rather than erroring. Same trap class as the known s32→f32 Cast CAST_NONE-only quirk: an op whose dav-m200 impl is `ASCENDC_REPORT_NOT_SUPPORT` cannot be trusted to fail loudly.
- **Fix**: replace `ShiftLeft(t, t, s, count)` with an int `Muls(t, t, 1<<s, count)` — vmuls s32 IS supported (overflow check: operands must satisfy |x|·2^s ≤ INT32_MAX; for fp32 exponent construction k+127 ≤ 253, so 253·2^23 < INT32_MAX holds). For right shifts there is no exact multiply form — reformulate (Cast scaling or a different bit trick) instead of emulating.
- **Evidence**: gelu 310p (2026-08-12) — failures_ledger precision iter 1: max_abs_diff_large traced to the un-shifted `2^k` reconstruction; the int-Muls fix shipped in the same chain → 29/29 Tier-1 PASS, det 29/29 bit-identical. Kernel anchor 内部路径（不公开）workspace/gelu/kernel/gelu_kernel.h:205 (workspace scratch, gitignored).
- **Other instances (predicted)**: any dav-m200 kernel doing fp32 bit manipulation (2^k exponent construction, mantissa extract/insert, magic-number float tricks). Note P-P200's `2^k` recipe already prefers `Adds(+127)→Muls(2^23)→Cast` for an independent reason (PB-23 int32 binary-scalar arithmetic) — that form is also the m200-safe one. Cross-ref [PB-105](dav_m200_datacopypad_unsupported_both_directions.md)/[PB-108](dav_m200_duplicate_uint8_rejected.md)/[PB-112](pb_112_select_with_selmode_vsel_cmpmask_spr_applies_the.md) (same dav-m200 `ASCENDC_REPORT_NOT_SUPPORT` class), PB-23, P-P200, OL-80 (lookup-chain extension: grep the per-SoC impl header before first use).
