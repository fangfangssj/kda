---
type: Runbook
title: "dav-m200 vconv accepts only a per-direction cast-mode whitelist (h→f `CAST_NONE`; f→h `CAST_NONE`/`CAST_ODD`)"
description: "dav-m200 vconv accepts only a per-direction cast-mode whitelist (h→f `CAST_NONE`; f→h `CAST_NONE`/`CAST_ODD`)"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["RoundMode::CAST_RINT"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-120 — PB-120: A5Ops additive extension

> 关联：同一 PB-120 主题已由既有卡 [pb_120_dav_m200_vconv_accepts_only_a_per_direction_cast.md](pb_120_dav_m200_vconv_accepts_only_a_per_direction_cast.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__CCE_AICORE__=200）；CANN 9.0.0；op_class=all。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（17_AdamW 设备运行，2026-08-15）。

**PB-120** — **dav-m200 vconv accepts only a per-direction cast-mode whitelist**: h→f `CAST_NONE` only; f→h `CAST_NONE`/`CAST_ODD`; `CAST_RINT` compiles but hits `ASCENDC_ASSERT(false)` → the cast NEVER EXECUTES → stale/uninit fp16 outputs (widened paths clean, no error raised — grep `RoundMode::CAST_RINT` to detect). f32→f16 `CAST_NONE` is RNE at normal magnitudes (0 diffs/1M vs CPU RNE) and SATURATES NaN/overflow→±65504 (== torch_npu `.to(fp16)`; canonical-grader-forgiven). PROMOTES OL-81's 2026-08-09 hedge — its stated condition (device run exercising CAST_RINT on dav-m200) met by 17_AdamW 310p 2026-08-15: 18/18 OVERALL_T1, 13/18 bit-exact, det 21/21, perf 0.7408x. CAST_RINT default is VOID on arch-200x half↔float. Cross-ref [OL-81](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md), [OL-120](../precision/ol_120_fp32_internal_single_cast_rint.md) (carries no dav-m200 caveat yet — flagged for owner), PB-107, EC-197, PB-105. backend=ascendc.

- Former A5Ops ID: `PB-120`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (17_AdamW device run, 2026-08-15)`
`unverified_on: soc=Ascend910_220x (a3), Ascend950PR (a5) — OL-81's CAST_RINT default remains the guidance there; do not import this restriction to those SoCs`

**Symptom**: fp16-path outputs read stale/uninitialized UB (narrowed outputs alternating `[0x0000, stale]`) while every WIDENED path (fp32/bf16) is clean; no error is raised — the narrowing cast silently did not happen. Grep a dav-m200 kernel for `RoundMode::CAST_RINT` on a half↔float cast: each hit is this bug.

**Mechanism**: the dav-m200 vconv implementation supports ONLY `CAST_NONE` for half→float and `CAST_NONE`/`CAST_ODD` for float→half (CANN 安装树内 asc/impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h:77-95,217-252). `CAST_RINT` is not in the whitelist; the impl reaches `ASCENDC_ASSERT(false)`, which in the release build skips the conversion rather than aborting — source and destination UB keep prior contents. Same dav-m200 SDK-surface-narrower-than-newer-CANN class as EC-197/PB-105.

**Numeric semantics of the supported modes** (probed on device, 17_AdamW):
- `CAST_NONE` f32→f16 behaves as RNE at normal magnitudes — 0 differing elements / 1M vs a CPU RNE cast; only overflow-boundary elements differ, because vconv SATURATES: NaN→65504, overflow→±65504. torch_npu's own `.to(fp16)` does the identical thing (probed); CPU torch casts overflow→±inf. Saturation-vs-inf is the canonical grader's documented forgiveness class (`src/scripts/orchestrator/precision/cannbench_grader/compare.py:373`).
- This RNE-at-normal-magnitudes probe PARTIALLY resolves the truncation-vs-RNE contest flagged in OL-81's layer_norm_quant row, for the fp32→fp16 pair; the fp16→int8 quant pair remains open.

**Fix**: use `CAST_NONE` for both directions on dav-m200 (内部路径（不公开）workspace/17_AdamW/kernel/adamw_kernel.h:190-272, 6 sites). This entry PROMOTES the 2026-08-09 UNVERIFIED hedge in OL-81 to a confirmed constraint — its stated promotion condition ("a provenance-resolved run actually exercises `RoundMode::CAST_RINT` for fp32↔fp16 on dav-m200 and reproduces the assert/garbage") is now met: 17_AdamW shipped 6× CAST_RINT, device-observed the uninit-UB symptom, and the CAST_NONE switch cleaned the fp16 path.

**Evidence**: 17_AdamW 310p (2026-08-15, benchmark, `torch_npu.npu_apply_adam` ref): precision iter-1 fp16 outputs stale/uninit with fp32/bf16 clean (内部路径（不公开）workspace/17_AdamW/failures_ledger.md:12); 6× CAST_RINT→CAST_NONE → pass_a 18/18 OVERALL_T1 (beats-CANN 18/18, 13/18 bit-exact), pass_b 645/645 T1 (605/645 bit-exact), det 21/21, perf 0.7408x mean device-event (`workspace/17_AdamW/pass_a_summary.json`, `pass_b_summary.json`, `perf_summary.json`). Device run — `device:1` recorded in all three summaries (note: `hardware/target/ascend310p.md` still carries the stale pre-device `device.runtime: NOT_RUN` flag; the run artifacts supersede it here). Prior art: inplace_add_layer_norm 2026-08-09 (the unverified hedge, OL-81); upstream-fidelity CAST_NONE choices — ge_glu_grad_v2 / mse_loss_v2 / rms_norm_grad 310p rows in OL-81.

**Cross-ref**: OL-81 (CAST_RINT default — now per-SoC split; dav-m200 takes CAST_NONE), OL-120 (fp32-internal + single CAST_RINT end-cast — the same per-SoC caveat applies on dav-m200; FLAGGED for owner: OL-120 itself carries no dav-m200 caveat line yet), PB-107 (fixpipe-flavor ±65504 saturation sibling), EC-197 / PB-105 (dav-m200 arch-surface gap class). backend=ascendc.
