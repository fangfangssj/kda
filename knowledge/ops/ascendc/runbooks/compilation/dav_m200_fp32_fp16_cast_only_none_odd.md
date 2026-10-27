---
type: Runbook
title: "fp32→fp16 `Cast` on dav-m200 supports only `CAST_NONE`/`CAST_ODD` — other round modes assert-trap mid-kernel leaving the output unwritten"
description: "fp32→fp16 `Cast` on dav-m200 supports only `CAST_NONE`/`CAST_ODD` — other round modes assert-trap mid-kernel leaving the output unwritten"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-205 — EC-205: A5Ops additive extension

> 关联：同一 EC-205 主题已由既有卡 [ec_205_fp32_fp16_cast_on_dav_m200_supports_only_cast_no.md](ec_205_fp32_fp16_cast_on_dav_m200_supports_only_cast_no.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__CCE_AICORE__=200）；CANN 9.0.0；op_class=all。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（gelu 310p，2026-08-12，fp16 benchmark 用例暴露；reduce_sum 310p 2026-08-17 跨 op 确认见正文）。

**EC-205** — **fp32→fp16 `Cast` on dav-m200 supports ONLY `CAST_NONE`/`CAST_ODD`** — `CAST_RINT`/`FLOOR`/`CEIL`/`ROUND`/`TRUNC` ASCENDC_ASSERT-trap mid-kernel, leaving the output unwritten (silent; compiles clean). Fix: `CAST_NONE` IS RNE narrowing = PyTorch `.half()`; read per-SoC round-mode support from the per-SoC impl header (OL-80 chain). Latent while the harness edge set is fp32-only. gelu 310p 2026-08-12 (fp16 bulk path broken both modes → CAST_NONE fix → acceptance 40/40). Cross-ref [PB-115](torch_npu_host_task_queue_overwrites_raw_launch_output.md), pows Cast-mode-per-SoC split. [dav-m200/Ascend310P3]

- Former A5Ops ID: `EC-205`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12 — fp16 benchmark cases exposed it)`

- **Symptom**: kernel with an f32→f16 `Cast(..., RoundMode::CAST_RINT, ...)` compiles clean, then the launch traps inside ASCENDC_ASSERT — output tensor never written (caller sees recycled/stale memory; combine with PB-115's queue race and the failure looks data-dependent).
- **Root cause**: the dav-m200 vconv f32→f16 impl only maps `CAST_NONE` (RNE narrowing) and `CAST_ODD`; every other round mode is an assert path (CANN 安装树内 impl/dav_m200/kernel_operator_vec_vconv_impl.h).
- **Fix**: use `CAST_NONE` — it IS round-to-nearest-even narrowing, matching PyTorch `.half()` semantics. `CAST_RINT` is NOT the safe "default" on dav-m200 despite being the PyTorch-alignment default elsewhere; per-SoC round-mode support must be read from the per-SoC impl header (OL-80 lookup chain), and note the pows precedent went the OTHER way (mirror-the-source: the upstream 200-branch used CAST_NONE where non-200 used CAST_RINT — OL Cast-mode-per-SoC split).
- **Latency note**: latent while only fp32 cases are verified (the Cast sits on the fp16 out path); any op whose harness edge set is fp32-only carries this silently until the first fp16 case.
- **Evidence**: gelu 310p (2026-08-12, failures_ledger precision rework iter): fp16 bulk path broken for BOTH gelu modes down to n=64 single-core; CAST_NONE fix → fp16 paths bit-stable, acceptance 40/40.

- **Evidence (cross-op confirmation)**: reduce_sum 310p (2026-08-17): vendor torch_npu fp16 sum = fp32 accumulate + RNE narrow; matching it with fp32 accumulation + `Cast(CAST_NONE)` gave bit-exact results (`max_abs_diff = 0.0`) on all 16 fp16 benchmark cases — 37/37 tier-1 PASS overall (`workspace/reduce_sum/verification.json`, workspace scratch, gitignored).
