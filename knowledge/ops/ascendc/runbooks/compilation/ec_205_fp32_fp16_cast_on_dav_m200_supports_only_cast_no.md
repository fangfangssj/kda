---
type: Runbook
title: "fp32→fp16 Cast on dav-m200 supports ONLY CAST_NONE/CAST_ODD"
description: "dav-m200 fp32→fp16 Cast 只支持 CAST_NONE/CAST_ODD，其他舍入模式编译能过但运行时 ASCENDC_ASSERT 静默不写输出；CAST_NONE 即 RNE，等价 torch `.half()`。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-205"]
---

# fp32→fp16 Cast on dav-m200 supports ONLY CAST_NONE/CAST_ODD

**EC-205** — **fp32→fp16 `Cast` on dav-m200 supports ONLY `CAST_NONE`/`CAST_ODD`** — `CAST_RINT`/`FLOOR`/`CEIL`/`ROUND`/`TRUNC` ASCENDC_ASSERT-trap mid-kernel, leaving the output unwritten (silent; compiles clean). Fix: `CAST_NONE` IS RNE narrowing = PyTorch `.half()`; read per-SoC round-mode support from the per-SoC impl header ([`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) chain). Latent while the harness edge set is fp32-only. gelu 310p 2026-08-12 (fp16 bulk path broken both modes → CAST_NONE fix → acceptance 40/40). Cross-ref [`PB-115`](pb_115_torch_npu_lazy_host_task_queue_vs_raw_aclrt_laun.md), pows Cast-mode-per-SoC split. [dav-m200/Ascend310P3]

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12 — fp16 benchmark cases exposed it)`

- **Symptom**: kernel with an f32→f16 `Cast(..., RoundMode::CAST_RINT, ...)` compiles clean, then the launch traps inside ASCENDC_ASSERT — output tensor never written (caller sees recycled/stale memory; combine with PB-115's queue race and the failure looks data-dependent).
- **Root cause**: the dav-m200 vconv f32→f16 impl only maps `CAST_NONE` (RNE narrowing) and `CAST_ODD`; every other round mode is an assert path (CANN 安装树内 impl/dav_m200/kernel_operator_vec_vconv_impl.h).
- **Fix**: use `CAST_NONE` — it IS round-to-nearest-even narrowing, matching PyTorch `.half()` semantics. `CAST_RINT` is NOT the safe "default" on dav-m200 despite being the PyTorch-alignment default elsewhere; per-SoC round-mode support must be read from the per-SoC impl header (OL-80 lookup chain), and note the pows precedent went the OTHER way (mirror-the-source: the upstream 200-branch used CAST_NONE where non-200 used CAST_RINT — OL Cast-mode-per-SoC split).
- **Latency note**: latent while only fp32 cases are verified (the Cast sits on the fp16 out path); any op whose harness edge set is fp32-only carries this silently until the first fp16 case.
- **Evidence**: gelu 310p (2026-08-12, failures_ledger precision rework iter): fp16 bulk path broken for BOTH gelu modes down to n=64 single-core; CAST_NONE fix → fp16 paths bit-stable, acceptance 40/40.

- **Evidence (cross-op confirmation)**: reduce_sum 310p (2026-08-17): vendor torch_npu fp16 sum = fp32 accumulate + RNE narrow; matching it with fp32 accumulation + `Cast(CAST_NONE)` gave bit-exact results (`max_abs_diff = 0.0`) on all 16 fp16 benchmark cases — 37/37 tier-1 PASS overall (`reduce_sum/verification.json`, workspace scratch, gitignored).

## 适用场景

- **工程形态**：dav-m200 fp32→fp16 Cast 只支持 CAST_NONE/CAST_ODD，其他舍入模式编译能过但运行。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
