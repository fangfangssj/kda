---
type: Runbook
title: "CANN `ARPESmall` (`apply_rotary_pos_emb_small.h`) silently ignores `rotary_mode` — always half-mode rotation"
description: "CANN `ARPESmall` (`apply_rotary_pos_emb_small.h`) silently ignores `rotary_mode` — always half-mode rotation"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["ARPESmall"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-110 — PB-110: A5Ops additive extension

> 关联：同一 PB-110 主题已由既有卡 [pb_110_cann_arpesmall_apply_rotary_pos_emb_small_h_sile.md](pb_110_cann_arpesmall_apply_rotary_pos_emb_small_h_sile.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3；CANN 9.0.0；op_class=rotary-position-embedding。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0；torch_npu 2.9.0（aarch64）。

**PB-110** — **CANN `ARPESmall` (`apply_rotary_pos_emb_small.h`) silently ignores `rotary_mode` — always half-mode rotation.** Single `ComputeTotary`, no mode dispatch; the attr is accepted but unused. Quarter passes by coincidence (≡ half for D∈{64,128}, the only supported head dims); interleave genuinely diverges (`~10` diff). Port fix: always half-mode regardless of `rotary_mode`. Verified by CANN interleave output ≈ half formula (`<0.002`). backend=all. apply_rotary_pos_emb 2026-08-10. Cross-ref [OL-120](../precision/ol_120_fp32_internal_single_cast_rint.md), [OL-187](../precision/ol_187_port_a3_emit_all_tiling_dispatch_variants.md), OL-542（A5Ops 绑定外部引用；本语料未收录目标）.

- Former A5Ops ID: `PB-110`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=rotary-position-embedding`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9.0 (aarch64)`
`unverified_on: soc=Ascend950PR, Ascend910 (other CANN rotary implementations may dispatch on mode — re-verify before assuming the same silent fallback)`

**Symptom**: precision cases for interleave-mode RoPE fail (`max_abs_diff ~10`) while half-mode and quarter-mode cases pass, against a CANN reference built from `apply_rotary_pos_emb_small.h`.

**Mechanism**: the 220x-family `ARPESmall` kernel has a single `ComputeTotary` that implements **half-mode** rotation (`out = x*cos + rotate_half(x)*sin`). There is **no mode dispatch in the kernel**. The `rotary_mode` attribute is accepted by the API surface but **silently ignored** — CANN always applies the half-mode formula regardless of the requested mode.

**Key insight (why quarter passes, interleave fails)**: quarter-mode rotation is **mathematically identical** to half-mode for `D ∈ {64, 128}` — the only head dims supported on this chip — so quarter cases pass *by coincidence*, not because the mode is honored. Interleave-mode rotation is a genuinely different formula and fails outright.

**Fix for a port**: always compute **half-mode** rotation in the kernel, regardless of the `rotary_mode` parameter the API accepts. Do not wire a mode-dispatch branch that CANN's own reference does not honor.

**Verification (worker)**: calling CANN with `rotary_mode='interleave'` and comparing its output against the half-mode formula yields `diff < 0.002` (fp16 rounding) — i.e. CANN's "interleave" output IS the half-mode output, confirming the mode attr is ignored. (Mechanism per worker observation of the CANN kernel + API behavior on this box; the falsifiable claim is the `< 0.002` interleave≈half diff.)

**Evidence**: apply_rotary_pos_emb iter 2 (2026-08-10, Ascend310P3, CANN 9.0.0) — interleave cases `~10` diff; root-caused to single-formula `ARPESmall`; kernel switched to always-`ComputeHalf`, mode dispatch removed. Failures-ledger: `iter 2 | mid(~10) | novel`.

**Cross-ref**: OL-120 (fp32-promote matching for RoPE-family references), OL-187 (port_a3 variant dispatch — the host-side axis where mode *would* dispatch if it were honored; here it isn't), OL-542 (the in-place capture pitfall on the same op).
