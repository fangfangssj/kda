---
type: Runbook
title: "CANN ARPESmall (apply_rotary_pos_emb_small.h) silently ignores rotary_mode — always half-mode rotation."
description: "CANN ARPESmall（apply_rotary_pos_emb_small.h）静默忽略 rotary_mode，永远做半量旋转——无 mode 分发"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-110"]
---

# CANN ARPESmall (apply_rotary_pos_emb_small.h) silently ignores rotary_mode — always half-mode rotation.

**PB-110** — **CANN `ARPESmall` (`apply_rotary_pos_emb_small.h`) silently ignores `rotary_mode` — always half-mode rotation.** Single `ComputeTotary`, no mode dispatch; the attr is accepted but unused. Quarter passes by coincidence (≡ half for D∈{64,128}, the only supported head dims); interleave genuinely diverges (`~10` diff). Port fix: always half-mode regardless of `rotary_mode`. Verified by CANN interleave output ≈ half formula (`<0.002`). backend=all. apply_rotary_pos_emb 2026-08-10. Cross-ref [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md), [`OL-187`](../precision/ol_187_port_a3_emit_all_tiling_dispatch_variants.md), [`OL-542`](ol_542_in_place_cann_ops_aclnnapply_mutate_their_inputs.md).

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

**Cross-ref**: [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md) (fp32-promote matching for RoPE-family references), [`OL-187`](../precision/ol_187_port_a3_emit_all_tiling_dispatch_variants.md) (port_a3 variant dispatch — the host-side axis where mode *would* dispatch if it were honored; here it isn't), [`OL-542`](ol_542_in_place_cann_ops_aclnnapply_mutate_their_inputs.md) (the in-place capture pitfall on the same op).
