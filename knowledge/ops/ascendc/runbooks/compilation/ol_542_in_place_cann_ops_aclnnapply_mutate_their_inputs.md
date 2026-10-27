---
type: Runbook
title: "in-place CANN ops (aclnnApply) mutate their inputs through the torch_npu wrapper"
description: "aclnnApply 系 in-place 算子会修改输入 tensor——评测/对比前不知情会污染基准数据"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-542"]
---

# in-place CANN ops (aclnnApply) mutate their inputs through the torch_npu wrapper

**OL-542** — **in-place CANN ops (`aclnnApply*`) mutate their inputs through the torch_npu wrapper** — capture/verify with fresh tensor copies, never warm up on the to-be-captured tensors. A warm-up call corrupts the shared q/k objects before the captured call → internally-consistent capture that mismatches the kernel (`max_abs_diff > 100`); looks like a kernel bug, is a capture-harness bug. Detection: single-call vs warm-up-then-capture diff. backend=all. apply_rotary_pos_emb 2026-08-10. Cross-ref [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md), [`OL-187`](../precision/ol_187_port_a3_emit_all_tiling_dispatch_variants.md), [`PB-110`](pb_110_cann_arpesmall_apply_rotary_pos_emb_small_h_sile.md).

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all (any op verified against a torch_npu wrapper of an in-place aclnn op)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9.0 (aarch64)`
`unverified_on: soc=Ascend950PR, Ascend910 (the in-place contract is an aclnn-level convention — likely transfers to any chip running the same CANN op; treat as suspect until measured)`

**Principle**: the CANN op behind `torch_npu.npu_apply_rotary_pos_emb` (and the wider `aclnnApply*` family) is **in-place** — it writes its result back into the q/k input tensors. The torch_npu wrapper forwards the *same* tensor objects the caller holds, so ANY call on a tensor mutates it for every subsequent observer. A capture/verification script that warms up or times the CANN reference with the SAME tensor objects it then feeds to the captured call is measuring corrupted inputs — the captured output looks internally consistent but mismatches the kernel by a large margin (observed `max_abs_diff > 100`), sending triage into the kernel when the bug is in the capture harness.

**Detection**: compare a single-call CANN output (no warm-up) against a warm-up-then-capture output on the SAME tensor objects. If they differ, the inputs were mutated in place by the warm-up call. (Equivalently: CANN reference and kernel agree when each is given a pristine input and disagree when CANN is given a warm-up-corrupted input.)

**Fix (decision rule — applies to ANY op whose reference is an in-place CANN/torch_npu op)**: give the CANN reference call a **fresh clone of every input** on every invocation; never warm up or reuse the tensor objects that will be captured. This is purely a capture-side discipline — the kernel side is unaffected.

**Concrete anchor** (the load-bearing line is the per-call clone):

**孪生卡 ol-646 合并的独有内容**（ol-646-in-place-cann-ops-aclnnapply-mutate-their-inputs.md 已删除）：

> # WRONG: warm-up reuses q_ref/k_ref, mutating them before the captured call

> _ = torch_npu.npu_apply_rotary_pos_emb(q_ref, k_ref, cos, sin, ...) # mutates q_ref/k_ref

> captured = torch_npu.npu_apply_rotary_pos_emb(q_ref, k_ref, cos, sin, ...) # now corrupted

> captured = torch_npu.npu_apply_rotary_pos_emb(q_ref.clone(), k_ref.clone(), cos, sin, ...)

> **Evidence**: apply_rotary_pos_emb iter 1 (2026-08-10, Ascend310P3, CANN 9.0.0) — A3 capture `max_abs_diff > 100` against the kernel despite a correct algorithm; root-caused to in-place mutation of q/k by the warm-up call; re-capture with fresh copies per call resolved it. Failures-ledger: `iter 1 | large(>100) | novel`.

> **Other instances (predicted)**: every op verified against a torch_npu wrapper of an in-place aclnn op — the `aclnnApply*` family (rotary, optimizer-step/apply ops), and any torch_npu op whose impl is in-place; any harness reusing reference input tensors across a warm-up + timed measurement.

> **Cross-ref**: [`OL-120`](../precision/ol_120_fp32_internal_single_cast_rint.md) (fp32-promote matching for RoPE-family references — the precision axis; this entry is the capture-hygiene axis), [`OL-187`](../precision/ol_187_port_a3_emit_all_tiling_dispatch_variants.md) (variant dispatch on the A3 wrapper), PB-64 (the ARPESmall `rotary_mode` pitfall surfaced on the same op).

## 补充证据（2026-10-10 合并自 `inplace_cann_ops_mutate_inputs_fresh_copies.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-542 — OL-542: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all (any op verified against a torch_npu wrapper of an in-place aclnn op)。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9.0 (aarch64)。

**OL-542** — **in-place CANN ops (`aclnnApply*`) mutate their inputs through the torch_npu wrapper** — capture/verify with fresh tensor copies, never warm up on the to-be-captured tensors. A warm-up call corrupts the shared q/k objects before the captured call → internally-consistent capture that mismatches the kernel (`max_abs_diff > 100`); looks like a kernel bug, is a capture-harness bug. Detection: single-call vs warm-up-then-capture diff. backend=all. apply_rotary_pos_emb 2026-08-10. Cross-ref [OL-120](../precision/ol_120_fp32_internal_single_cast_rint.md), [OL-187](../precision/ol_187_port_a3_emit_all_tiling_dispatch_variants.md), [PB-110](cann_arpesmall_ignores_rotary_mode_half_only.md).

- Former A5Ops ID: `OL-542`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all (any op verified against a torch_npu wrapper of an in-place aclnn op)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9.0 (aarch64)`
`unverified_on: soc=Ascend950PR, Ascend910 (the in-place contract is an aclnn-level convention — likely transfers to any chip running the same CANN op; treat as suspect until measured)`

**Principle**: the CANN op behind `torch_npu.npu_apply_rotary_pos_emb` (and the wider `aclnnApply*` family) is **in-place** — it writes its result back into the q/k input tensors. The torch_npu wrapper forwards the *same* tensor objects the caller holds, so ANY call on a tensor mutates it for every subsequent observer. A capture/verification script that warms up or times the CANN reference with the SAME tensor objects it then feeds to the captured call is measuring corrupted inputs — the captured output looks internally consistent but mismatches the kernel by a large margin (observed `max_abs_diff > 100`), sending triage into the kernel when the bug is in the capture harness.

**Detection**: compare a single-call CANN output (no warm-up) against a warm-up-then-capture output on the SAME tensor objects. If they differ, the inputs were mutated in place by the warm-up call. (Equivalently: CANN reference and kernel agree when each is given a pristine input and disagree when CANN is given a warm-up-corrupted input.)

**Fix (decision rule — applies to ANY op whose reference is an in-place CANN/torch_npu op)**: give the CANN reference call a **fresh clone of every input** on every invocation; never warm up or reuse the tensor objects that will be captured. This is purely a capture-side discipline — the kernel side is unaffected.

**Concrete anchor** (the load-bearing line is the per-call clone):
