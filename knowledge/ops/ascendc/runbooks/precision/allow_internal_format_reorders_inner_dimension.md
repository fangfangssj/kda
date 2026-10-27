---
type: Runbook
title: "allow_internal_format=True reorders within the inner dimension (NZ/fractal) — for element-order-dependent ops it silently swaps the wrong elements; disable it for RoPE/rotation verification"
description: "allow_internal_format=True reorders within the inner dimension (NZ/fractal) — for element-order-dependent ops it silently swaps the wrong elements; disable it for RoPE/rotation verification"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["RoPE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-543 — OL-543: A5Ops additive extension

> 关联：同一 OL-543 主题已由既有卡 [ol_543_allow_internal_format_true_reorders_within_the_i.md](../compilation/ol_543_allow_internal_format_true_reorders_within_the_i.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all (any element-order-dependent op verified via torch_npu on a chip that applies NZ internal format)。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9.0 (aarch64)。

**OL-543** — **`allow_internal_format=True` reorders within the inner dimension (NZ/fractal) — for element-ORDER-dependent ops it silently swaps the wrong elements; disable it for RoPE/rotation verification.** NZ layout reorders D → half-rotation swaps wrong positions (`max_abs_diff ~16`). Decision rule resolves the tension with PB-58: need NZ (cube weights / cast must stick) → set True (PB-58); verifying an order-dependent op → set False (this entry). Set-only attr in torch_npu 2.9.0 (readback raises AttributeError). backend=all. apply_rotary_pos_emb 2026-08-10. Cross-ref [PB-58](../compilation/npu_format_cast_fractal_nz_silent_nd_fallback.md), [PB-106](../compilation/torch_npu_dispatcher_raw_launch_not_stream_ordered.md), [OL-542](../compilation/ol_542_in_place_cann_ops_aclnnapply_mutate_their_inputs.md).

- Former A5Ops ID: `OL-543`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all (any element-order-dependent op verified via torch_npu on a chip that applies NZ internal format)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu 2.9.0 (aarch64)`
`unverified_on: soc=Ascend950PR, Ascend910 (NZ internal-format application is a torch_npu/CANN-layout behavior — likely transfers; re-verify the mismatch magnitude on other chips)`

**Principle**: with `torch_npu.npu.config.allow_internal_format = True`, torch_npu may lay tensors in fractal/NZ format, which **reorders elements within the inner dimension D**. An op whose correctness depends on element ORDER within D — a RoPE half/quarter rotation that pairs and swaps specific positions, any even/odd or strided permutation — then operates on a reordered buffer and **swaps the wrong elements**, producing large mismatches (`max_abs_diff ~16` for large cos/sin values) that look like an algorithm bug.

**Both directions (decision rule — resolves the apparent tension with PB-58)**:
- **You NEED the NZ format** (cube weights that physically live in NZ; a `npu_format_cast(t, 29)` that must stick) → set `allow_internal_format = True` BEFORE the cast. With it `False` (the default) the cast **silently degrades to base/ND** and returns shape-unchanged — see **PB-58**. True is correct *there*.
- **You are VERIFYING an element-order-dependent op** against torch_npu (RoPE rotation, even/odd/strided permutations) → set `allow_internal_format = False`, so torch_npu does not reorder D behind your back and the reference matches the kernel's ND layout. True is wrong *here*.
The two rules are complementary, not contradictory: the question is whether NZ layout is the *desired* state of the tensor (keep True) or an *undesired* reordering of a verify-reference (force False).

**Concrete anchor**:
