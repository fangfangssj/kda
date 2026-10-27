---
type: Runbook
title: "allow_internal_format=True reorders within the inner dimension (NZ/fractal) — for element-ORDER-dependent ops"
description: "allow_internal_format=true 会在迭代内重排格式——涉及精度比对时需注意其数值影响"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.cube]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-543"]
---

# allow_internal_format=True reorders within the inner dimension (NZ/fractal) — for element-ORDER-dependent ops

**OL-543** — **`allow_internal_format=True` reorders within the inner dimension (NZ/fractal) — for element-ORDER-dependent ops it silently swaps the wrong elements; disable it for RoPE/rotation verification.** NZ layout reorders D → half-rotation swaps wrong positions (`max_abs_diff ~16`). Decision rule resolves the tension with PB-58: need NZ (cube weights / cast must stick) → set True (PB-58); verifying an order-dependent op → set False (this entry). Set-only attr in torch_npu 2.9.0 (readback raises AttributeError). backend=all. apply_rotary_pos_emb 2026-08-10. Cross-ref PB-58, [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md), [`OL-542`](ol_542_in_place_cann_ops_aclnnapply_mutate_their_inputs.md).

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

**孪生卡 ol-647 合并的独有内容**（ol-647-allow-internal-format-true-reorders-within-the-i.md 已删除）：

> # For RoPE / rotation verification: pin the reference to ND layout

> torch_npu.npu.config.allow_internal_format = False # before any reference call

> Quirk (torch_npu 2.9.0): the attribute is **set-only** — assigning `False` succeeds, but reading it back raises `AttributeError`. Do not use a readback as the assertion; just set it.

> **Evidence**: apply_rotary_pos_emb (2026-08-10, Ascend310P3, CANN 9.0.0) — RoPE output differed (`max_abs_diff ~16` for large cos/sin) between `allow_internal_format=True` and `False`; NZ layout had reordered elements within D, making the half-rotation swap wrong positions. `False` matches the kernel.

> **Other instances (predicted)**: any element-order-dependent op verified through torch_npu on a chip/format that applies NZ internal layout — rotations, even/odd interleaves, strided permutations; any capture harness that flips `allow_internal_format` globally and then verifies order-sensitive ops.

> **Cross-ref**: PB-58 (the complement — cast silently degrades to ND when internal formats disabled; set True when you NEED NZ), PB-60 (another torch_npu-build ordering hazard on this box), [`OL-542`](ol_542_in_place_cann_ops_aclnnapply_mutate_their_inputs.md) (in-place capture pitfall on the same op).

## 适用场景

- **工程形态**：allow_internal_format=true 会在迭代内重排格式。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
