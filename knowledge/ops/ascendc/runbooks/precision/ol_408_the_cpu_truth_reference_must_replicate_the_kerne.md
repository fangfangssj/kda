---
type: Runbook
title: "the CPU-truth reference must replicate the kernel's out-of-range index policy: scatter/index kernels that SKIP"
description: "scatter/index 类 kernel 跳过越界索引时，CPU 参考必须同样 mask，绝不能 clamp——clamp 会把越界索引映射到合法位置造成假精度错"
tags: [task.debugging, topic.data_movement.data_copy, topic.compilation, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-408"]
---

# the CPU-truth reference must replicate the kernel's out-of-range index policy: scatter/index kernels that SKIP

**OL-408** — **the CPU-truth reference must replicate the kernel's out-of-range index policy: scatter/index kernels that SKIP OOB indices require the reference to MASK them, never clamp.** `.clamp(0,N-1)` remaps OOB indices onto a valid position the kernel never writes → precision mismatch on large-index cases (~1e17 from int32/int64 edge data). Fix = validity mask. Rule = *match the kernel* (skip-or-clamp, either is valid if both sides agree). Anchor `scatter_elements_v2/verification.json`. Evidence: scatter_elements_v2 310p iter-2, 9/9 T1 after fix. backend=ascendc.

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=scatter/index (scatter_elements, gather_elements, index_put, index_select, index_fill)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3; cann=9.0.0 (scatter_elements_v2 310p 2026-08-08)`
`unverified_on: other SoCs — the principle (reference must match kernel OOB policy) is arch-agnostic; the specific skip-vs-clamp policy is op/source-specific and must be read from each kernel.`
