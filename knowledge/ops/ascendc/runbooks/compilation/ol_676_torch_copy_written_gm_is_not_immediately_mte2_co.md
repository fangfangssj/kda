---
type: Runbook
title: "torch copy_-written GM is NOT immediately MTE2-consumable by AI kernels on this stack"
description: "torch `copy_` 写的 GM 不能立刻被同流 kernel 的 MTE2 读消费（run 间 flap）；kernel→kernel 生产链没这个问题。"
tags: [task.debugging, topic.data_movement.data_copy]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-676"]
---

# torch copy_-written GM is NOT immediately MTE2-consumable by AI kernels on this stack

`applies_to: soc=Ascend310P3; cann=9.1.0; op_class=any torch-write->kernel-read boundary (out.copy_ then AI-kernel MTE2 read)`
**OL-676 — torch copy_-written GM is NOT immediately MTE2-consumable by AI kernels on this stack** — `out.copy_(x)` followed (same stream) by a kernel whose first act is MTE2 reads of `out` flaps between runs (21_Scatter fp32 delta cases 24/41/46). Kernel→kernel producer chains (widen kernel writes → delta kernel reads) are stable at the same boundary. Stage through an own kernel or a fresh buffer. backend=ascendc.
