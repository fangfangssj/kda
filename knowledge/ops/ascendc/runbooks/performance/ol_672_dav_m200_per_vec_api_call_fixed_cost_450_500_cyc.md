---
type: Runbook
title: "dav-m200 per-VEC-API-call fixed cost ≈ 450-500 cycles, length-independent"
description: "dav-m200 每次 VEC API 调用固定开销约 450-500 周期且与长度无关——行密小 shape 是调用次数瓶颈不是带宽瓶颈，要少而长的 V 调用"
tags: [task.optimization, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-672"]
---

# dav-m200 per-VEC-API-call fixed cost ≈ 450-500 cycles, length-independent

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; op_class=row-dense data-movement shapes (pad/tile/copy)`
**OL-672 — dav-m200 per-VEC-API-call fixed cost ≈ 450-500 cycles, length-independent** — row-dense pad shapes are V-CALL-COUNT bound, not bandwidth bound (15_Pad ko-8 measured decomposition: case 8 = 1 Adds/row × 1207 rows/core = 378us; vendor PadV3 reaches ~60cyc/row on the same shapes). Remedy: fewer/longer V calls + geometry-driven tile sizing. Cross-ref CAND-DAV-M200-VEC-CALL-OVERHEAD. backend=ascendc.
