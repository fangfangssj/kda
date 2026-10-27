---
type: Runbook
title: "dav-m200 unaligned-dst VEC ops are LEGAL (arbitrary off + non-AE count)"
description: "dav-m200 上 VEC 算子的非对齐 dst + 非整 AE count 是合法的（51/51+177/177 实测）——32B 约束是 DataCopy 的约束被错误移植到了 V 管线"
tags: [task.optimization, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-669"]
---

# dav-m200 unaligned-dst VEC ops are LEGAL (arbitrary off + non-AE count)

## 优化点

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; op_class=any data-movement kernel staging into UB with unaligned offsets or non-AE counts`
**OL-669 — dav-m200 unaligned-dst VEC ops are LEGAL (arbitrary off + non-AE count)** — the head/bulk/tail alignment split (and any "32B constraint" applied to the V pipe) is a DataCopy constraint transplanted where it was never required. 15_Pad ko-8 (2026-09-01): `Adds(buf[off], src[skew], 0, n)` with unaligned off and non-AE n verified CORRECT on 51/51 + 177/177 (every unaligned geometry). ko-2's "structural ceiling" argument was this transplant. Cross-ref OL-251/255, CAND-DAV-M200-BORDER-FILL-INTERIOR-OVERWRITE. backend=ascendc.
