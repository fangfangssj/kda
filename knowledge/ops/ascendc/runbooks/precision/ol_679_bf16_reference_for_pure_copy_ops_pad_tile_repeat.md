---
type: Runbook
title: "bf16 reference for PURE-COPY ops (pad/tile/repeat/broadcast) can legally run the NPU fp32-roundtrip"
description: "纯拷贝类算子（pad/tile/repeat）的 bf16 参考可以合法走 NPU fp32 往返（bf16→fp32 无损，实测位一致）；算术类不适用"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-679"]
---

# bf16 reference for PURE-COPY ops (pad/tile/repeat/broadcast) can legally run the NPU fp32-roundtrip

`applies_to: soc=Ascend310P3; cann=9.1.0; op_class=pure-copy ops (pad/tile/repeat/broadcast) on bf16-less vendor kernels (OL-68 Case A)`
**OL-679 — bf16 reference for PURE-COPY ops (pad/tile/repeat/broadcast) can legally run the NPU fp32-roundtrip** (`x.to(fp32) → vendor-fp32 → to(bf16)`) when the vendor kernel lacks bf16 (OL-68 Case A) — probed BIT-EXACT vs the CPU-fallback path (10/10, 15_Pad 2026-09-01: pad is pure copy, bf16→fp32 lossless, fp32→bf16 exact-rounds the bf16 source). Removes the H2D+CPU+D2H inflation that made bf16 ratios N/A/inflated and drops wall acceptance to the honest number. NOT for arithmetic ops — probe bit-exactness first. backend=all.
