---
type: Runbook
title: "aten CPU-fallback scatter-reduce chain table on 310p: fp16/bf16 are fp32-shadow + ONE final RNE round (add AND multiply); fp32 is exact native; reduce=None is ascending last-wins"
description: "310p 上 aten CPU-fallback 的 scatter-reduce 链实测表：fp16/bf16 是 fp32 影子+单次 RNE，fp32 精确原生，reduce=None 是升序后写覆盖——以被评分的参考为准，别从兄弟算子继承"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compute.reduction]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-671"]
---

# aten CPU-fallback scatter-reduce chain table on 310p: fp16/bf16 are fp32-shadow + ONE final RNE round (add AND multiply); fp32 is exact native; reduce=None is ascending last-wins

`applies_to: soc=Ascend310P3; cann=9.0.0; op_class=scatter_(reduce) via aten CPU fallback`
**OL-671 — aten CPU-fallback scatter-reduce chain table on 310p: fp16/bf16 are fp32-shadow + ONE final RNE round (add AND multiply); fp32 is exact native; reduce=None is ascending last-wins** — measured per OL-545 probe-first (21_Scatter probes/cpu_chain_probe*.json, 16/16 per dtype×reduce). DIFFERS from ops grading against an aclnn NPU chain (19_IndexPut bf16 = native-chained): the GRADED reference determines the chain — probe-first beats cousin-inheritance. backend=ascendc.
