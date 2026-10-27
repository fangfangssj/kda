---
type: Runbook
title: "msprof pipe-ratio metrics (PipeUtilization op-summary / sample-based ai_core_utilization) are UNAVAILABLE for raw-launch (aclrtBinary) kernels on this box — all-zero rows"
description: "raw-launch kernel 在 msprof 里 pipe-ratio 指标全零不可用——改用 Task_Duration 分解"
tags: [task.optimization, topic.performance.analysis]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-678"]
---

# msprof pipe-ratio metrics (PipeUtilization op-summary / sample-based ai_core_utilization) are UNAVAILABLE for raw-launch (aclrtBinary) kernels on this box — all-zero rows

`applies_to: soc=Ascend310P3; cann=9.1.0; op_class=raw-launch (aclrtBinary) kernel perf analysis`
**OL-678 — msprof pipe-ratio metrics (PipeUtilization op-summary / sample-based ai_core_utilization) are UNAVAILABLE for raw-launch (aclrtBinary) kernels on this box — all-zero rows** — use Task_Duration decomposition + arithmetic instead. 15_Pad ko-8. backend=ascendc.
