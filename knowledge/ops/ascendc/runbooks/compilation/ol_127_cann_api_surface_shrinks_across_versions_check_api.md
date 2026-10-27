---
type: Runbook
title: "CANN API surface shrinks across versions — check API availability before using A5-KB patterns on older CANN [V220, CANN 9.0.0]"
description: "CANN API surface shrinks across versions — check API availability before using A5-KB patterns on older CANN [V220, CANN 9.0.0]"
tags: [task.debugging, topic.data_movement.data_copy, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-127-cann-api-surface-shrinks-across-versions-check-api.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3, "950"]
aliases: ["OL-127"]
---

# CANN API surface shrinks across versions — check API availability before using A5-KB patterns on older CANN [V220, CANN 9.0.0]

> 原 OKF v1 卡号：**OL-127**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - paradigm: ascendc

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

**Principle**: CANN 9.0.0 on V220 has a smaller API surface than CANN 8.x on A5. Several APIs commonly used in A5 KB patterns and code templates do not exist on V220. Workers porting from A5 patterns MUST verify each non-trivial API against the V220 header set before committing to it in Phase B.

**Concrete anchor — APIs absent from V220 CANN 9.0.0**:

| Absent API | Replacement |
|---|---|
| `DataCopyPad(dst, src, DataCopyParams)` + `DataCopyParams` struct | `DataCopy(dst, src, count)` — simple 3-arg form only |
| `PadNone` / `padNone` enum | Does not exist. Align buffers to datablock size (fp32:8, fp16:16) |
| `aclrtLaunchKernelWithHostArgs` (bisheng auto-gen host stub) | Manual `extern "C" uint32_t aclrtlaunch_<name>(uint32_t blockDim, void* stream, ...)` stubs |
| `GM_ADDR` type alias | `__gm__ uint8_t*` (kernel entry), `void*` (host stub) |
| `Divs` (tensor / scalar division) | Pre-compute reciprocal in pybind, use `Muls` in kernel |

**Worker build checklist for V220**: before completing Phase B, grep kernel code for `DataCopyPad`, `DataCopyParams`, `PadNone`, `GM_ADDR`, `aclrtLaunchKernelWithHostArgs`, `Divs`.

**Evidence**: op#31 IOU DS cold-start kw-1 (2026-05-01): 12 build iterations stuck on DataCopyPad. op#17 AdamW (2026-05-02): Divs compile error.

**Other instances (predicted)**: any A5→V220 port using strided DataCopy, bisheng auto-gen host stubs, pad-mode control, or scalar division via Divs.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-127（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
