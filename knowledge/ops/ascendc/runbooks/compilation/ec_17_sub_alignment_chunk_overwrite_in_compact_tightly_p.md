---
type: Runbook
title: "Sub-alignment chunk overwrite in compact (tightly-packed) output"
description: "Sub-alignment chunk overwrite in compact (tightly-packed) output"
tags: [task.debugging, topic.precision, topic.data_movement.data_copy, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-17-sub-alignment-chunk-overwrite-in-compact-tightly-p.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-17"]
---

# Sub-alignment chunk overwrite in compact (tightly-packed) output

> 原 OKF v1 卡号：**EC-17**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Precision failures when chunk_size < DataCopy alignment AND output elements are tightly packed (no gaps between chunks from different outer iterations)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Precision failures when chunk_size < DataCopy alignment AND output elements are tightly packed (no gaps between chunks from different outer iterations)
- **Root cause**: DataCopy writes aligned count of elements. When chunk < align, excess elements overwrite the next chunk's data. In Cat's output (strided with gaps), overlapping tail write works. In Split's output (compact), there are no gaps — adjacent chunks are immediately adjacent.
- **Fix (host-side)**: Detect `chunk < align && outer > 1`. Use `nblk=1` (serial execution — overwrites self-correct within one block) + allocate padded output + narrow to exact size.
- **Applicability**: Any kernel writing to compact output with non-aligned chunk boundaries
- **Evidence**: Split V1 failed 4/57 cases, fixed in V2 (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-17，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
