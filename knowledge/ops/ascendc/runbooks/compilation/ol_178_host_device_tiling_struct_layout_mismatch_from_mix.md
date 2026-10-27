---
type: Runbook
title: "Host↔device tiling struct layout mismatch from mixed-width fields — use explicit packing"
description: "applies_to: soc=all; cann=all; bisheng=all; op_class=all_data_movement"
tags: [task.debugging, topic.programming_model.execution_unit.cube, topic.data_movement, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-178-host-device-tiling-struct-layout-mismatch-from-mix.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3]
aliases: ["OL-178"]
---

# Host↔device tiling struct layout mismatch from mixed-width fields — use explicit packing

> 原 OKF v1 卡号：**OL-178**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=all; cann=all; bisheng=all; op_class=all_data_movement

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=all; cann=all; bisheng=all; op_class=all_data_movement`
`verified_on: soc=Ascend910_9382; cann=9.0.0; bisheng=15.0.5`

**Principle**: When a tiling struct (shared between host GCC and device bisheng) contains mixed-width integer fields (int32_t alongside int64_t), the two compilers may insert different padding, leading to field misalignment.

**Fix**: Wrap the tiling struct in `#pragma pack(push, 1)` / `#pragma pack(pop)` in the kernel header.

**Evidence**: 12_Permute a3-ds kw-3 iter-3 (2026-05-21, Ascend910_9382 V220). `#pragma pack(1)` fixed stride array garbage reads.

**Cross-ref**: OL-77 (GM tiling struct field-by-field read — kernel-side rule); EC-40 (Cube tiling host POD size mismatch).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-178（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
