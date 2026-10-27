---
type: Runbook
title: "CANN merge_mix_obj.sh Crash with Empty --build-type"
description: "CANN merge_mix_obj.sh Crash with Empty --build-type"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-7-cann-merge-mix-obj-sh-crash-with-empty-build-type.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["PB-7"]
---

# CANN merge_mix_obj.sh Crash with Empty --build-type

> 原 OKF v1 卡号：**PB-7**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - make fails at 95% with Error 1 in merge_mix_obj.sh — shift 2 fails

- **Symptom**: `make` fails at 95% with `Error 1` in `merge_mix_obj.sh` — `shift 2` fails
- **Root cause**: `cmake` invokes `merge_mix_obj.sh --build-type` without a value when `CMAKE_BUILD_TYPE` is unset. The bash `shift 2` fails because only 1 arg remains.
- **Affected**: CANN 9.0.0, AIV-only kernels (AIC dir empty, merge step still runs)
- **Workaround**: Always set `-DCMAKE_BUILD_TYPE=Release` in cmake invocation
- **Status**: OPEN (CANN build system bug)
- **Evidence**: MXFP4 project (2026-04-07), `merge_mix_obj.sh` line `shift 2` on `--build-type`

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Build Integration Issues

<!-- 迁移自 porter kb/target/ascendc/（PB-7，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
