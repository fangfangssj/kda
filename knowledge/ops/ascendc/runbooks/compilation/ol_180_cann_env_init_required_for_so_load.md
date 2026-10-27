---
type: Runbook
title: "AscendC kernel .so loading requires CANN environment initialization"
description: "Before torch.ops.load_library(.so), CANN env must be initialized (source set_env.sh or set LD_LIBRARY_PATH/ASCEND_HOME_PATH), else .so load fails with undefined-symbol errors."
tags: [task.debugging, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-180-cann-env-init-required-for-so-load.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: [a3]
aliases: ["OL-180"]
---

# AscendC kernel .so loading requires CANN environment initialization

> 原 OKF v1 卡号：**OL-180**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - torch.ops.load_library(<kernel>.so) fails with undefined symbol errors when the CANN environment was not initialized in the process first

## 现象 / 触发
`torch.ops.load_library(<kernel>.so)` fails with undefined-symbol errors because the CANN runtime environment was never initialized in the process before the load.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
Loading an AscendC kernel `.so` requires the CANN environment to be live first: source `set_env.sh`, or configure `LD_LIBRARY_PATH` / `ASCEND_HOME_PATH` so the dynamic loader can resolve the CANN runtime symbols the kernel references. Without this, the `.so` load fails with undefined-symbol errors (the missing symbols are the CANN runtime, not the kernel's own code).

Verified on soc=Ascend910_9382 (V220), cann=8.3.RC1. Evidence: 25_NLLLoss a3-ds kw-5 (2026-05-21) — `.so` load failed until the CANN env was properly initialized.
