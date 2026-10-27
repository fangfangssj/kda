---
type: Runbook
title: "CANN typed kernel entry _fp32 crashes with error 507035"
description: "CANN 9.0.T501's auto-generated typed kernel entry points (e.g., `aclrtlaunch_pooling_forward_kernel_fp32`) crash at launch time with error code 507035. The legacy untyped entry points (e.g., `aclrtlau"
tags: [task.debugging, topic.performance.analysis, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-12-cann-typed-kernel-entry-fp32-crashes-with-error-50.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-12"]
---

# CANN typed kernel entry _fp32 crashes with error 507035

> 原 OKF v1 卡号：**OL-12**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when registering kernel entry points for fp32 kernels on CANN 9.0.T501

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
CANN 9.0.T501's auto-generated typed kernel entry points (e.g., `aclrtlaunch_pooling_forward_kernel_fp32`) crash at launch time with error code 507035. The legacy untyped entry points (e.g., `aclrtlaunch_pooling_forward_kernel`) work correctly for the same kernel. The `_fp16` and `_bf16` typed entries are unaffected. This is a CANN runtime bug specific to the `_fp32` suffix. Workaround: use legacy (untyped) entry points for all fp32 kernels. May be fixed in CANN 9.0.0.beta.1.

## 证据
docs/guides/A5_CONTAINER_SETUP.md ("Known Issues"), tests/npu/npu_prod_benchmark.cpp lines 41-42, output/docs/archive/BENCHMARK_RESULTS_legacy_timing.md line 248

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-12（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
