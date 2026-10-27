---
type: Runbook
title: "SOC_VERSION Ascend910B2 causes 507035 on Ascend950PR hardware"
description: "SOC_VERSION Ascend910B2 causes 507035 on Ascend950PR hardware"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-12-soc-version-ascend910b2-causes-507035-on-ascend950.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a2, "950"]
aliases: ["PB-12"]
---

# SOC_VERSION Ascend910B2 causes 507035 on Ascend950PR hardware

> 原 OKF v1 卡号：**PB-12**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - ALL kernels crash with 507035 (vector core exception, error 259 = illegal instruction) at PC offset 0x80. Build succeeds but runtime crashes every time.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A2 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with three CANN-pass ops (17_EmbeddingWithInitialLayernormBackward, 20_FusedRopeWithQkNormAndKvCacheUpdate, 22_HybridAttentionMaskPreparation). Do not downgrade.
- **Status**: CONFIRMED (2026-04-16)
- **Symptom**: ALL kernels crash with 507035 (vector core exception, error 259 = illegal instruction) at PC offset 0x80. Build succeeds but runtime crashes every time.
- **Root cause**: `build_ascendc.py` defaults to `-v Ascend910B2` if no SOC_VERSION specified. Ascend910B2 binary contains instructions not supported on Ascend950PR AIV cores.
- **Fix**: Always pass `-v Ascend950PR_9589` when building on A5 hardware.
- **Impact**: This explains ALL previous unexplained 507035 crashes. Workers MUST always specify SOC_VERSION.
- **Evidence**: 14_AdaptiveInstanceNormalization2DBackward (4th attempt): all kernel variants crashed identically until SOC_VERSION corrected.

<!-- 迁移自 porter kb/target/ascendc/（PB-12，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
