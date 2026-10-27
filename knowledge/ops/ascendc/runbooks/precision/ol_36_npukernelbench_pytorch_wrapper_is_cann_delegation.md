---
type: Runbook
title: "NPUKernelBench — PyTorch wrapper is CANN delegation (PROHIBITED)"
description: "NPUKernelBench — PyTorch wrapper is CANN delegation (PROHIBITED)"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-36-npukernelbench-pytorch-wrapper-is-cann-delegation.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-36"]
---

# NPUKernelBench — PyTorch wrapper is CANN delegation (PROHIBITED)

> 原 OKF v1 卡号：**OL-36**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - when considering "simple" implementation via PyTorch ops

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass ops (20_FusedRopeWithQkNormAndKvCacheUpdate, 22_HybridAttentionMaskPreparation). Do not downgrade.
- **Category**: trust_calibration
- **Loaded by**: Builder, QA
- **Trigger**: when considering "simple" implementation via PyTorch ops
- **Lesson**: Implementing model_new_ascendc.py by calling PyTorch ops (torch.permute, torch.sort, F.layer_norm, etc.) delegates to CANN via torch_npu. This is the SAME as calling aclnn* APIs inside a kernel — it's wrapper hacking. ALL computation must use AscendC primitives (DataCopy, VEC ops, TQue/TBuf, scalar GetValue/SetValue).
- **Evidence**: Permute (#12) was caught as CANN delegation and reverted (2026-04-09)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-36（category=trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
