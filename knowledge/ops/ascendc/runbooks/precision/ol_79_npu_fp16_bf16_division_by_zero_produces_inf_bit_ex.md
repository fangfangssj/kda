---
type: Runbook
title: "NPU fp16/bf16 division by zero produces inf — bit-exact match with PyTorch CPU"
description: "Ascend950PR fp16/bf16 division (including CANN torch_npu Div and AscendC VEC Div) produces `inf` when the divisor is near zero, with behavior **exactly matching** PyTorch CPU (bit-exact match). `pow(s"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-79-npu-fp16-bf16-division-by-zero-produces-inf-bit-ex.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-79"]
---

# NPU fp16/bf16 division by zero produces inf — bit-exact match with PyTorch CPU

> 原 OKF v1 卡号：**OL-79**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - kernel has `A / std` or `pow(std, -3)` style ops that may produce inf, and the reference in native dtype also produces inf

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
Ascend950PR fp16/bf16 division (including CANN torch_npu Div and AscendC VEC Div) produces `inf` when the divisor is near zero, with behavior **exactly matching** PyTorch CPU (bit-exact match). `pow(std, -3)` overflow also produces `inf`, matching CPU. This means **no special handling of inf/NaN is needed**: do the division in native dtype directly and the resulting inf/NaN will match the reference exactly. Do NOT clamp early or cast to fp32 to avoid inf — that will instead cause a precision mismatch (reference expects inf but you returned a finite value).

## 证据
2026-04-16 A5 experiment `tests/repro/fp16_inf_div_test.py`: 6 fp16 division groups (including 1/0=inf, 100/6.1e-5=inf, -1/-0=inf) CPU vs NPU exact match; pow(std,-3) 4 groups exact match; bf16 3 groups exact match. E1 level (hardware measured).
  - 7_MoeGatingTopKSoftmax Phase D iter 1 (2026-04-17): the `is_finished` row used a -3.4e38 sentinel → max-subtract collapses to zero → softmax = 1/N uniform, which is wrong. PyTorch semantics: a fully-masked row softmax → NaN. Fix: short-circuit `is_finished` directly to NaN output. Same spirit as OL-79 (native semantics naturally match).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-79（category=precision，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
