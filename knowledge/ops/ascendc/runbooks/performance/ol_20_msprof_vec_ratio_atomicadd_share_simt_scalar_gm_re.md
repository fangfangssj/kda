---
type: Runbook
title: "msprof vec_ratio ≠ atomicAdd share (SIMT scalar GM reads also go through the VEC pipe)"
description: "In SIMT mode, `aiv_vec_ratio=0.99` was misread as \"99% atomicAdd\". Measurement showed that removing atomicAdd saves only 4.4% (158us/3605us). 95%+ of VEC time is scalar GM random reads (indirect addre"
tags: [task.optimization, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.implementation.atomic, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-20-msprof-vec-ratio-atomicadd-share-simt-scalar-gm-re.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-20"]
---

# msprof vec_ratio ≠ atomicAdd share (SIMT scalar GM reads also go through the VEC pipe)

> 原 OKF v1 卡号：**OL-20**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - msprof shows high vec_ratio in SIMT backward kernel

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
In SIMT mode, `aiv_vec_ratio=0.99` was misread as "99% atomicAdd". Measurement showed that removing atomicAdd saves only 4.4% (158us/3605us). 95%+ of VEC time is scalar GM random reads (indirect addressing such as `input[expert*hdim+tid]`). NPU has no L2 cache; every scalar GM read goes to HBM through the VEC pipe. Must compare "with atomicAdd" vs "without atomicAdd" kernels to determine the actual atomicAdd cost. → P-P24 Sort-to-Reuse optimizing GM read amplification is the key.

## 证据
E11 msprof: BwdFull=3605us, grad_weight_only(no atomicAdd)=3447us, diff=158us(4.4%)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-20（category=profiling_interpretation，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
