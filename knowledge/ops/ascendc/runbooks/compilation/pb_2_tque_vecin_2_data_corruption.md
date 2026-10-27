---
type: Runbook
title: "TQue<VECIN,2> Data Corruption"
description: "TQue<VECIN,2> Data Corruption"
tags: [task.debugging, topic.implementation.synchronization, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-2-tque-vecin-2-data-corruption.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3, "950"]
aliases: ["PB-2"]
---

# TQue<VECIN,2> Data Corruption

> 原 OKF v1 卡号：**PB-2**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - 99.5% elements corrupted when using TQue with depth 2

- **Symptom**: 99.5% elements corrupted when using TQue with depth 2
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use TQue<VECIN,4> (depth 4 works correctly)
- **Status**: OPEN
- **Evidence**: kb/okf/runbooks/hardware/target-ascend950pr.md, E13 test data

<!-- 迁移自 porter kb/target/ascendc/（PB-2，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## V351 实战补充（2026-08-26，57_ParallelPolarizedSelfAttention_evo D1 沉淀）

- 57 是 PB-2 在 V351 的活实例：A3 上验证过的 `TQue<...,2>` + InitBuffer depth-2 在 A5 输出全零——O5 45/50 FAIL 且 `matched_count==small_count`（57 ledger 行 9）。
- 修复：3 个 TQue（xQue_/xOutQue_/wQue_）depth 2→4 + InitBuffer depth 参数同步 2→4（ledger 行 9）。
- **联动代价**：depth 翻倍直接放大 UB 占用；57 升 depth-4 后 11 个冻结 shape 的 InitBuffer 总和越 A5 248KB 可用上限（ledger 行 11）。**升 depth 后必须立即重算全部 InitBuffer 字节和**；越界就减容 / 转 TQue<VECIN,1> / 重排 buffer 顺序。
- 判别注意：全 case mismatch 且输出全零时先确认 small_count>0，否则 `matched_count==small_count` 是伪迹（57-D3）。
