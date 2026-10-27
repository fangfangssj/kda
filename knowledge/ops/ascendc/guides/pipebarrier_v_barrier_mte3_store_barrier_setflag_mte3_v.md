---
type: Guide
title: "PipeBarrier 删减实测边界：同管道 V barrier 删除安全但饱和管道无收益；MTE3 逐 store barrier 被 SetFlag<MTE3_V> 覆盖"
description: "4-14 实测：266 个 V→V barrier 删除逐位不变但 perf≈0（吞吐饱和非杠杆）；逐 store MTE3 barrier 冗余可删；删除前先判延迟停滞 vs 吞吐饱和"
tags: [task.optimization, topic.implementation.synchronization, topic.programming_model.execution_unit.vector]
status: draft
sources: []
created_at: "2026-09-25T00:18:31Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PipeBarrier"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PipeBarrier 删减实测边界：同管道 V barrier 删除安全但饱和管道无收益；MTE3 逐 store barrier 被 SetFlag<MTE3_V> 覆盖

## Evidence

- **run_id**: a37a0522-5a5f-4bc8-b9ab-fc68bdcf7443
- **host**: Ascend950DT
- **clock**: 2026-09-25T00:20Z
- **trigger**: manual-post-run-intake
- **authorization**: 2026-09-25 ruling

## 适用

AscendC vector/cube kernel 的 PipeBarrier 删减决策——什么时候能删（安全）、什么时候删了也没用（别花迭代）。

## 机制与实测

1. **同管道 `PipeBarrier<PIPE_V>`（V→V 之间）**：官方卡 `tbuf_cross_pipe_dependency_requires_pipe_all` 已证 V 管道指令按序执行、可删。本卡补一条实测边界：**在吞吐饱和的 vector 循环上删除 perf 收益≈0**——4-14 上删除每 unit 266 个 barrier，输出 50/50 逐位不变，但 perf 无变化（gemm 的 2×Broadcast+MulAddDst 每个 rank-1 项本就背靠背退休）。结论：该类删除安全但**不是杠杆**，管道饱和时别为它花 optimizer 迭代。
2. **逐 store `PipeBarrier<PIPE_MTE3>`**：当下一个跨管道 flag 是 `SetFlag<MTE3_V>`（在 MTE3 管道上发出）时，它已经把 V 的 wait 排在 store 完成之后——独立 barrier 冗余，删除后逐位不变。

## 证据

run a37a0522-5a5f-4bc8-b9ab-fc68bdcf7443（4-14，fixed-50）：两处删除均 50/50 PASS 逐位不变（每用例 max_abs 与删除前完全相同）。

## 关联卡

姊妹卡：`runbooks/precision/tbuf_cross_pipe_dependency_requires_pipe_all`（V 管道指令按序执行的官方机制卡；本卡补其实测边界与 MTE3 store barrier 冗余案例）。

## 判据

删 barrier 前先问：这条管道是延迟停滞还是吞吐饱和？前者删了有收益，后者删除只降指令数不提性能（EV 低于噪声底）。
