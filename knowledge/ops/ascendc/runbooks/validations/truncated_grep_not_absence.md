---
type: Runbook
title: "grep | head 的截断输出不是「没有调用方」的证据——判「未接线」前必须 unpiped 全量或计数重跑"
description: "grep | head 的截断输出不是「没有调用方」的证据——判「未接线」前必须 unpiped 全量或计数重跑"
tags: [task.debugging, topic.implementation]
status: draft
sources: []
created_at: "2026-09-20T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PROCESS-TRUNCATED-GREP-NOT-ABSENCE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PROCESS-TRUNCATED-GREP-NOT-ABSENCE

## 适用场景

- 所有「谁调用了 X」的静态排查。

## 结论（39_ChunkKdaFwd，付过真金）

`grep -rn <sym> | head` 静默不等于无调用方。本次代价：基于截断 grep 的「两个缺口」故事做了两处控制面编辑，其中一处差点删掉一个有测试钉住的安全属性。判 unwired 前 unpiped/计数重跑。检查拒绝时第一假设应是「检查是对的、环境错了」——反面假设恰好能解锁你，正是纸面修补的形状。

- **Evidence**：39 归档 knowledge_update.md finding 2。
- **Cross-ref**：a5ops-development-practice（反压力协议 P9）。
