---
type: Runbook
title: "序列类算子的 Pass-A 采集体积随 ΣT 增长（ΣT≈60k 时 1.5GB）——采集前查磁盘水位，容器内留存要预先规划"
description: "序列类算子的 Pass-A 采集体积随 ΣT 增长（ΣT≈60k 时 1.5GB）——采集前查磁盘水位，容器内留存要预先规划"
tags: [task.testing, topic.implementation]
status: draft
sources: []
created_at: "2026-09-20T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-CAPTURE-SIZE-SCALES-WITH-SEQLEN"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-CAPTURE-SIZE-SCALES-WITH-SEQLEN

## 适用场景

- 序列/递归类算子的 Pass-A 采集（KDA、mamba、attention 等变长 T）。

## 结论（39_ChunkKdaFwd 实测）

Pass-A capture 体积 ≈ ΣT 线性（39 的 40 例 ΣT≈60k → 1.5GB）。采集**前**查磁盘水位线（本机 20/40GiB 两档规则），容器内留存提前规划，不要采集到一半写爆 /data 或根盘。

- **Evidence**：39 归档 knowledge_update.md。
- **Cross-ref**：disk-safe-cleanup 水位纪律。
