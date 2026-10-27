---
type: Runbook
title: "OL-114 载体：两遍 tile 循环——D 超大时的一趟策略放不下就两遍走"
description: "OL-114 锚载体卡——正文即 optimizations/ol_114_two_pass_tile_loop_large_d_reductions.md 的规则面"
tags: [task.optimization, topic.memory]
status: draft
sources: []
created_at: "2026-09-21T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-114"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-114 — 两遍 tile 循环（two-pass tile loop for large-D reductions）

## 适用场景

- D（内维）超过 UB 一趟容量的归约/逐点 op（P-P45 单趟 UB 驻留为首选，OL-114 是放不下的 fallback）。

## 规则面

大 D 归约拆两遍 tile 循环（完整正文与实测见 optimizations/ol_114_two_pass_tile_loop_large_d_reductions.md——原 OKF v1 卡号 OL-114）。

- **Cross-ref**：P-P45（单趟首选）、P-P107（GM workspace 三趟流式）。
