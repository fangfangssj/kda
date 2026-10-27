---
type: "Runbook"
title: "Mix合图包入 V2 致任务数减半 E2E 劣化"
description: "Ascend950（CANN 9.1.0）上 sparse_flash_attention_quant _d_950 变体，把 sg_set_scope Mix合图闭合点后移使 V2 段（division+cast）也包入调度域后，AICore Utilization 从 67% 升到 80% 但 Total Task Count 从 2304 减半到 1536（并行度下降），E2E 反而从 729us 劣化约 1.13× 到 824us；精度未退化但仍回退，V2 入 Mix合图在本算子结构下 banned。"
tags: ["task.optimization", "topic.implementation.pipeline", "topic.performance.analysis", "topic.programming_model.execution_unit.cube"]
status: stable
sources: []
created_at: "2026-08-18T08:41:27Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
aliases: ["sparse_flash_attention_quant_d_950"]
---

# Mix合图包入 V2 致任务数减半 E2E 劣化

## 现象

在 Ascend950（CANN 9.1.0）上调优 `sparse_flash_attention_quant` 的 `_d_950` 变体。当前最优配置（Mix合图 `sg_set_scope=20001` 覆盖 V0→C1→V1→C2，V2 段在 scope 闭合后裸跑）E2E 处于最优、Utilization 67%、Total Task Count 2304、精度 28 误差点通过。为尝试进一步减少 V2 段的 GM↔UB 搬运，把 Mix合图 scope 闭合点后移、把 V2 段也包入 Mix合图。改后单测仍通过（28 误差点，未退化），但 AICore profiling 显示：Utilization 升到约 80%，Total Task Count 却减半到约 1536，E2E 反而劣化约 1.13×。即"利用率升 ≠ E2E 降"——任务数减半伴随的利用率提升来自分母（时间）变大而非有用工作量增加。agent 据此判断并行度下降致劣化并回退到 V2 不入 Mix合图的最优配置。

## 适用场景

- **工程形态**：kernel 直调（pypto.frontend.jit DSL，`@pypto.frontend.jit` 装饰算子经 PyPTO 前端编译为 `_d_950` kernel）。
- **API 路径**：`pypto.set_pass_options(sg_set_scope=20001)` 开启 Mix合图、`sg_set_scope=-1` 闭合；Mix合图把多段算子并入同一调度域以减少段间 GM↔UB 搬运。
- **验证过的平台与 CANN 版本**：Ascend950（CANN 9.1.0）。`platforms: [950]` 与此一致。
- **适用粒度**：`operator_family`（适用于 flash attention 类多段算子的 Mix合图边界决策）。
- **路线优先级**：`banned`——把 V2 段（小计算量、依赖重）包入 Mix合图在本算子结构下无收益且劣化 E2E，禁止作为优化方向。
- **不适用或未验证场景**：a2/a3 未验证；若 V2 段计算量大或与前置段存在大量中间张量复用，结论可能不成立（需独立实测）。

## 失败路线

**尝试**：把 Mix合图 scope 闭合点从 V1→C2 之后后移到 V2 之后，使 V2 段（division + cast）也落入 `sg_set_scope=20001` 调度域。目标：减少 V2 段与前置段之间的 GM↔UB 搬运，进一步降低 E2E。

**客观失败**：改后同一开发会话 AICore profiling 实测——V2 入 Mix合图态 Total Task Count 减半（2304→约 1536），E2E 劣化约 1.13×（最优态 729us 量级 → 824us 量级），Utilization 反升（67%→约 80%），精度 28 误差点两态均通过未退化。Mix合图把 V2 并入调度域后，框架为维持依赖把多核任务合并、任务数减半，并行度下降的代价超过了段间搬运节省的收益，E2E 不降反升。本路线对"V2 入 Mix合图以降 E2E"判 `banned`。

**回退**：撤销 scope 后移，V2 段回到 scope 闭合后裸跑的最优配置（374 号 tool_result 的最优态基线）。

**边界**：利用率提升看似收益，实为任务数下降的副作用——当合图导致任务数下降时，利用率提升来自分母（时间）变大而非有用工作量增加。结论绑定本 SFA 算子 V2 段（division+cast，计算量小、依赖重）的结构；若 V2 段计算量大或与前置段存在大量中间张量复用，包入 Mix合图可能反而有益，结论不适用。

## 验证信号

- V2 入 Mix合图态 AICore profiling：Total Task Count 减半（2304→约 1536）、Utilization 升（67%→约 80%）、E2E 劣化约 1.13×、精度 28 误差点通过（来源平台：950；证据等级：tool_observed）
- 最优对照态（V2 不入 Mix合图）AICore profiling：Total Task Count 2304、Utilization 67%、E2E 处于最优量级、精度 28 误差点通过（来源平台：950；证据等级：tool_observed）
- V2 入 Mix合图态二次确认 run：Total Task Count、Utilization 与 E2E 与首次一致（来源平台：950；证据等级：tool_observed）

## 失效边界

- 平台：仅 Ascend950 实测；a2/a3 的多核任务调度策略可能不同，未验证，推断可迁移但需独立实测。
- 算子/段：结论针对本 SFA 算子 V2 段（division+cast，计算量小、依赖重）；若 V2 段计算量大或与前置段存在大量中间张量复用，包入 Mix合图可能反而有益，结论不适用。
- 调度机制：Mix合图导致任务数减半的确切机制（依赖驱动的任务合并还是 scope 内串行化）未追到实现层，"任务数下降即不采纳"为实测经验判据。