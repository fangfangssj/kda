---
type: Optimization
title: "`__threadfence` misused as delay/wait"
description: "__threadfence() is a memory barrier (ensures writes become visible to other threads), not a delay/wait. The correct approach is cooperative-group __shfl sync or a spin-wait."
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/f-ap2-threadfence-misused-as-delay-wait.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# `__threadfence` misused as delay/wait

> 原 OKF v1 卡号：**F-AP2**。

## 反模式

`__threadfence()` is a memory barrier (ensures writes become visible to other threads), not a delay/wait. The correct approach is cooperative-group `__shfl` sync or a spin-wait.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-AP2，convert_patterns_to_okf.py）。confidence 未升格。 -->
