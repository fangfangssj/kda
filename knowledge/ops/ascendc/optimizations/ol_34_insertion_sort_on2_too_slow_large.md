---
type: Optimization
title: "Insertion sort is O(N²) — too slow for sort_len > 4K"
description: "Scalar insertion sort is correct but O(N^2) per line; sort_len=16384 is ~268M comparisons/line and times out on NPU. Use radix O(N*k) or bitonic O(N*log^2 N) sort for sort_len > 4K."
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-34-insertion-sort-on2-too-slow-large.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Insertion sort is O(N²) — too slow for sort_len > 4K

> 原 OKF v1 卡号：**OL-34**。

## 优化点 / 选型

**场景 / Trigger**: implementing sort / topk operations.

**选型**: Insertion sort via scalar `GetValue` / `SetValue` is correct but **O(N²) per line**. For `sort_len=16384` that is ~268M comparisons per line — a timeout on NPU. For large cases use **radix sort (O(N·k))** or **bitonic sort (O(N·log²N))**.

**Tradeoff**: Insertion sort is fine for small `sort_len` (fp32 small cases pass); only large cases (>4K) need the higher-complexity algorithms. For a concrete working large-sort choice on AscendC, see OL-3 (counting sort).

**Evidence**: Sort V1 (2026-04-09), fp32 small cases pass but large cases timeout.
