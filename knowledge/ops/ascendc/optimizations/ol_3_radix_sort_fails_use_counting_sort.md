---
type: Optimization
title: "Radix sort fails on AscendC; use counting sort"
description: "Multi-pass radix sort with global atomicAdd scatter fails on AscendC (0/61); single-pass counting sort (histogram->prefix_sum->scatter) gives 61/61 correct and 2.84x over host std::sort."
tags: [task.optimization, topic.programming_model.execution_model.simt, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-3-radix-sort-fails-use-counting-sort.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Radix sort fails on AscendC; use counting sort

> 原 OKF v1 卡号：**OL-3**。

## 优化点 / 选型

**场景 / Trigger**: implementing an on-device sort for AscendC SIMT kernels.

**选型**: A 4-pass byte-wise radix sort using global `atomicAdd` scatter was implemented first and failed completely (0/61 clusters correct). Multi-pass scatter is unstable on AscendC — each pass's global `atomicAdd` destroys the ordering established by the previous pass. Switching to **counting sort** (histogram -> prefix_sum -> scatter, single-pass) achieved **61/61 correct and 2.84x speedup** over host `std::sort`.

**Why counting sort works here**: same-key ordering does not affect the downstream register accumulation, so the sort does not need to preserve stability across passes.

**Evidence**: OPTIMIZATION_PLAN.md Batch 7 ("radix sort 0/61 correct"), `output/src/pooling/radix_sort_kernel.h`.
