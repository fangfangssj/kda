---
type: Optimization
title: "Pure-copy ops — TQueBind replaces the VECIN→VECOUT bridge"
description: "Pure-copy ops (Cat/Split/Pad/Repeat) can bind VECIN with VECOUT via TQueBind and skip the Adds(0.0f) bridge (EC-21 workaround) for higher efficiency. UNVERIFIED on A5 — needs experimental confirmation on CANN 9.0.0."
tags: [task.optimization, topic.programming_model.execution_unit.vector]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-49-tquebind-pure-copy-ops.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
---

# Pure-copy ops — TQueBind replaces the VECIN→VECOUT bridge

> 原 OKF v1 卡号：**OL-49**。

## 优化点 / 选型

**场景 / Trigger**: data-movement ops (Cat / Split / Pad / Repeat) with no real VEC compute.

**选型**: Pure-copy ops can bind VECIN with VECOUT using the **TQueBind** interface and skip the `Adds(0.0f)` bridge. Current Cat/Split/Pad all use the Adds bridge (EC-21 workaround; see OL-37); TQueBind is a more efficient alternative.

**Status**: **UNVERIFIED on A5** — need to experimentally confirm the TQueBind API works on CANN 9.0.0 before adopting.

**Source**: hiascend.com best practices (2026-04).
