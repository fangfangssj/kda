---
type: Optimization
title: "Multi-dtype support architecture"
description: "Templated kernel + per-type dispatcher: cpp template <typename T> __simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...) { ... } extern \"C\" __global__ __aicore__ void kernel_fp32(...) { Sim"
tags: [task.optimization, topic.precision, topic.programming_model.execution_model.simt]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/f-p2-multi-dtype-support-architecture.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Multi-dtype support architecture

> 原 OKF v1 卡号：**F-P2**。

## 优化点

Templated kernel + per-type dispatcher:
```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...) { ... }

extern "C" __global__ __aicore__ void kernel_fp32(...) {
    Simt::VF_CALL<kernel_vf<float>>(Simt::Dim3{threads}, ...);
}
// Same for fp16, bf16
```

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-P2，convert_patterns_to_okf.py）。confidence 未升格。 -->
