---
type: Optimization
title: "bf16 precision handling (scatter-add waiver)"
description: "Anti-pattern: using fp32 atol/rtol for bf16 tests → many false-positive FAILs Correct pattern: cpp float atol = (dtype == \"bf16\") ? 2e-2f : 1e-4f; bool waiver = (dtype == \"bf16\"); compare_data(npu, gp"
tags: [task.debugging, topic.precision]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/f-p1-bf16-precision-handling-scatter-add-waiver.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# bf16 precision handling (scatter-add waiver)

> 原 OKF v1 卡号：**F-P1**。

## 优化点

**Anti-pattern**: using fp32 atol/rtol for bf16 tests → many false-positive FAILs

**Correct pattern**:
```cpp
float atol = (dtype == "bf16") ? 2e-2f : 1e-4f;
bool waiver = (dtype == "bf16");
compare_data(npu, cpu_truth, n, dtype_str, atol, rtol, "fwd", waiver);
```

**Note**: bf16 mismatch is expected behavior **only** for scatter-add-class ops (Pooling fwd/bwd). **SG forward is deterministic computation — bf16 must not have any mismatch**; if it does, it is a bug.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-P1，convert_patterns_to_okf.py）。confidence 未升格。 -->
