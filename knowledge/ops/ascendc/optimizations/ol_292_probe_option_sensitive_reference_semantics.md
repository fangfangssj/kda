---
type: Optimization
title: "Probe option-sensitive CPU and NPU reference semantics before deriving the formula"
description: "Measure CPU-vs-NPU and NPU option-A-vs-option-B behavior first; framework bindings may ignore, gate or collapse an option before device dispatch."
tags: [task.optimization, topic.implementation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-292-probe-option-sensitive-reference-semantics.md
    role: primary
created_at: "2026-07-29T00:00:00Z"
updated_at: "2026-07-29T00:00:00Z"
platforms: [a3]
---

# Probe option-sensitive CPU and NPU reference semantics before deriving the formula

> 原 OKF v1 卡号：**OL-292**。

`applies_to: soc=all; cann=all; op_class=reference/truth preflight; backend=ascendc`

## Principle

A framework binding may ignore, gate or collapse a documented option before reaching the device primitive. Before generating a forward or backward candidate, evaluate at least two option values on identical inputs and compare CPU-vs-NPU plus NPU-option-A-vs-option-B. If options collapse, record the measured behavior and identify the required authority: API mathematical truth, declared test reference behavior, or shipped-device parity. Never substitute one silently.

Save inputs, option values, outputs and framework/CANN versions. For optional-tensor-gated features also vary tensor presence (OL-202). Differentiate backward truth from the exact measured forward semantics. Cross-ref OL-89/OL-109.

**Evidence / provenance**: derived from historical card TR-OL-9. On A3 (2026-05-04, torch_npu 2.7.1/CANN 8.3), `gelu(approximate=none)` differed from CPU/textbook erf by 4.74e-4 but matched tanh approximation within 4.77e-7; two option values were bit-identical on NPU while CPU distinguished them by 4.73e-4. The measurements are empirical; declaring truth authority is the process conclusion.
