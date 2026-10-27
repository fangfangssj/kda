---
type: Optimization
title: "Duplicate-index CPU scatter may be schedule-dependent and is not unquestioned truth"
description: "Probe same-input determinism before using a CPU scatter call as truth; reject unsupported duplicates or use an explicit fixed-order oracle for scatter and embedding gradients."
tags: [task.optimization, topic.precision]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-291-duplicate-index-cpu-scatter-reference.md
    role: primary
created_at: "2026-07-29T00:00:00Z"
updated_at: "2026-07-29T00:00:00Z"
platforms: []
---

# Duplicate-index CPU scatter may be schedule-dependent and is not unquestioned truth

> 原 OKF v1 卡号：**OL-291**。

`applies_to: soc=all; cann=all; op_class=scatter/index-put/embedding-gradient verifier; backend=ascendc`

## Principle

Overwrite scatter with duplicate destinations has undefined winner semantics, and large CPU implementations may enter parallel paths whose result varies across identical reruns. Probe determinism before using a fresh API call as truth. If duplicates are outside the contract, reject them or generate unique indices. If intentional, define an explicit fixed-order oracle and report that sub-semantic; do not chase CPU or NPU scheduling.

`torch.set_num_threads(1)` may stabilize a diagnostic but does not create a portable contract. For accumulation scatter/backward gradients, use a deterministic fixed-order/high-precision oracle and compare every gradient. Cross-ref OL-90/P-P67 and OL-89.

**Evidence / provenance**: derived from historical card TR-OL-29. On 19_IndexPut (2026-05-21), case 13 (`N=32768`, `M=16384`, fp32 duplicate indices) produced 1,398 differing positions across three CPU invocations; smaller cases at `N=128` and `N=2048` were stable, and one CPU thread collapsed the observed spread to zero. This measures scheduling, not a portable winner semantic.
