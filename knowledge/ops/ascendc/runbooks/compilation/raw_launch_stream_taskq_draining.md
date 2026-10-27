---
type: Runbook
title: "Raw-launch stream handoff must be `.stream(true)` (taskQ-draining) on CANN 9.1.0"
description: "Raw-launch stream handoff must be `.stream(true)` (taskQ-draining) on CANN 9.1.0"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-RAW-LAUNCH-STREAM-TASKQ-DRAINING"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-RAW-LAUNCH-STREAM-TASKQ-DRAINING — Raw-launch stream handoff must be `.stream(true)` (taskQ-draining) on CANN 9.1.0

## 适用场景

- 适用平台与范围：soc=Ascend950DT/Ascend950PR；CANN 9.1.0；所有 benchmark pybind raw-launch。
- 验证来源：soc=Ascend950DT_9582、CANN 9.1.0（13_Cat M-I run-1 kw-1，2026-09-17）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950DT/Ascend950PR; cann=9.1.0; op_class=all-benchmark-pybind`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (13_Cat M-I run-1 kw-1, 2026-09-17)`

`raw_launch_stream_guard` (CANN 9.1.0 harness) requires `c10_npu::getCurrentNPUStream().stream(true)` — the taskQ-draining form — before `aclrtlaunch_*` raw launches. The older `.stream()` (seen in pre-guard archives) fails `build_ascendc.py` pre-compile with `ValueError: Async raw launch ... requires a taskQ-draining ... handoff`. One-token fix in pybind.

- **Symptom signature**: build fails at pre-compile with the ValueError above, on a kernel whose device code is unchanged.
- **Evidence**: 13_Cat M-I run-1 kw-1 iter 2 — build-gate FAIL → one-token fix → Build OK, .so RTLD_NOW load-checked 1/1.
- **Cross-ref**: CAND-DUAL-TILE-SIZE-CONSTANT-DIVERGENCE (same pybind host/device divergence family). backend=ascendc.
