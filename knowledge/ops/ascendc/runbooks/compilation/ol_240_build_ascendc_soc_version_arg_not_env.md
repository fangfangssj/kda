---
type: Runbook
title: "build_ascendc.py takes SOC from the --soc-version arg, not an env var — manual builds silently mis-target the arch"
description: "A manual build_ascendc.py invocation that only exports SOC_VERSION env silently compiles the device kernel for the WRONG arch (default Ascend910B2/V220), then the A5 NPU throws error 507035 (vector-core address out-of-bounds) on every case despite a clean build and a healthy NPU."
tags: [task.debugging, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-240-build-ascendc-soc-version-arg-not-env.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: [a2, a3, "950"]
aliases: ["OL-240"]
---

# build_ascendc.py takes SOC from the --soc-version arg, not an env var — manual builds silently mis-target the arch

> 原 OKF v1 卡号：**OL-240**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - A manual/standalone build_ascendc.py build (perf-remeasure, hardware-probe, hand-build) crashes 507035 on every case on a verified-idle A5 NPU, with a clean build

## 现象 / 触发
`applies_to: soc=Ascend950PR (V351); cann=9.1.0; op_class=all`.

A manual/standalone `build_ascendc.py` invocation that only exports a `SOC_VERSION` environment variable silently compiles the device kernel for the WRONG arch. At runtime the A5 NPU throws **error 507035 (vector-core address out-of-bounds)** on every case, with a clean build and a healthy NPU — so it looks like a kernel or env failure but is purely the SOC-arg mistake.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A2 系列、Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
`build_ascendc.py` reads the target SOC from its `-v/--soc-version` argument (default `Ascend910B2`), NOT from a `SOC_VERSION` environment variable. Exporting only the env var leaves the default `Ascend910B2` (A2/V220) arch in effect. Fix: always pass `-v Ascend950PR_9579` (or the actual target soc) to build_ascendc.py; never rely on the env var.

Scope: the orchestrator's deploy/build path passes `-v` correctly (UNAFFECTED) — this only bites MANUAL builds (perf-remeasure, hardware-probe, hand-build).

Concrete anchor: iou_v2 perf-remeasure first build (env-only `SOC_VERSION`) crashed 507035 at every case on a verified-idle A5 NPU; rebuilding with `-v Ascend950PR_9579` fixed it (NPU health independently confirmed via a generic matmul, so it was the SOC-arg, not a kernel/env/hardware fault).

## 证据
- iou_v2 perf-remeasure (2026-06-21, .141/NPU5, A5 Ascend950PR_9579): env-only build -> 507035 all cases -> `-v Ascend950PR_9579` cleared. Reconfirmed in modulate + roi_align_rotated perf-characterization manual builds.
- Predicted: any manual/standalone `build_ascendc.py` invocation outside the orchestrator (perf re-measure agents, hardware probes, hand builds) on a non-default SOC; any harness path that sets SOC via env and assumes the build tool reads it.
- Cross-ref: PORT_A3_PERF_METHODOLOGY (perf-remeasure manual builds); the orchestrator deploy/build passes -v so production op-gen is unaffected.
