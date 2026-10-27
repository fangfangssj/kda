---
type: Runbook
title: "torch_npu `.contiguous()` / `.copy_()` of a transpose-to-INNERMOST permute HANGS (wedges) the device — a permute that keeps the last dim is fine"
description: "<!-- applies_to_backend: all -->"
tags: [task.debugging, topic.implementation.atomic, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-46-torch-npu-contiguous-copy-of-a-transpose-to-innerm.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["PB-46"]
---

# torch_npu `.contiguous()` / `.copy_()` of a transpose-to-INNERMOST permute HANGS (wedges) the device — a permute that keeps the last dim is fine

> 原 OKF v1 卡号：**PB-46**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - the pybind/host call to .contiguous() or .copy_() never returns; the device wedges. Easily misdiagnosed as a kernel bug because the next kernel launch on the sa

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — not retested; this is a torch_npu/CANN-runtime layout-materialization behavior, may differ by CANN version)`

- **Severity**: HIGH — silent device wedge (no error code, no fault). The hung op leaves the NPU unusable for subsequent launches (D-state, unkillable), so it also masquerades as a downstream "kernel hang" on the same card (see OL-189 for the rotate-to-fresh-NPU diagnostic).
- **The runtime fact**: a 4D **middle-swap** permute such as `(0,2,1,3)` that KEEPS the last (innermost) dim materializes fine via `.contiguous()` / `dst.copy_(view)`. But ANY permute that moves a dim INTO the innermost position — e.g. `(0,2,3,4,1)`, `(0,2,3,4,5,1)`, or even a collapsed 3D `(0,2,1)` — HANGS both `.contiguous()` and `dst.copy_(view)`. The trigger is "innermost-stride change requiring a true transpose-materialize", not rank.
- **Symptom**: the pybind/host call to `.contiguous()` or `.copy_()` never returns; the device wedges. Easily misdiagnosed as a kernel bug because the next kernel launch on the same (now-wedged) card also hangs.
- **Consequence for port_a3**: a V220 generic kernel whose input layout is head-major / query-innermost CANNOT be produced by a pybind transpose into that layout. **Mitigation — author the A5 kernel to read the STANDARD (un-transposed) framework layout directly.** For MSDA: query-major — each core owns whole `(batch, query)` rows; `loc[b,q,:]` and `attn[b,q,:]` are contiguous, `value[b,key,h,:]` is element-contiguous with key-stride `nh*ed`. Reading the standard layout also removed SetAtomicAdd → deterministic by construction.
- **Evidence**: MultiScaleDeformableAttnFunction port_a3_to_a5 kw-1 (2026-06-20, A5/Ascend950PR_957b/CANN 9.1.T500). iter-2 hung on `attn (0,2,3,4,1)` + `loc (0,2,3,4,5,1)` transposes; rewriting the kernel to read the standard mmcv query-major layout (no pybind transpose) → 33/34 inclusive vs fp64 CPU truth, determinism 34/34, std median 80.9µs.
- **Other instances (predicted)**: any port that tries to pre-transpose framework inputs into a kernel-private innermost layout via `.contiguous()`/`.copy_()` — head-major attention layouts, channel-last↔channel-first conversions, any `permute` that lands a previously-outer dim last. Prefer authoring the kernel to consume the framework's native layout.
- **Cross-reference**: OL-189 (wedged-NPU masquerades as kernel hang — rotate to a fresh physical NPU before declaring a kernel bug), OL-165 (no `.cpu()` round-trip / pybind-transpose ban — author the kernel for the native layout), P140 (standalone pybind + `ACLRT_LAUNCH_KERNEL` verify path used here).

<!-- 迁移自 porter kb/target/ascendc/（PB-46，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
