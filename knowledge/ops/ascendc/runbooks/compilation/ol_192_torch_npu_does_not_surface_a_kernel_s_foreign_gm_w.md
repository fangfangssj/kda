---
type: Runbook
title: "torch_npu does not surface a kernel's foreign GM writes into a returned raw-byte workspace tensor — probe via TYPED OUTPUT tensors"
description: "applies_to: soc=all; cann=9.0.0; bisheng=n/a; op_class=all (debugging methodology)"
tags: [task.debugging, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-192-torch-npu-does-not-surface-a-kernel-s-foreign-gm-w.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3]
aliases: ["OL-192"]
---

# torch_npu does not surface a kernel's foreign GM writes into a returned raw-byte workspace tensor — probe via TYPED OUTPUT tensors

> 原 OKF v1 卡号：**OL-192**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=all; cann=9.0.0; bisheng=n/a; op_class=all (debugging methodology)

`applies_to: soc=all; cann=9.0.0; bisheng=n/a; op_class=all (debugging methodology)`
`verified_on: soc=Ascend910_9382; cann=9.0.0`

**Principle**: when debugging a cube↔vec kernel by returning its GM scratch buffer (`at::zeros({totalWs}, kByte)`) from pybind to inspect intermediate workspace tensors, torch_npu reads the buffer back as ALL-ZERO — it does not track the kernel's foreign (device-side, non-torch) writes into a raw-byte tensor's storage, so the readback is not a faithful view of what the kernel wrote. Workspace-tensor readback is therefore an unreliable probe and will send you chasing a phantom "workspace is zero" bug. Instead, probe kernel internals through **typed OUTPUT tensors** (the op's real fp16/bf16/fp32 outputs, which torch_npu DOES track) — e.g. temporarily route an intermediate to a real output slot, or add a debug output tensor of the correct dtype.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Evidence
- lightning_indexer_grad (A3, 2026-05-27): returning the raw-byte workspace from pybind read back all-zero, falsely suggesting the gather never wrote `ws_gk`; the gather was actually fine. The trustworthy signal was the typed `dq`/`dk`/`dweights` output tensors (e.g. dq's alternating-zero-over-D signature localized a fractal bug).

## Other instances (predicted)
Any AscendC kernel debugged by returning GM scratch to Python; any multi-stage cube↔vec / workspace-queue kernel where you want to inspect an intermediate.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-192（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
