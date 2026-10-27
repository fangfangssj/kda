---
type: Runbook
title: "Pybind padding wrapper when an upstream kernel uses one input's dim as buffer extent for differently-shaped tensors"
description: "An L1-verbatim A3 kernel hard-coding one shape dim as buffer extent traps on A5 as MTE DDR out of range (95) on asymmetric shapes; fix host-side with a torch::zeros padding wrapper, not in the kernel."
tags: [task.debugging, topic.precision, topic.implementation.atomic, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-162-pybind-padding-wrapper-asymmetric-shape.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-162"]
---

# Pybind padding wrapper when an upstream kernel uses one input's dim as buffer extent for differently-shaped tensors

> 原 OKF v1 卡号：**OL-162**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - L1-verbatim A3 kernel traps on A5 V351 with `errcode:(95) MTE DDR out of range` (EC-53) on asymmetric-shape cases (e.g. M<N); the same kernel ran fine on A3 V220 where MTE did not check

## 现象 / 触发

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=port_a3_to_a5, asymmetric_shape`. Unverified on
Ascend910_V220 (A3 silently tolerates the overshoot — the bug is invisible there). Loaded by
`aog-kernel-worker` (Phase C/D — MTE-OOR shows up only on A5) and `aog-precision-probe` (A3-vs-A5
capture diverges on asymmetric shapes).

An upstream A3 kernel was authored assuming related tensors share a single shape dim (the original
test driver used N==M). The kernel hard-codes that one dim as `SetGlobalBuffer` extent / `InitOutput`
extent / batch-stride for tensors whose actual allocation uses a **different** dim. On A5 V351, MTE
boundary checking traps as `errcode:(95) MTE DDR out of range` (EC-53). On A3 V220 the same kernel
ran "fine" because MTE did not check.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

For a **verbatim L1 port** (OL-143 mechanical port — the kernel body is byte-identical to upstream
`arch35/`), DO NOT edit the kernel to fix an upstream asymmetric-shape OOR bug. Instead install a
**pybind-side padding wrapper** that:

1. Detects the asymmetric condition host-side (e.g. `M < N`).
2. Allocates padded versions of the OOR-prone tensors using the larger dim (`max(N, M)`), backed by
   `torch::zeros` (NOT `torch::empty` — must match A3's `aclrtMemset`-to-zero output behavior, and
   the zeros must neutralize the kernel's extra iterations).
3. Copies user inputs into the padded buffers' active region.
4. Allocates output GM with the larger extent so `InitOutput` overshoot stays in-buffer.
5. Launches the kernel against padded buffers — the kernel body sees exactly the shapes the upstream
   code expects.
6. After kernel return, narrows the output back to the user-visible shape (`out.narrow(dim, 0, M)`).

**Correctness invariant**: padding values must make the extra kernel iterations **numerically no-ops**
for the active region — e.g. for gradient-style ops, `grad_dist2_padded[i ≥ M] = 0` makes any
atomic-add from padded indices a zero contribution. Pick padding values that make the kernel's
spurious work invisible in the active output, not just "some default".

**Anti-patterns**:
- DON'T edit the kernel body to add an `if (idx >= realLen) return;` guard — that breaks the
  L1-verbatim contract (OL-143); the patch becomes a hidden fork that future `arch35/` refreshes
  can't reconcile. The pybind wrapper isolates the workaround to the host side.
- DON'T use `torch::empty` for the padding buffers — uninitialized GM at the tail produces
  non-deterministic atomic-add residuals, so the bug reappears one step removed.
