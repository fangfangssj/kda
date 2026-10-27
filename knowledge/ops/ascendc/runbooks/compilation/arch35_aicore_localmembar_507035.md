---
type: "Runbook"
title: "arch35 __aicore__ 调 LocalMemBar 报 507035"
description: "Ascend950PR (arch35/dav_c310) 上 Reg::LocalMemBar<VEC_STORE,VEC_LOAD> 是 __simd_callee__ 函数，须在 __simd_vf__ (VF) 调用上下文内使用；从常规 __aicore__ kernel 函数直接调用触发运行时 507035 (AIV 错误)。本 trace 两次独立调用（旧 4-buffer 代码、新 5-buffer 重排代码）均 507035。作为 count-based Compare→Select mask 隐患的最小 sync 修复手段 banned。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-18T07:10:44Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["AIV_error", "count_based_Compare"]
---

# arch35 __aicore__ 调 LocalMemBar 报 507035

## 现象

在 Ascend950PR (arch35/dav_c310, CANN 9.0.0-beta.2) 排查 `softplus_v2_grad` 的 count-based Compare→Select async-VF mask VEC_STORE→VEC_LOAD RAW 隐患时，找到 arch35 上针对 VEC_STORE→VEC_LOAD 的专用内存屏障 `AscendC::Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>`（softmax regbase `SoftmaxFlashNDImpl` 在用）。尝试在 `ComputeFp32Core`（`__aicore__ inline` 常规函数）内、`Compare` 之后直接调用：

```
Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
```

两次独立尝试（旧 4-buffer 代码、新 5-buffer 重排代码）均触发运行时错误 507035 (AIV error)，kernel 执行失败。

## 适用场景

- **工程形态**：kernel 直调（RegBase，`__aicore__` 常规函数内调 `AscendC::` Level2 API，未把算子段包进 `__simd_vf__`）。
- **API 路径**：`AscendC::Reg::LocalMemBar<src, dst>`（头 `reg_compute/kernel_reg_compute_membar_intf_impl.h`）；`Reg::MemType::VEC_STORE` / `VEC_LOAD` / `SCALAR_LOAD`（头 `reg_compute/` 内 enum）。
- **验证过的平台与 CANN 版本**：Ascend950PR (arch35/dav_c310)，CANN 9.0.0-beta.2。仅本平台实测，`platforms: [950]`。
- **适用粒度**：`operator_family`（适用于 arch35 上任何想在 `__aicore__` 常规函数里调 `Reg::LocalMemBar` 的算子）。
- **路线优先级**：`banned`——从 `__aicore__` 直接调 `Reg::LocalMemBar` 作为修复手段 banned；须把调用点包进 `__simd_vf__` 函数或在 `__simd_vf__` 融合体内使用。
- **不适用或未验证场景**：`__simd_vf__` 函数内调用同一 API（softmax regbase 路径）不在本卡否定范围；非 arch35 平台 (a2/a3) 未验证，`__simd_callee__` 调用约定可能不同。

## 失败路线

> 来源：本 trace 在 Ascend950PR + CANN 9.0.0-beta.2 上的两次重编重跑。

**banned 路线：在 `ComputeFp32Core`（`__aicore__ inline`）内 `Compare` 之后直接插入 `Reg::LocalMemBar<VEC_STORE, VEC_LOAD>()`，意图排空 count-based Compare 的 async-VF VEC_STORE 使后续 Select 的 VEC_LOAD 可见。**

- **尝试**：两版独立代码（旧 4-buffer 代码、新 5-buffer 重排代码）均在 `Compare` 后插入上述 `Reg::LocalMemBar` 调用，重编重跑。
- **客观失败**：运行时 507035 (AIV error)，两次独立代码版本均复现，kernel 执行失败（来源平台：950；证据等级：tool_observed）。
- **边界**：`Reg::LocalMemBar` 声明为 `__simd_callee__`，要求调用点处于 `__simd_vf__` (VF) 调用上下文；从常规 `__aicore__` 函数直接调用违反调用约定，AIV 报 507035。该约束来自 CANN 头 `reg_compute/kernel_reg_compute_membar_intf_impl.h` 与 softmax regbase `SoftmaxFlashNDImpl`（`__simd_vf__ inline` 函数）在其内调用同一 API 的对照（来源平台：950；证据等级：artifact_verified）。故从 `__aicore__` 常规函数直接调 `Reg::LocalMemBar` 作为 count-based Compare→Select 隐患的最小 sync 修复，**banned**。
- **对照**：repeat-based Compare (Level0) 内部带 `Reg::LocalMemBar<VEC_STORE, SCALAR_LOAD>`（读 cmpmask 进 `g_cmpMaskLow`），而 count-based Compare (Level2) 内部无 LocalMemBar——这正是 RAW 隐患成因（来源平台：950；证据等级：artifact_verified）。

唯一本 trace 实测可见的排空仍是 `LocalTensor::GetValue` 标量同步；正向方向是把整段算子（含 Compare/Select）包进 `__simd_vf__` Reg:: 融合体，在 VF 上下文内按需用 `Reg::LocalMemBar`，或令 mask 全程留 MaskReg 不做 UB 往返——均未在本 trace 完成（重写未完成即被回退）。

## 验证信号

本段只收本 trace 内 tool_observed 实测与从 CANN 头源码逐字核对的结构事实（artifact_verified）；agent 推断的机制假设不进此段。

- 从 `__aicore__` 常规函数调 `Reg::LocalMemBar<VEC_STORE,VEC_LOAD>` 触发 507035 (AIV error)，两次独立代码版本（旧 4-buffer、新 5-buffer 重排）重编重跑均复现（来源平台：950；证据等级：tool_observed）
- `Reg::LocalMemBar` 声明为 `__simd_callee__`，softmax regbase `SoftmaxFlashNDImpl`（`__simd_vf__ inline` 函数）在其内调用同一 `Reg::LocalMemBar<VEC_STORE, VEC_LOAD>`（头 `reg_compute/kernel_reg_compute_membar_intf_impl.h`、`.../softmax/regbase/c310/softmax_flash_impl.h`）（来源平台：950；证据等级：artifact_verified）
- count-based Compare (Level2) 内部无 `LocalMemBar`；repeat-based Compare (Level0) 内部带 `Reg::LocalMemBar<VEC_STORE, SCALAR_LOAD>`（头 `.../reg_compute/dav_c310/kernel_reg_compute_vec_cmpsel_impl.h`）（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 本卡否定的是「从 `__aicore__` 常规函数直接调用」这一调用上下文；在 `__simd_vf__` 函数内调用同一 API 不受否定（softmax regbase 路径即如此）。
- 非 arch35 平台 (a2/a3) 未验证，`__simd_callee__` 调用约定可能不同；**未验证，推断可迁移**性低。
- 把整段算子包进 `__simd_vf__` 后 `Reg::LocalMemBar<VEC_STORE,VEC_LOAD>` 是否真能排空该 RAW 隐患，未在本 trace 验证（重写未完成即被回退），不作为本卡结论。

## 坏实践

- 在 `__aicore__` 常规 kernel 函数里直接调 `Reg::` (`__simd_callee__`) 命名空间的 sync API，不看调用上下文约束。
- 据 softmax regbase「能调 `Reg::LocalMemBar`」反推任意 kernel 都能直接调——softmax 是在 `__simd_vf__` 函数内调用，调用上下文不同。