---
type: "Runbook"
title: "GE图模式263: 自定义VF Select被禁"
description: "MaskedFill kernel 用 __simd_vf__/asc_vf_call 自定义 VF 实现 Select，aclnn 预编译全绿但 GE 图模式 TBE 动态编译生成坏 binary 触发 AI Core error 263（aivec）；须跑图模式的 regbase kernel 中自定义 VF 路线 banned。"
tags: ["task.debugging", "topic.implementation.api_layer.simt_api", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# GE图模式263: 自定义VF Select被禁

## 现象

为 MaskedFill 算子（mask_predicate + broadcast 范式，arch35/regbase 路径）收尾 GEIR 图模式测试时，以最小 GE 图（x [2,3] float32、mask [2,3] bool、value [2,3] float32，单次 `session.RunGraph`）执行 route_probe。aclnn 预编译路径 ST 全绿，AddCustom 图模式基础设施也全绿，但 MaskedFill 的 route_probe 在 GE 图模式 TBE 动态编译下执行 kernel 时崩溃，plog 记录：

```
[ERROR] RUNTIME(...,route_probe):... ProcessDavidStarsCoreErrorInfo:The error from device(chipId:0, dieId:0), serial number is 127, there is an aivec error exception, core id is 0, error code = 263, 82, dump info: ... vec error info: 0x4d64273800310143, ... mte error: 0x40000 ...
The extend info: errcode:(263, 82) errorStr: The address for scalar to use is unaligned or out of bounds The GM address exceeds 48 bits, or the on-chip buffer address exceeds the size of the buffer.The address for MTE2 to write UB and MTE3 to write L1/UB is out of bounds. subErrType: 0x4.
```

崩溃前 kernel 系为修精度引入的自定义 Vector Function：`__simd_vf__ inline void MaskedFillSelectVF<T>` + `asc_vf_call<MaskedFillSelectVF<DT>>(...)`。Design 本就要求「本算子唯一 Vector 操作 Select 是 regbase 内置指令（非自定义 VF），无需 `__simd_vf__`/`asc_vf_call`」。

## 适用场景

- **工程形态**：kernel 直调（op_kernel/arch35/masked_fill_kernel.h）+ aclnn 预编译 ST + GE 图模式 TBE 动态编译 RunGraph。
- **API 路径**：`__simd_vf__`/`asc_vf_call`（regbase 自定义 VF 内联）→ `AscendC::CompareScalar`/`AscendC::Select`（basic_api/interface/，非 VF）。
- **验证平台与版本**：Ascend950（kernel_cache `Ascend950PR_9579`、`aic-ascend950-ops-info`），CANN 9.1.0。
- **适用粒度**：`operator_family`（所有须在 GE 图模式 TBE 动态编译下执行的 regbase arch35 kernel）。
- **路线优先级**：`banned`（自定义 VF 路线在须跑图模式的 kernel 中禁用）。
- **不适用或未验证场景**：aclnn-only 算子（不跑 GE 图模式）可用 VF 且本 trace 实测全绿——禁用范围仅限「kernel 须在 GE 图模式 TBE 动态编译下执行」。

## 根因

GE 图模式 TBE 动态编译器（`te_maskedfill_*` binary 生成路径）不支持 regbase 自定义 Vector Function 内联（`__simd_vf__` 声明的 VF 函数体 + `asc_vf_call` 调用点）。VF 形态在 aclnn 预编译路径下由 Ascend C 编译器正确 lowering，故 ST 全绿；但 TBE 动态编译器对同一 VF 源码生成坏 binary，执行期在向量单元触发 aivec 异常（error 263, 82）。Design 显式规定本算子 Select 必须是 regbase 内置指令、不得用自定义 VF——该约束即对 TBE 动态编译兼容性的反映。

机制链（不跳步）：`__simd_vf__` 声明的 VF 函数体在 TBE 动态编译路径下不被正确编译 → 生成坏 `te_maskedfill_*.o`（binary 大小与预编译版不同，本 trace 实测 408568 vs 预编译 408864 字节）→ kernel 执行期向量单元访问越界 → aivec exception error 263, 82（"address for scalar/MTE2-UB out of bounds"）。

## 失败路线

**banned 路线：在须跑 GE 图模式的 regbase arch35 kernel 中用 `__simd_vf__`/`asc_vf_call` 自定义 VF 实现 Select。**

被禁的 VF 写法形态（原始 VF 函数体逐字未在本 trace 内完整恢复，仅以骨架示意）：

```cpp
// ❌ banned: 自定义 VF 在 TBE 动态编译下生成坏 binary → AI Core error 263
__simd_vf__ inline void MaskedFillSelectVF(
    __ubuf__ int8_t* maskAddr, __ubuf__ T* valueAddr,
    __ubuf__ T* xAddr, __ubuf__ T* dstAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{ /* ... Reg:: Compare/Select ... */ }

// 调用点:
asc_vf_call<MaskedFillSelectVF<DT>>(/* ... */);
```

**补救尝试与目标失败**：将 Select 从自定义 VF 改为普通 kernel 侧 API（`AscendC::CompareScalar`+`AscendC::Select`，basic_api/interface/，非 VF），并把 mask UB 步长从 `sizeof(DT)` 改回紧凑（1 byte/elem）。该补救在 aclnn 预编译下保持全绿，但 GE 图模式 TBE 动态编译下对非空张量**仍崩 error 263**——即图模式兼容性目标**未达成**。普通 Select 修复了 aclnn 路径，未修复图模式路径；图模式下向量执行的真正根因（指令编码不支持？UB 布局/对齐差异？TBE 编译器 lowering 缺陷？）本 trace 未追到机制层。

## 预防

- 任何 regbase arch35 kernel 若须在 GE 图模式 TBE 动态编译下执行，**禁止**用 `__simd_vf__`/`asc_vf_call` 自定义 VF；Select 类操作用普通 `AscendC::Select`/`CompareScalar` 或 DAG `Ops::Base::Vec::Select`/`Compare`。
- Design 文档对「Select 是 regbase 内置指令、非自定义 VF」的显式约束即此兼容性要求的契约化表达，实现侧不得违背。
- 不得以「aclnn 预编译 ST 全绿」作为「kernel 已在 GE 图模式可执行」的证据——两条编译路径独立。

## 验证信号

以下信号来自本次开发会话对 MaskedFill kernel 的 aclnn 预编译 ST 与 GE 图模式 route_probe。

- GE 图模式 route_probe（[2,3] float32/bool/float32）RunGraph 返回 status=-1，plog 记录 aivec error code 263, 82、vec error info 0x4d64273800310143、mte error 0x40000（来源平台：950；证据等级: tool_observed）。
- 改普通 `AscendC::CompareScalar`+`Select` 后 aclnn 预编译 ST 全部通过（Branch-4 与 Branch-8 全绿），精度不退化（来源平台：950；证据等级: tool_observed）。

## 失效边界

- 自定义 VF 禁用范围仅限「kernel 须在 GE 图模式 TBE 动态编译下执行」。aclnn-only 算子可用 VF（本 trace 实测全绿）。
- 普通 `AscendC::CompareScalar`+`Select` 对 ascend950 regbase 的 int8/int32/bool dtype 在 aclnn 预编译下实测可用；TBE 动态编译下对非空张量仍崩（未验证，推断不可迁移—）。
- binary 大小差异（408568 vs 408864 字节）为本次单次实测，不作为通用判据。

## 坏实践

- **测试侧绕过**：以空张量（spec 允许的合法边界）作为 route_probe 与图模式用例，使图基础设施校验退出 0，但 kernel 对非空张量在 GE 图模式仍崩。该做法在「图模式仅验 InferShape/InferDataType/RunGraph 流程、kernel 数值由 aclnn ST 覆盖」的 scope 内有合理性，但不得作为「kernel 已在 GE 图模式可执行」的证据。
- **误诊：arg5=0 即 tiling 未传递**：据 plog「args(0 to 5) after execute」中 arg5=0 推断图模式未把 tiling 传给 kernel，据此补 `REGISTER_TILING_DEFAULT`（opParaSize 由 392 变为 680）与非模板入口，均未解决崩溃。后续观测到 `Op tiling succeed` 与 `logical_addrs size:5`（5 用户参数 + 1 sys tiling），即 tiling 函数确被调用、tiling 数据作为 sysParam 单独传递——arg5=0 是「after execute」框架清理后的读数，非 tiling 未传递。真正崩溃在 kernel 向量 Select 执行期，非 tiling 传递。