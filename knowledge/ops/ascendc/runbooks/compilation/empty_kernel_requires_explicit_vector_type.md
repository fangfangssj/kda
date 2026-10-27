---
type: "Runbook"
title: "空 Ascend C kernel 骨架需显式标注 Vector 执行类型"
description: "纯占位 kernel 没有设备指令可供 Bisheng 自动推导任务类型时，显式添加 __vector__ 可关闭链接期 auto derivate failed。"
tags: ["task.debugging", "topic.compilation", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-14T00:44:51Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# 空 Ascend C kernel 骨架需显式标注 Vector 执行类型

## 现象

在 CANN 9.2.0、dav-2201 的 classic Ascend C 直调工程中，模板 kernel 只有参数消音，没有 Vector 或 Cube 操作。对象编译完成后，设备链接对各 dtype 实例报告 `kernel function type auto derivate failed`，且不生成共享库。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`__vector__`、`__aicore__`、`auto_derivate`、`Bisheng` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`platform_wide`，仅覆盖同工具链下缺少可推导设备指令的纯 Vector 空骨架。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

空函数体没有设备操作可供 Bisheng 自动判断 AIV、AIC 或 mix 任务类型。显式 `__vector__` 给出与计划实现一致的执行类型，因而不再依赖函数体推导；本次修复没有改变参数签名。

## 解决配方

下面是逐字恢复的完整 kernel 函数。若最终实现并非纯 Vector，不能照搬该限定符。

### before

```cpp
template <typename T>
__global__ __aicore__ void rms_norm_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                           int64_t numRows, int64_t numCol,
                                           int64_t rowStride, float epsilon,
                                           int64_t blockRowNum, int64_t tileRowNum)
{
    // Phase 6: batched per-row reduction via AscendC::Sum + Broadcast + Muls/Sqrt/Reciprocal.
    (void)x;
    (void)gamma;
    (void)y;
    (void)numRows;
    (void)numCol;
    (void)rowStride;
    (void)epsilon;
    (void)blockRowNum;
    (void)tileRowNum;
}
```

### after

```cpp
template <typename T>
__global__ __aicore__ __vector__ void rms_norm_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                           int64_t numRows, int64_t numCol,
                                           int64_t rowStride, float epsilon,
                                           int64_t blockRowNum, int64_t tileRowNum)
{
    // Phase 6: batched per-row reduction via AscendC::Sum + Broadcast + Muls/Sqrt/Reciprocal.
    (void)x;
    (void)gamma;
    (void)y;
    (void)numRows;
    (void)numCol;
    (void)rowStride;
    (void)epsilon;
    (void)blockRowNum;
    (void)tileRowNum;
}
```

## 设计决策

对计划为纯 Vector 的临时骨架，入口属性比制造无意义的数据搬运更直接。若骨架需要同时验证 GM/UB 数据通路，则可改用边界明确的最小真实设备主体，但该主体不能当作算子正确性实现。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：空 kernel 链接阶段报 kernel type auto-derive failed。（来源平台：a3；证据等级: tool_observed）。
- 修复后：显式 vector 类型后构建成功。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

本卡只证明空纯 Vector 骨架的构建闭环，不证明骨架具备目标算子语义。其他平台注册平台和 CANN 版本未验证；已有真实设备操作时应继续检查入口属性、参数 ABI 和实际执行单元。

## 坏实践

- 把 `__vector__` 机械添加到 Cube 或 mix kernel。
- 用“共享库可加载”代替目标算子的数值验证。
- 看到自动推导失败就先改参数 ABI，却不检查函数体是否完全为空。