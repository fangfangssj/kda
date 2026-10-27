---
type: "Runbook"
title: "VECTOR_REG_WIDTH 只能在 Device 作用域求值"
description: "Ascend950 Reg 向量宽度符号在全局 Host 翻译上下文不可见；应在 __aicore__ 或 __simd_vf__ 函数内定义局部常量。"
tags: ["task.debugging", "topic.programming_model.execution_model.simd"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# VECTOR_REG_WIDTH 只能在 Device 作用域求值

## 现象

`AscendC::VECTOR_REG_WIDTH` 在 `__simd_vf__` 探针内可用，但把同一表达式提到 `.asc` 文件全局作用域后，ASC 编译器报：

```text
error: no member named 'VECTOR_REG_WIDTH' in namespace 'AscendC'
constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);
                          ~~~~~~~~~^
```

这会让所有依赖全局 `VL_F` 的 wrapper 和 UB sizing 代码无法进入设备编译。

## 适用场景

- 工程形态：Ascend950 原生 `AscendC::Reg` kernel；
- API 路径：通过 `AscendC::VECTOR_REG_WIDTH` 推导 Reg lane 数；
- 验证过的平台与版本：950 / CANN 9.0.1；与 frontmatter `platforms` 一致；
- 适用粒度：`platform_wide`，针对 Host/Device 双翻译上下文中的符号可见性；
- 路线优先级：`recommended_fastpath`；
- 未验证场景：其他 CANN 版本、A2/A3、非 Reg 编程路径。

## 根因

CANN 9.0.1 dav-3510 的 Reg 常量在 device/VF 函数上下文内可见，将它提前到文件作用域会脱离该解析上下文。

## 解决配方

删除全局 `VL_F`，在每个需要它的 `__aicore__`/`__simd_vf__` 函数内求局部 `VL`。以下 helper 区域 verbatim 来自事件流。

### before

revision: 未提交 pre-fix 工作树快照
path: `operators/softmax/softmax.asc`
symbol: 全局 `VL_F` 与 `ChunkCount`
line segment: wrapper 区域全体（编译器定位原工作树第 279 行）

```cpp
// ===========================================================================
// __aicore__ wrappers (LocalTensor -> __ubuf__ -> asc_vf_call)
// ===========================================================================
constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);

static __aicore__ inline uint32_t ChunkCount(uint32_t count) {
    return (count + VL_F - 1) / VL_F;
}
```

### after

revision: fixed 工作树快照，后并入 `09b27d1`
path: `operators/softmax/softmax.asc`
symbol: `ChunkCount`
line segment: wrapper 区域全体

```cpp
// ===========================================================================
// __aicore__ wrappers (LocalTensor -> __ubuf__ -> asc_vf_call)
// ===========================================================================
// VECTOR_REG_WIDTH is only visible in device context, so VL is defined locally
// inside each __aicore__ / __simd_vf__ function (256B / 4B = 64 fp32 lanes).

static __aicore__ inline uint32_t ChunkCount(uint32_t count) {
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    return (count + VL - 1) / VL;
}
```

## 设计决策

- 选择在设备函数局部求值，而不是复制工具包探针观测到的常数值；这样向量宽度仍由目标编译环境决定；
- 未用预处理宏硬编码 lane 数，避免 dtype 与架构变化时失配；
- 当前方案属于 `recommended_fastpath`，局部 constexpr 不产生运行时开销。

## 预防

- 同一符号在 `__simd_vf__` 可用不代表它能出现在 `.asc` 全局 Host 区域；
- 设备属性、向量宽度和 intrinsic 常量优先在设备限定函数内求值；
- 需要 Host sizing 时使用明确的 Host API/编译配置输入，不跨翻译域偷用设备符号。

## 验证信号

来源为一次 Ascend950 Reg kernel 的工具包探针、远程构建和 NPU 验证。

- 工具包探针在设备相关头文件中观测到 `VECTOR_REG_WIDTH=256` 字节，设备函数内的同名表达式可通过早期 probe 构建（证据等级: tool_observed；来源平台: 950）；
- 全局 constexpr 版本稳定报 `no member named 'VECTOR_REG_WIDTH'`（证据等级: tool_observed；来源平台: 950）；
- 移入 `__aicore__`/`__simd_vf__` 函数后，远程构建输出 `[100%] Built target op_softmax`（证据等级: tool_observed；来源平台: 950）；
- 最终干净重建与 NPU pytest 为 58/58 通过（证据等级: tool_observed；来源平台: 950）。

- 独立复现 R1：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R3：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 该作用域行为只在 CANN 9.0.1 / 950 实测；其他版本可能提供 Host 可见的查询方式；
- A2/A3 未验证，推断可迁移但不得当作已验证；
- 设备局部求值不能直接给 Host 侧内存规划使用，Host 需要独立的信息来源。

## 坏实践

- 把探针观测到的 256 字节直接硬编码为全局常量；
- 因为一个设备函数编译过，就把所有设备符号上提到全局作用域。

## 误诊

这不是头文件完全缺少 `VECTOR_REG_WIDTH`；同一 trace 的设备探针可见，错误由全局 Host 名称查找触发。