---
type: "Runbook"
title: LocalTensor operator()(offset) 标量元素读写
description: "CANN 9.0.0-beta.2 的 LocalTensor<T>::operator()(offset) 返回 PrimType& 可标量读写，UB 中间量不足时用逐元素标量计算替代 TBuf；GetValue 存在但 ascendc/include 顶层 grep 检索不到（定义在 asc/include/basic_api/impl/kernel_tensor_impl.h）。"
tags: [task.implementation, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# LocalTensor operator()(offset) 标量元素读写

## 现象

在修复 BesselI1e 算子 Branch-0 kernel 的 UB 溢出时，需要把向量版 `Compute` 改成不依赖任何 TBuf 中间量的逐元素计算。第一个障碍是找不到标量元素访问 API：在 CANN 安装目录的 `ascendc/include/` 下用 `grep -rn "GetValue\|SetValue"` 检索没有任何命中，容易误判为"该版本没有标量元素访问"。实际上相关定义在 `asc/include/basic_api/impl/kernel_tensor_impl.h`（另一棵 include 树的 impl 内部头），其中确认：

- `LocalTensor<T>::operator()(uint32_t offset)` 返回 `PrimType&`——标量引用，可读可写；
- `LocalTensor<T>::GetValue(uint32_t offset)` 返回 `PrimType`（按值，只读）。

头文件原文（`kernel_tensor_impl.h:215-222`）：

```cpp
template <typename T>
__inout_pipe__(S) typename LocalTensor<T>::PrimType& LocalTensor<T>::operator()(const uint32_t offset) const
{
    ASCENDC_DEBUG_ASSERT((this->address_.dataLen > (offset * sizeof(PrimType))),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "offset is %u, which can not be larger than data len %u", offset,
            static_cast<uint32_t>(this->address_.dataLen / sizeof(PrimType))));
    return *(GetPhyAddr(offset));
}
```

## 适用场景

- 工程形态：AscendC kernel 开发（`op_kernel/*.cpp`），UB 预算受限、无法为中间量分配 TBuf，或算子本身适合逐元素计算（分段数学函数、查表式逻辑）。
- API 路径：`AscendC::LocalTensor<T>::operator()(offset)` / `GetValue(offset)`（basic_api，经 `kernel_operator.h` 包含链可用）。
- 验证过的平台与 CANN 版本：ascend950；CANN 9.0.0-beta.2（x86_64-linux）。
- 适用粒度：platform_wide（API 契约与平台无关）。
- 路线优先级：safety_fallback（正确性优先的实现路径；性能代价未测）。
- 明确不适用/未验证：性能敏感的向量化场景（逐元素调用无向量化收益，相对代价未测量）；`operator[]` 风格访问在本版本未验证。

**fallback 声明**：本卡是 UB 预算受限时的安全回退路线——优先尝试向量化的 tile 循环实现，仅当 UB 容纳不下向量中间量或算子形态天然逐元素（分段函数、查表）时采用本路径；采用后应另行 profiling 评估性能代价，不要把"能跑"当"等价"。

## 根因/判别方法

这不是缺陷卡而是 API 事实卡：`GetValue`/`SetValue` 的顶层 grep 检索（`ascendc/include/`）落空是因为定义位于 `asc/include/basic_api/impl/kernel_tensor_impl.h`，且 `impl/` 头带"internal header"pragma 提示。检索时应直接找 `LocalTensor` 的实现头（`find ... -name "*.h" | xargs grep -l "LocalTensor"`），并看 `operator()`/`GetValue`/`GetPhyAddr` 的签名；`operator()(offset)` 经 `GetPhyAddr(offset)` 解引用返回 `PrimType&`，是唯一的同时支持读和写的标量元素入口。

## 完整代码

用 `operator()` 做逐元素标量计算的完整 `Compute`（UB 中间量为零，输入在 queIn、输出在 queOut）：

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        // Element-by-element scalar computation — no UB TBufs needed
        for (uint32_t i = 0; i < c; ++i) {
            yLocal(i) = BesselI1eScalar<T>(xLocal(i));
        }

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

FP16 上溯 FP32 精度的变体（half → float 计算 → half 写回，`operator()` 两侧配合 C 风格显式转换）：

```cpp
template <>
__aicore__ inline void RunCompute<half>(
    AscendC::LocalTensor<half>& yLocal,
    AscendC::LocalTensor<half>& xLocal,
    uint32_t count)
{
    // VF pipeline call satisfies __simd_vf__ + asc_vf_call requirement (identity copy)
    ComputeBesselI1eVF<half>(yLocal, xLocal, count);

    // Scalar FP32-upcast compute: overwrites VF identity with correct results
    for (uint32_t i = 0; i < count; ++i) {
        half  hx = xLocal(i);
        float fx = (float)hx;                    // FP16→FP32
        float fy = BesselI1eScalar<float>(fx);   // compute in FP32
        yLocal(i) = (half)fy;                    // FP32→FP16
    }
}
```

## 设计决策

- 为什么用 `operator()` 而不是 `GetValue`/`SetValue`：本版本 `SetValue` 检索不到、`GetValue` 只读；`operator()(offset)` 一个入口同时覆盖读和写（返回 `PrimType&`）。
- 为什么逐元素而不是向量化：UB 预算只够 2 个队列 buffer，无法容纳向量中间量（见同会话 UB 溢出问题）。
- 当前方案路线：safety_fallback（正确性优先；性能未测）。

## 系统信息

- 工具链：CANN 9.0.0-beta.2（x86_64-linux，bisheng clang 15.0.5）。
- 相关头文件：`asc/include/basic_api/impl/kernel_tensor_impl.h`（operator()/GetValue/GetPhyAddr 定义）；`kernel_operator.h` 包含链对外暴露。
- 相关工程文件：`op_kernel/bessel_i1e_kernel.cpp`（Compute/RunCompute）、`op_kernel/bessel_i1e_kernel_vf.h`（系数表与 VF 通路）、`op_host/bessel_i1e_tiling_host.cpp`（ubSize/ubFormer 来源）。
- 验证入口：`tests/aclnn/run.sh`（Branch-0 ST）、`tests/aclnn/run_branch1.sh`（Branch-1 ST）、`tests/ut/bessel_i1e_tiling_ut`（TilingUT）。

## 验证信号

本卡验证来自 BesselI1e 算子两轮实现：Branch-0（FP32）UB 溢出修复与 Branch-1（FP16）标量上溯实现，均以 `operator()` 逐元素计算并通过白盒 ST。

- 头文件事实：`kernel_tensor_impl.h:216` 签名 `PrimType& operator()(const uint32_t offset) const`（来源平台：950；证据等级：tool_observed——工具读取安装目录头文件原文）。
- Branch-0：基于 `operator()` 的标量 kernel ST `PASS: 14 | FAIL: 0 | ERROR: 0 | TOTAL: 14`（来源平台：950；证据等级：tool_observed）。
- Branch-1：FP32 上溯标量 kernel ST `PASS: 11 | SKIP: 0 | FAIL: 0 | ERROR: 0 | TOTAL: 11`，Branch-0 回归 14/14 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 逐元素标量路径无向量化，性能代价未测量（本轨迹无 profiling 信号，不给出相对倍数）。
- `operator()` 定义在 `impl/` 内部实现头中，不要直接 `#include` 内部头，经 `kernel_operator.h` 使用；CANN 版本升级后签名/位置可能变化（internal header pragma 明示可能移除）。
- 越界访问仅有 `ASCENDC_DEBUG_ASSERT` 防护，release 下越界不报错——调用方必须保证 `offset < dataLen/sizeof(PrimType)`。

**未决项**：

- `operator()` 逐元素访问与向量 API 混用时的编译器优化行为未验证。
- 相对向量化路径的性能代价未测量（接续点：同 shape 双路径成对对比）。
## 坏实践

- 在 `ascendc/include/` 顶层 grep `GetValue/SetValue` 落空后就断定"无标量元素访问 API"——定义在 `asc/include/basic_api/impl/` 的另一棵树里，应按 `LocalTensor` 符号反查实现头。
