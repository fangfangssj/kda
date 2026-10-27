---
type: "Runbook"
title: "CANN 9.0.1 SIMT 入口类型无法派生—改混合 VF"
description: "CANN 9.0.1 的混合 ASC 动态库中，纯 SIMT 全局入口无法稳定完成 kernel 类型派生；改用 Vector 入口承载 SIMT VF 和三参数 launch。"
tags: ["task.debugging", "topic.implementation.api_layer.simt_api", "topic.programming_model.development_mode.hybrid_programming"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# CANN 9.0.1 SIMT 入口类型无法派生—改混合 VF

## 现象

在 CANN 9.0.1 上把包含 Host 注册和 Device 模板函数的 `.asc` 文件编译成 Torch 扩展动态库时，代码最初把逐线程逻辑直接放进 `__global__ __aicore__ __launch_bounds__` 入口，并从 Host 使用四参数 `<<<grid, threads, 0, stream>>>` 启动。远端 `build` 先在链接阶段对每个模板实例报告：

```text
ld.lld: error: Error: the type of kernel function ... auto derivate failed,
please manually add the function type attribute
```

给入口补 `__vector__` 后，链接类型标记问题变成启动约定冲突：

```text
error: no matching function for call to 'gather_kernel'
candidate function template not viable: requires 11 arguments, but 12 were provided
```

也就是说，`__vector__` 让入口按 Vector kernel 的三参数执行配置解释，而调用点仍提供纯 SIMT 的 `grid + threads + dyn_ub + stream` 四参数配置。

## 适用场景

- 工程形态：CANN 9.0.1 的单文件 `.asc` Torch 自定义动态库，Host 与 Device 代码同编译单元；
- API 路径：`__global__ __aicore__ __vector__`、`__simt_vf__`、`asc_vf_call`、`cce::dim3` 和三参数 kernel launch；
- 验证过的平台与版本：Ascend 950 / dav-3510，CANN 9.0.1；与 frontmatter `platforms: [950]` 一致；
- 适用粒度：`exact_task`。只有同一编译形态和 CANN 版本的直接证据，跨算子复用必须重新构建验证；
- 路线优先级：`safety_fallback`。这是 fallback，性能不是最优，追求性能请用另一路线（见 `## 相关`）；本轨迹只验证了正确性兜底，没有验证性能最优；
- 明确不适用 / 未验证场景：CANN 9.1.0 及以后纯 SIMT 全局入口、独立 Device 编译单元、非 dav-3510 架构均未验证。

## 根因

纯 SIMT 全局入口的四参数启动形态与 CANN 9.0.1 混合 ASC 动态库所需的 kernel 类型标记和三参数 Vector 入口调用约定不匹配。

机制链如下：

1. 入口只有 `__global__ __aicore__ __launch_bounds__` 时，编译器可以生成对象，但链接器无法为模板实例派生 AI Core kernel 类型；
2. 直接加 `__vector__` 虽然显式标记了入口类型，却把执行配置切换为 Vector kernel 的 `<<<block_num, dyn_ub, stream>>>`；
3. 调用点继续传四参数时，多出来的 `threads` 被计入函数实参，因此模板诊断从 11 个函数参数变成了 12 个；
4. 正确分层是：三参数启动一个 `__global__ __aicore__ __vector__` AI Core 入口，再由入口用 SIMT 版 `asc_vf_call(cce::dim3(thread_count), ...)` 启动 `__simt_vf__` 线程函数。

## 解决配方

代码证据来自同一开发轨迹对 `operators/gather/gather.asc` 的真实写入；before 为完整写入态加随后精确签名替换后的失败工作树，after 为修复后工作树。失败态没有独立 40 位提交号；修复态在轨迹中只观测到短提交号 `f3a3abc`。以下代码按事件流逐字保留。

### before

路径：`operators/gather/gather.asc`；revision：未提交工作树；symbol：`gather_kernel<X_T, IDX_T>`；原文件行段：58–72。该完整函数支撑“纯 SIMT 全局入口无法派生 kernel 类型”的结论。

```cpp
template <typename X_T, typename IDX_T>
__global__ __aicore__ __launch_bounds__(1024) void gather_kernel(
    __gm__ const X_T* x_gm,
    __gm__ const IDX_T* index_gm,
    __gm__ X_T* y_gm,
    __gm__ const int64_t* strides_gm,
    int64_t totalLength,
    int32_t ndim,
    int32_t dim,
    int64_t xGatherDimSize)
{
    // Phase 6: strided indexed gather loop body.
    (void)x_gm; (void)index_gm; (void)y_gm; (void)strides_gm;
    (void)totalLength; (void)ndim; (void)dim; (void)xGatherDimSize;
}
```

路径：`operators/gather/gather.asc`；revision：未提交工作树；symbol：`op_gather::gather` 内的 `LAUNCH_GATHER`；原文件行段：151–155。该调用点支撑“四参数执行配置与后续 `__vector__` 类型不相容”的结论。

```cpp
#define LAUNCH_GATHER(X_T, IDX_T)                                                       \
    gather_kernel<X_T, IDX_T><<<static_cast<uint32_t>(gridDim),                         \
                                static_cast<uint32_t>(blockDim), 0, aclStream>>>(      \
        (__gm__ const X_T*)x_ptr, (__gm__ const IDX_T*)i_ptr, (__gm__ X_T*)y_ptr,      \
        (__gm__ const int64_t*)s_ptr, totalLength, ndim, static_cast<int32_t>(dim), xGatherDimSize)
```

### after

路径：`operators/gather/gather.asc`；revision：`f3a3abc`（轨迹观测到的短提交号）；symbol：文件级 include；原文件行段：52。该配置行支撑“入口调用的是 SIMT `asc_vf_call(dim3, ...)` 重载”的结论。

```cpp
#include "simt_api/asc_simt.h"
```

路径：`operators/gather/gather.asc`；revision：`f3a3abc`（轨迹观测到的短提交号）；symbols：`gather_vf<X_T, IDX_T>`、`gather_kernel<X_T, IDX_T>`；原文件行段：76–127。两个完整函数支撑“Vector 入口承载 SIMT VF”的结论。

```cpp
template <typename X_T, typename IDX_T>
__simt_vf__ __launch_bounds__(1024) inline void gather_vf(
    __gm__ const X_T* x_gm,
    __gm__ const IDX_T* index_gm,
    __gm__ X_T* y_gm,
    __gm__ const int64_t* strides_gm,
    int64_t totalLength,
    int32_t ndim,
    int32_t dim,
    int64_t xGatherDimSize)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
                  + static_cast<int64_t>(threadIdx.x);
    int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);

    for (int64_t o = tid; o < totalLength; o += stride) {
        int64_t idx = static_cast<int64_t>(index_gm[o]);
        if (idx < 0 || idx >= xGatherDimSize) {
            y_gm[o] = static_cast<X_T>(0);
            continue;
        }
        int64_t rem = o;
        int64_t xoff = 0;
        for (int32_t d = 0; d < ndim; ++d) {
            int64_t outStride = strides_gm[d];
            int64_t xStride = strides_gm[ndim + d];
            int64_t c = rem / outStride;
            rem -= c * outStride;
            if (d == dim) {
                c = idx;
            }
            xoff += c * xStride;
        }
        y_gm[o] = x_gm[xoff];
    }
}
```

同一修复态中的完整 AI Core 入口函数：

```cpp
template <typename X_T, typename IDX_T>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR strides,
    int64_t totalLength, int32_t ndim, int32_t dim, int64_t xGatherDimSize)
{
    constexpr uint32_t THREADS = static_cast<uint32_t>(GATHER_THREADS_PER_BLOCK);
    asc_vf_call<gather_vf<X_T, IDX_T>>(
        cce::dim3(THREADS),
        (__gm__ const X_T*)x, (__gm__ const IDX_T*)index, (__gm__ X_T*)y,
        (__gm__ const int64_t*)strides, totalLength, ndim, dim, xGatherDimSize);
}
```

路径：`operators/gather/gather.asc`；revision：`f3a3abc`（轨迹观测到的短提交号）；symbol：`op_gather::gather` 内的 `LAUNCH_GATHER`；原文件行段：211–215。该调用点支撑“Vector 入口必须使用三参数执行配置”的结论。

```cpp
#define LAUNCH_GATHER(X_T, IDX_T)                                                       \
    gather_kernel<X_T, IDX_T><<<static_cast<uint32_t>(gridDim), nullptr, aclStream>>>( \
        x_ptr, i_ptr, y_ptr, s_ptr, totalLength, ndim,                                  \
        static_cast<int32_t>(dim), xGatherDimSize);                                     \
    (void)block
```

## 设计决策

- 选择混合模型，是因为它同时满足 CANN 9.0.1 对 kernel 类型的显式标记和 SIMT 随机 GM 访问需求；
- 被放弃的“给纯 SIMT 入口直接补 `__vector__`”方案会改变 launch 契约，却没有把线程维移入 `asc_vf_call`，因此必然出现实参数量错位；
- 当前方案属于 `safety_fallback`：它恢复了可构建、可运行、正确的通用路径，但本轨迹的评测明确显示性能低于已有基线，不能标成 `recommended_fastpath`。

## 系统信息

- Toolchain：CANN 9.0.1，bisheng/ld.lld；
- 目标：Ascend950PR_957c，NpuArch=dav-3510；
- 工程形态：单文件 `.asc` 编译 Torch 自定义动态库；
- 相关文件：`operators/gather/gather.asc`、`operators/gather/CMakeLists.txt`；
- 构建前置：`set(CMAKE_ASC_ENABLE_SIMT ON)` 必须在 `find_package(ASC)` 前生效，并包含 `simt_api/asc_simt.h`。

## 预防

- 先按 CANN 版本确认 kernel 入口与 `<<<>>>` 参数个数，不要只看 `__global__` 就套用四参数启动；
- `__vector__` 不是给纯 SIMT 入口“补链接属性”的孤立修饰符；一旦采用它，就要同步改为三参数入口并把线程维放入 `asc_vf_call`；
- `asc_vf_call` 同名但有 Basic 与 SIMT 两套重载；检查 include 后再判断 `cce::dim3` 是否会被当成 VF 数据参数；
- 链接器出现 `auto derivate failed` 时，先检查入口执行模型、函数类型修饰符和 host launch 契约三者是否成套。

## 验证信号

来源是一次 Ascend 950 / CANN 9.0.1 自定义算子开发：在纯 SIMT 入口连续触发链接和调用约定错误后，改为混合 SIMT VF，并通过指定远端 helper 重新构建和上板验证。

- 远端构建输出 `[100%] Built target op_gather`，动态库成功链接（来源平台：950；证据等级：tool_observed）；
- 自定义动态库加载与注册接口测试 `1 passed`（来源平台：950；证据等级：tool_observed）；
- 完整 pytest 结果为 `201 passed, 18 skipped`，跳过项是测试矩阵显式不适用组合（来源平台：950；证据等级：tool_observed）；
- cann-bench 正确性评测 `20/20` 通过（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_gather。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_gather。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 该结论只在 CANN 9.0.1 的混合 ASC 动态库工程实测；CANN 9.1.0 及以后是否可直接使用纯 SIMT 四参数入口，未验证；
- A2/A3 不在本轨迹验证范围，也不应从 `platforms` 推断支持；
- 如果 Device kernel 独立编译、不是 `__vector__` 入口，或者构建系统已提供其他 kernel 类型标注，调用约定可能不同；
- 此配方只解决执行模型与构建契约，不解决随机访存路径的性能瓶颈。

## 坏实践

只给链接失败的入口补 `__vector__`、却保留四参数 launch。这样会把 `threads` 从执行配置错位成普通函数实参，形成“requires 11 arguments, but 12 were provided”的二次报错。

## 误诊

- `__aicore__` 缺失是更早的独立机械错误；补上后进入 kernel 类型派生失败，不能把它当成完整根因；
- `dim3` 命名空间和 SIMT header 重载是混合模型落地后的两个独立编译问题，本卡不把它们合并成纯入口类型派生的根因。