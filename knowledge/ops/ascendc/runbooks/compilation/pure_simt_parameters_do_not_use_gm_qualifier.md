---
type: "Runbook"
title: "纯 SIMT kernel 形参 __gm__ 导致编译失败"
description: "CANN 9.0.1 的纯 SIMT 编译模式会把 __gm__ 展开为 device attribute；kernel 形参应使用普通类型化指针。"
tags: ["task.debugging", "topic.implementation.api_layer.simt_api", "topic.programming_model.language_extension"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 纯 SIMT kernel 形参 __gm__ 导致编译失败

## 现象

在 Ascend950 的 CANN 9.0.1 工程中，以 `--enable-simt` 编译
`operators/gather/gather.asc` 的纯 SIMT `__global__` kernel。设备函数已经从混合
VF 路线改成四参数 launch，但 BiSheng 在 kernel 形参处报错：

```text
#define __gm__ __attribute__((device))
                              ^
gather.asc:51:62: error: __constant__, __device__, and __managed__ are not allowed on non-static local variables
void gather_kernel(__gm__ const T *x, __gm__ const I *index, __gm__ T *y,
                                                             ^
gather.asc:52:20: error: __constant__, __device__, and __managed__ are not allowed on non-static local variables
                   __gm__ const int64_t *meta, int64_t totalElements)
```

报错发生在设备参数声明阶段，不是 gather 寻址、线程切分或精度逻辑造成的。

## 适用场景

- 工程形态：PyTorch 自定义算子的 `.asc` 单文件 Host+Device 构建；
- API 路径：`--enable-simt`、`__global__ __launch_bounds__`、四参数 `<<<grid, block, dyn, stream>>>`；
- 验证过的平台与版本：950 / CANN 9.0.1；
- 适用粒度：`platform_wide`，结论约束纯 SIMT kernel 的形参声明；
- 路线优先级：`recommended_fastpath`，纯 SIMT kernel 直接使用普通类型化指针；
- 明确不适用：经典 AscendC Tensor API 或 SIMD/SIMT 混合 VF 的地址空间声明；CANN 其他版本未验证。

## 根因

CANN 9.0.1 的纯 SIMT 编译前端把 `__gm__` 展开成变量级
`__attribute__((device))`，该 attribute 不能修饰 `__global__` 函数的非静态形参。

## 解决配方

保留纯 SIMT 的 `__global__` kernel 与寻址算法，只从函数形参移除 `__gm__`，让
普通 `const T *` / `T *` 承载 GM 地址。以下代码逐字恢复自轨迹写入；内部事件定位仅记录在
manifest。

### before

- revision：未提交的失败前 trace snapshot（规范化轨迹未提供完整 Git object id）；
- path：`operators/gather/gather.asc`；
- symbol：`gather_kernel<T, I>`；
- line：49–92；
- 支撑结论：`__gm__` 只出现在 kernel 形参，正对应编译器报错位置。

```cpp
template <typename T, typename I>
__global__ __launch_bounds__(THREAD_NUM)
void gather_kernel(__gm__ const T *x, __gm__ const I *index, __gm__ T *y,
                   __gm__ const int64_t *meta, int64_t totalElements)
{
    const int64_t ndim = meta[0];
    const int64_t dim = meta[1];
    const int64_t gatherDim = meta[2];

    int64_t outStride[8];
    int64_t xStride[8];
    for (int64_t j = 0; j < ndim && j < 8; ++j) {
        outStride[j] = meta[4 + j];
        xStride[j] = meta[4 + ndim + j];
    }

    int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
                  + static_cast<int64_t>(threadIdx.x);
    int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);

    for (int64_t yIdx = tid; yIdx < totalElements; yIdx += stride) {
        int64_t iv = static_cast<int64_t>(index[yIdx]);
        if (gatherDim <= 0) {
            continue;
        }
        // Clamp out-of-range indices (torch.gather declares OOB undefined; clamping
        // avoids a device-side OOB GM read while leaving in-range results exact).
        if (iv < 0) {
            iv = 0;
        } else if (iv >= gatherDim) {
            iv = gatherDim - 1;
        }

        int64_t rem = yIdx;
        int64_t xIdx = 0;
        for (int64_t j = 0; j < ndim && j < 8; ++j) {
            int64_t coord = (outStride[j] > 0) ? (rem / outStride[j]) : 0;
            rem -= coord * outStride[j];
            int64_t src = (j == dim) ? iv : coord;
            xIdx += src * xStride[j];
        }
        y[yIdx] = x[xIdx];
    }
}
```

### after

- revision：`c43e44b`（工具输出只记录短 revision，完整 object id 未在轨迹中恢复）；
- path：`operators/gather/gather.asc`；
- symbol：`gather_kernel<T, I>`；
- line：49–92；
- 支撑结论：仅改变形参限定，线程索引、地址计算和写回逻辑保持逐字一致。

```cpp
template <typename T, typename I>
__global__ __launch_bounds__(THREAD_NUM)
void gather_kernel(const T *x, const I *index, T *y,
                   const int64_t *meta, int64_t totalElements)
{
    const int64_t ndim = meta[0];
    const int64_t dim = meta[1];
    const int64_t gatherDim = meta[2];

    int64_t outStride[8];
    int64_t xStride[8];
    for (int64_t j = 0; j < ndim && j < 8; ++j) {
        outStride[j] = meta[4 + j];
        xStride[j] = meta[4 + ndim + j];
    }

    int64_t tid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x)
                  + static_cast<int64_t>(threadIdx.x);
    int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);

    for (int64_t yIdx = tid; yIdx < totalElements; yIdx += stride) {
        int64_t iv = static_cast<int64_t>(index[yIdx]);
        if (gatherDim <= 0) {
            continue;
        }
        // Clamp out-of-range indices (torch.gather declares OOB undefined; clamping
        // avoids a device-side OOB GM read while leaving in-range results exact).
        if (iv < 0) {
            iv = 0;
        } else if (iv >= gatherDim) {
            iv = gatherDim - 1;
        }

        int64_t rem = yIdx;
        int64_t xIdx = 0;
        for (int64_t j = 0; j < ndim && j < 8; ++j) {
            int64_t coord = (outStride[j] > 0) ? (rem / outStride[j]) : 0;
            rem -= coord * outStride[j];
            int64_t src = (j == dim) ? iv : coord;
            xIdx += src * xStride[j];
        }
        y[yIdx] = x[xIdx];
    }
}
```

## 设计决策

- 选择普通类型化指针，因为纯 SIMT launch 已经规定设备执行空间，不需要再用混合编程的 `__gm__` 宏标记形参；
- 未采用重新引入 `kernel_operator.h` 的方案：同一轨迹已经观察到它与 `--enable-simt` 的头文件路径不兼容；
- 当前方案属于 `recommended_fastpath`：它是该纯 SIMT 编译模式下通过语法门禁的直接写法，不代表整个算子的性能路线最优。

## 系统信息

- toolchain：CANN 9.0.1，BiSheng/Clang 15.0.5；
- platform：Ascend950PR 系列目标，真实 NPU 验证使用平台注册值 950；
- 相关文件：`operators/gather/gather.asc`、`operators/gather/CMakeLists.txt`；
- 构建开关：`--npu-arch=dav-3510 --enable-simt`。

## 预防

- 先判定当前是纯 SIMT 还是 Tensor API / 混合 VF 路线，再选择地址空间语法；
- 纯 SIMT `__global__` kernel 形参优先用普通、类型完整的指针；
- 看到错误文本点名 `__attribute__((device))` 时，先展开 `__gm__`，不要修改线程索引或数据算法。

## 验证信号

本知识来自一次 Ascend950 纯 SIMT 索引搬运算子的编译修复；移除参数限定后该错误不再出现，后续修复独立的 Host 分发问题后完成构建与上板测试。

- 远程构建输出 `[100%] Built target op_gather`（来源平台：950；证据等级：tool_observed）；
- 删除构建目录后重新构建，仍输出 `[100%] Built target op_gather`（来源平台：950；证据等级：tool_observed）；
- 完整 NPU pytest 输出 `36 passed`（来源平台：950；证据等级：tool_observed）。

## 失效边界

只在 CANN 9.0.1 的 `--enable-simt` 纯 SIMT 工程中验证。经典 AscendC Tensor API、
混合 VF 以及 CANN 其他版本的 `__gm__` 契约未验证；A2/A3 未验证，不能据此迁移。

## 坏实践

- 在纯 SIMT kernel 形参上照搬混合编程的 `__gm__`，会把地址空间意图错误地翻译成变量 attribute；
- 报错落在参数声明却去修改 grid-stride 或索引公式，不能消除语法阶段失败。

## 误诊

轨迹早期把纯 SIMT 与混合 VF 的地址宏视为可直接互换；编译器展开信息已经表明问题发生在
限定符语义，而不是计算逻辑。