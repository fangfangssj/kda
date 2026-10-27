---
type: "Runbook"
title: "SIMT API 未进入 AscendC 命名空间导致编译失败"
description: "dav-3510 的 SIMT 设备函数位于 AscendC::Simt；在自定义命名空间内直接写 Simt:: 会触发未声明标识符，须引入 AscendC 或完整限定。"
tags: ["task.debugging", "topic.implementation.api_layer.simt_api"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# SIMT API 未进入 AscendC 命名空间导致编译失败

## 现象

在为 Ascend950 / dav-3510 编写单文件 direct-launch Gather 算子时，`gather.asc` 的 `gather_detail::GatherSimt` 是 `__simt_vf__` 函数，循环却在自定义命名空间内直接调用 `Simt::GetThreadIdx()` 和 `Simt::GetThreadNum()`。通过远程 helper 执行 CMake/make 后，BiSheng 编译器在这两个调用点分别报出：

```text
error: use of undeclared identifier 'Simt'; did you mean 'AscendC::Simt'?
```

同次构建最终以 `gmake: *** [Makefile:91: all] Error 2` 退出。

## 适用场景

- 工程形态：Ascend C kernel 直调；单个 `.asc` 同时包含 host 注册、`__global__ __aicore__ __vector__` 入口和 `__simt_vf__` 设备函数。
- API 路径：在自定义 C++ 命名空间内调用 `AscendC::Simt` 下的 `GetThreadIdx`、`GetThreadNum` 或其他 SIMT 设备函数。
- 验证过的平台与版本：Ascend950PR_957b / dav-3510，CANN 9.0.1；与 frontmatter `platforms: ['950']` 一致。
- 适用粒度：`operator_family`，适用于同一工具链下使用 SIMT API 的索引/随机访问类 kernel；其他 API 是否同属该命名空间仍须查声明。
- 路线优先级：`recommended_fastpath`。在局部实现中引入 `AscendC` 后继续使用简洁的 `Simt::`，或逐处写完整限定名，都是正确的编译路线；本轨迹验证的是前者。
- 明确不适用/未验证场景：未验证非 dav-3510 平台、不同 CANN 版本、完全不使用 SIMT API 的 SIMD/经典 Ascend C kernel。

## 根因

`GatherSimt` 位于 `gather_detail` 自定义命名空间，而 SIMT 接口的真实限定名是 `AscendC::Simt`，该作用域没有引入 `AscendC`，因此未限定的 `Simt::` 无法完成名字查找。

## 解决配方

在包含 SIMT 函数的最窄自定义命名空间中显式引入 `AscendC`，使函数体里的 `Simt::` 解析为 `AscendC::Simt`；若不希望使用 using-directive，则应逐处写完整限定名。下面的 before/after 均由归一化事件流中的完整文件写入与精确编辑原样恢复，未改写函数内容。

### before

代码证据：revision=`working-tree-before-first-build`；path=`operators/gather/gather.asc`；symbol=`gather_detail::GatherSimt`；lines=`34-108`。这是轨迹中完整 Write 保存的编译失败前工作树；轨迹未记录该状态的 40 位 Git commit 或 blob URL，精确内部事件锚点只保存在 manifest。

```cpp
namespace gather_detail {

constexpr static uint32_t GATHER_LAUNCH_BOUND = 2048;
constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
constexpr static int32_t MAX_RANK = 8;
// params layout (uint32): [0]=n [1]=dim [2]=xShapeDim
//   outStride[0..MAX_RANK-1] at [3 .. 3+MAX_RANK-1]
//   xStride [0..MAX_RANK-1] at [3+MAX_RANK .. 3+2*MAX_RANK-1]
constexpr static int32_t PARAMS_SIZE = 3 + 2 * MAX_RANK;

// ---------------------------------------------------------------------------
// SIMT vector function: one block processes [yIndexBase, yIndexBase+currentCoreElements).
// Threads stride by Simt::GetThreadNum() over the block's output range.
// ---------------------------------------------------------------------------
template <typename X_T, typename INDICES_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(GATHER_LAUNCH_BOUND) inline void GatherSimt(
    uint32_t yIndexBase, uint32_t currentCoreElements,
    __gm__ const uint32_t *params,
    __gm__ const INDICES_T *index,
    __gm__ const X_T *x,
    __gm__ volatile X_T *y)
{
    const uint32_t n = params[0];
    const uint32_t dim = params[1];
    const uint32_t xShapeDim = params[2];
    const __gm__ const uint32_t *outStride = params + 3;
    const __gm__ const uint32_t *xStride = params + 3 + MAX_RANK;

    for (uint32_t t = Simt::GetThreadIdx(); t < currentCoreElements; t += Simt::GetThreadNum()) {
        const uint32_t yIdx = yIndexBase + t;

        // Decompose yIdx into output coords via output contiguous strides, accumulating
        // baseSrc = sum_{k!=dim} c_k * xStride[k] using X's own strides (prefix-mismatch safe).
        uint32_t rem = yIdx;
        uint32_t baseSrc = 0;
        const uint32_t nm1 = n - 1;
        for (uint32_t k = 0; k < nm1; ++k) {
            const uint32_t osK = outStride[k];
            uint32_t cK;
            if (osK <= 1) {
                cK = rem;          // stride 1 (or degenerate): quotient == remainder
            } else {
                cK = rem / osK;    // plain integer division (GM-bound gather: not the bottleneck)
            }
            rem = rem - cK * osK;   // remainder after this axis
            if (k != dim) {
                baseSrc += cK * xStride[k];
            }
        }
        // last axis (k = n-1): outStride[n-1] == 1, coordinate == remainder
        if (nm1 != dim) {
            baseSrc += rem * xStride[nm1];
        }

        // Read index value at the same linear offset (index.shape == output.shape, contiguous).
        INDICES_T rawIdx = index[yIdx];
        int32_t idxVal = static_cast<int32_t>(rawIdx);   // sign-extended (int8/int32/int64 -> int32, value < 2^20)
        if (idxVal < 0) {
            idxVal += static_cast<int32_t>(xShapeDim);   // torch.gather negative-index wrap
        }
        bool oob = (idxVal < 0) || (idxVal >= static_cast<int32_t>(xShapeDim));
        if (oob) {
            y[yIdx] = static_cast<X_T>(0);
        } else {
            uint32_t srcOffset = baseSrc + static_cast<uint32_t>(idxVal) * xStride[dim];
            y[yIdx] = x[srcOffset];
        }
    }
}
```

### after

代码证据：revision=`working-tree-after-namespace-edit-before-059102e`；path=`operators/gather/gather.asc`；symbol=`gather_detail::GatherSimt`；lines=`34-110`。这是基于完整 Write 与精确 Edit 确定性重建的工作树；该编辑只在命名空间开头加入 `using namespace AscendC;`。轨迹未记录 40 位 Git commit 或 blob URL，精确内部事件锚点只保存在 manifest。

```cpp
namespace gather_detail {

using namespace AscendC;

constexpr static uint32_t GATHER_LAUNCH_BOUND = 2048;
constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
constexpr static int32_t MAX_RANK = 8;
// params layout (uint32): [0]=n [1]=dim [2]=xShapeDim
//   outStride[0..MAX_RANK-1] at [3 .. 3+MAX_RANK-1]
//   xStride [0..MAX_RANK-1] at [3+MAX_RANK .. 3+2*MAX_RANK-1]
constexpr static int32_t PARAMS_SIZE = 3 + 2 * MAX_RANK;

// ---------------------------------------------------------------------------
// SIMT vector function: one block processes [yIndexBase, yIndexBase+currentCoreElements).
// Threads stride by Simt::GetThreadNum() over the block's output range.
// ---------------------------------------------------------------------------
template <typename X_T, typename INDICES_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(GATHER_LAUNCH_BOUND) inline void GatherSimt(
    uint32_t yIndexBase, uint32_t currentCoreElements,
    __gm__ const uint32_t *params,
    __gm__ const INDICES_T *index,
    __gm__ const X_T *x,
    __gm__ volatile X_T *y)
{
    const uint32_t n = params[0];
    const uint32_t dim = params[1];
    const uint32_t xShapeDim = params[2];
    const __gm__ const uint32_t *outStride = params + 3;
    const __gm__ const uint32_t *xStride = params + 3 + MAX_RANK;

    for (uint32_t t = Simt::GetThreadIdx(); t < currentCoreElements; t += Simt::GetThreadNum()) {
        const uint32_t yIdx = yIndexBase + t;

        // Decompose yIdx into output coords via output contiguous strides, accumulating
        // baseSrc = sum_{k!=dim} c_k * xStride[k] using X's own strides (prefix-mismatch safe).
        uint32_t rem = yIdx;
        uint32_t baseSrc = 0;
        const uint32_t nm1 = n - 1;
        for (uint32_t k = 0; k < nm1; ++k) {
            const uint32_t osK = outStride[k];
            uint32_t cK;
            if (osK <= 1) {
                cK = rem;          // stride 1 (or degenerate): quotient == remainder
            } else {
                cK = rem / osK;    // plain integer division (GM-bound gather: not the bottleneck)
            }
            rem = rem - cK * osK;   // remainder after this axis
            if (k != dim) {
                baseSrc += cK * xStride[k];
            }
        }
        // last axis (k = n-1): outStride[n-1] == 1, coordinate == remainder
        if (nm1 != dim) {
            baseSrc += rem * xStride[nm1];
        }

        // Read index value at the same linear offset (index.shape == output.shape, contiguous).
        INDICES_T rawIdx = index[yIdx];
        int32_t idxVal = static_cast<int32_t>(rawIdx);   // sign-extended (int8/int32/int64 -> int32, value < 2^20)
        if (idxVal < 0) {
            idxVal += static_cast<int32_t>(xShapeDim);   // torch.gather negative-index wrap
        }
        bool oob = (idxVal < 0) || (idxVal >= static_cast<int32_t>(xShapeDim));
        if (oob) {
            y[yIdx] = static_cast<X_T>(0);
        } else {
            uint32_t srcOffset = baseSrc + static_cast<uint32_t>(idxVal) * xStride[dim];
            y[yIdx] = x[srcOffset];
        }
    }
}
```

## 设计决策

1. 选择在 `gather_detail` 作用域加入 `using namespace AscendC;`，因为同一作用域后续还会使用多个 Ascend C 类型/API，并且轨迹读取的生产 Gather 源码采用了相同的命名空间引入形态。
2. 当前方案标为 `recommended_fastpath`：它只是名字解析修复，不改变 kernel 数据通路或性能路线。

## 系统信息

- 工具链：CANN 9.0.1，远程 BiSheng/Ascend C 构建。
- SoC：Ascend950PR_957b，NpuArch=dav-3510。
- 工程：direct-launch Flash，单文件 `operators/gather/gather.asc`。
- 相关文件：本闭环只修改 `operators/gather/gather.asc`；构建由工程 `CMakeLists.txt` 驱动。

## 预防

- 在写 `__simt_vf__` 前先从目标 CANN 版本头文件确认 API 的完整限定名，不从示例中的省略写法推断作用域。
- 若复制的生产代码使用 `using namespace AscendC;`，同时复制其命名空间上下文；只复制函数体会丢失名字查找条件。
- 首次构建时保留编译器的 “did you mean” 候选，它通常直接给出真实命名空间。

## 验证信号

来源为一次 Ascend950 Gather direct-launch 开发会话：首次远程构建记录了名字查找错误，引入 `AscendC` 并完成同批独立编译修复后，远程 clean build、NPU 测试和最终评测均有工具输出。

- 首次构建在 `Simt::GetThreadIdx/GetThreadNum` 处报告两条未声明标识符错误并以 gmake Error 2 退出（来源平台：950；证据等级：tool_observed）。
- 修复后构建输出 `[100%] Linking ASC shared library libop_gather.so` 与 `[100%] Built target op_gather`（来源平台：950；证据等级：tool_observed）。
- 最终 clean build 再次输出 `[100%] Built target op_gather`，随后 NPU 测试为 `141 passed, 42 skipped`（来源平台：950；证据等级：tool_observed）。
- cann-bench 全部 `20/20` 用例通过，输出为 exact match 或误差为零（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 这条结论依赖 CANN 9.0.1 / dav-3510 中 SIMT API 位于 `AscendC::Simt`；其他 CANN 版本必须重新查头文件。
- 对没有自定义命名空间、已经完整限定 `AscendC::Simt`、或完全不使用 SIMT 的 kernel 不适用。
- a3、a2 是否存在同名 SIMT API 未验证；不得由本卡推断其可迁移性。

## 坏实践

- 只复制生产函数体中的 `Simt::` 调用，却遗漏外层 `using namespace AscendC;`，会让看似相同的代码在新工程中无法完成名字查找。
- 在错误出现后反复增删 include，而不先采纳编译器给出的 `AscendC::Simt` 限定提示，会偏离真实根因。

## 误诊

轨迹没有实际执行过其他误诊性修复；“缺少头文件”只是常见可能性，不能作为本次事实。
