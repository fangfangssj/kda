---
type: "Runbook"
title: "SIMT 内读块号致 Bisheng 崩溃—移到核入口"
description: "在 __simt_vf__ 内读取 BlockIdx/BlockNum 会使 Bisheng 前端异常终止；在 __aicore__ 调用侧切分并传参。"
tags: ["task.debugging", "topic.programming_model.execution_model.simt"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# SIMT 内读块号致 Bisheng 崩溃—移到核入口

## 现象

在给前缀扫描 kernel 的 SIMT 路径增加多核分片时，`SimtComputeVF` 被改成在 `__simt_vf__` 内直接调用 `AscendC::GetBlockNum()` 和 `AscendC::GetBlockIdx()`。远端编译没有给出普通 C++ 诊断，而是 Bisheng 前端直接异常终止：

```text
bisheng: error: unable to execute command: Aborted (core dumped)
bisheng: error: clang frontend command failed due to signal
```

即使回退当时被怀疑的 RegBase `UnPack` 实现，崩溃仍可复现；把块级元数据读取移回 `__aicore__` 调用侧后，kernel 目标恢复构建。

## 适用场景

- 工程形态：Ascend C kernel 直调构建；
- API 路径：`__aicore__` 调用 `cce::async_invoke<__simt_vf__>` 的混合路径；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；
- 适用粒度：`platform_wide`，约束与函数作用域和编译器 lowering 有关；
- 路线优先级：`recommended_fastpath`，块级分片在核入口计算，SIMT callee 只消费标量范围；
- 明确不适用 / 未验证场景：纯 `__aicore__` kernel、没有 SIMT callee 的路径，以及 A2/A3 未验证。

## 根因

块级运行时元数据 API 放进 `__simt_vf__` 作用域，超出了该 callee 的可 lowering 能力并触发 Bisheng 前端崩溃。

## 解决配方

在 `__aicore__` 的 `Init` 或调用函数中读取块号并计算 `mStart/mCount`，通过 `async_invoke` 的普通标量参数传给 `__simt_vf__`；SIMT callee 内只读取 thread id。

### before

证据定位：revision=轨迹工具写入前状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`SimtComputeVF`；行段=完整函数；支撑结论=SIMT callee 内直接读取块号并计算分片。

```cpp
template <typename X, typename ARGMIN, bool IS_FLOAT>
__simt_vf__ __launch_bounds__(THREAD_DIM) inline void SimtComputeVF(
    __gm__ X* xGm, __gm__ X* yGm, __gm__ ARGMIN* argminGm,
    int32_t M, int32_t R, int32_t N)
{
    // Multi-core partitioning: each block handles a contiguous range of M.
    // baseM = M / blockNum, extraM = M % blockNum.
    // Block b processes M range [mStart, mStart + mCount).
    int32_t blockNum = static_cast<int32_t>(AscendC::GetBlockNum());
    int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
    int32_t baseM = M / blockNum;
    int32_t extraM = M % blockNum;
    int32_t mStart = blockIdx * baseM + (blockIdx < extraM ? blockIdx : extraM);
    int32_t mCount = baseM + (blockIdx < extraM ? 1 : 0);

    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    // Total lanes for this block = mCount * N
    int32_t totalLanes = mCount * N;
    for (int32_t lane = tid; lane < totalLanes; lane += THREAD_DIM) {
        int32_t m = mStart + lane / N;
        int32_t n = lane % N;
        int32_t idx = m * R * N + n;
        X minVal = xGm[idx];
        ARGMIN minIdx = 0;
        yGm[idx] = minVal;
        argminGm[idx] = minIdx;
        for (int32_t j = 1; j < R; j++) {
            int32_t currIdx = idx + j * N;
            X curr = xGm[currIdx];
            bool takeCurr;
            if constexpr (IS_FLOAT) {
                takeCurr = (curr != curr) || (curr <= minVal);
            } else {
                takeCurr = (curr <= minVal);
            }
            if (takeCurr) {
                minVal = curr;
                minIdx = static_cast<ARGMIN>(j);
            }
            yGm[currIdx] = minVal;
            argminGm[currIdx] = minIdx;
        }
    }
}
```

证据定位：revision=轨迹工具写入前状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`ComputeSimT`；行段=完整函数；支撑结论=caller 只传入 `M/R/N`，没有传递已计算的块区间。

```cpp
    __aicore__ inline void ComputeSimT()
    {
        int32_t M = static_cast<int32_t>(tiling_.M);
        int32_t R = static_cast<int32_t>(tiling_.R);
        int32_t N = static_cast<int32_t>(tiling_.N);

        __gm__ X* xPtr = (__gm__ X*)xGm_.GetPhyAddr();
        __gm__ X* yPtr = (__gm__ X*)yGm_.GetPhyAddr();
        __gm__ ARGMIN* argminPtr = (__gm__ ARGMIN*)argminGm_.GetPhyAddr();

        if constexpr (std::is_same_v<X, int32_t>) {
            cce::async_invoke<SimtComputeVF<int32_t, int64_t, false>>(
                cce::dim3(THREAD_DIM), xPtr, yPtr, argminPtr, M, R, N);
        } else if constexpr (std::is_same_v<X, float>) {
            cce::async_invoke<SimtComputeVF<float, int64_t, true>>(
                cce::dim3(THREAD_DIM), xPtr, yPtr, argminPtr, M, R, N);
        } else if constexpr (std::is_same_v<X, half>) {
            cce::async_invoke<SimtComputeVF<half, int64_t, true>>(
                cce::dim3(THREAD_DIM), xPtr, yPtr, argminPtr, M, R, N);
        } else if constexpr (std::is_same_v<X, bfloat16_t>) {
            cce::async_invoke<SimtComputeVF<bfloat16_t, int64_t, true>>(
                cce::dim3(THREAD_DIM), xPtr, yPtr, argminPtr, M, R, N);
        }
    }
```

### after

证据定位：revision=轨迹工具写入后状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`SimtComputeVF`；行段=完整函数；支撑结论=SIMT callee 只消费 caller 传入的 `mStart/mCount`。

```cpp
template <typename X, typename ARGMIN, bool IS_FLOAT>
__simt_vf__ __launch_bounds__(THREAD_DIM) inline void SimtComputeVF(
    __gm__ X* xGm, __gm__ X* yGm, __gm__ ARGMIN* argminGm,
    int32_t R, int32_t N, int32_t mStart, int32_t mCount)
{
    // mStart/mCount are computed by the caller (ComputeSimT) using GetBlockIdx.
    // GetBlockIdx/GetBlockNum are NOT available inside __simt_vf__ functions.
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    // Total lanes for this block = mCount * N
    int32_t totalLanes = mCount * N;
    for (int32_t lane = tid; lane < totalLanes; lane += THREAD_DIM) {
        int32_t m = mStart + lane / N;
        int32_t n = lane % N;
        int32_t idx = m * R * N + n;
        X minVal = xGm[idx];
        ARGMIN minIdx = 0;
        yGm[idx] = minVal;
        argminGm[idx] = minIdx;
        for (int32_t j = 1; j < R; j++) {
            int32_t currIdx = idx + j * N;
            X curr = xGm[currIdx];
            bool takeCurr;
            if constexpr (IS_FLOAT) {
                takeCurr = (curr != curr) || (curr <= minVal);
            } else {
                takeCurr = (curr <= minVal);
            }
            if (takeCurr) {
                minVal = curr;
                minIdx = static_cast<ARGMIN>(j);
            }
            yGm[currIdx] = minVal;
            argminGm[currIdx] = minIdx;
        }
    }
}
```

证据定位：revision=轨迹工具写入后状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`ComputeSimT`；行段=完整函数；支撑结论=块区间在 `__aicore__` caller 中取得并作为标量参数传入。

```cpp
    __aicore__ inline void ComputeSimT()
    {
        int32_t R = static_cast<int32_t>(tiling_.R);
        int32_t N = static_cast<int32_t>(tiling_.N);
        // Multi-core: use mStart_/mCount_ (already computed in Init via GetBlockIdx)
        int32_t mStart = static_cast<int32_t>(mStart_);
        int32_t mCount = static_cast<int32_t>(mCount_);

        __gm__ X* xPtr = (__gm__ X*)xGm_.GetPhyAddr();
        __gm__ X* yPtr = (__gm__ X*)yGm_.GetPhyAddr();
        __gm__ ARGMIN* argminPtr = (__gm__ ARGMIN*)argminGm_.GetPhyAddr();

        if constexpr (std::is_same_v<X, int32_t>) {
            cce::async_invoke<SimtComputeVF<int32_t, int64_t, false>>(
                cce::dim3(THREAD_DIM), xPtr, yPtr, argminPtr, R, N, mStart, mCount);
        } else if constexpr (std::is_same_v<X, float>) {
            cce::async_invoke<SimtComputeVF<float, int64_t, true>>(
                cce::dim3(THREAD_DIM), xPtr, yPtr, argminPtr, R, N, mStart, mCount);
        } else if constexpr (std::is_same_v<X, half>) {
            cce::async_invoke<SimtComputeVF<half, int64_t, true>>(
                cce::dim3(THREAD_DIM), xPtr, yPtr, argminPtr, R, N, mStart, mCount);
        } else if constexpr (std::is_same_v<X, bfloat16_t>) {
            cce::async_invoke<SimtComputeVF<bfloat16_t, int64_t, true>>(
                cce::dim3(THREAD_DIM), xPtr, yPtr, argminPtr, R, N, mStart, mCount);
        }
    }
```

## 设计决策

- 块级切分属于 `__aicore__` 生命周期，线程级遍历属于 `__simt_vf__` 生命周期，二者以显式参数边界连接；
- 没有回退到单块执行，因为多核分片仍可由 caller 保留；
- 当前方案属于 `recommended_fastpath`；它保留多核并行且避开非法作用域，轨迹未提供另一条同等可编译路线。

## 预防

- 新增 SIMT helper 时，把 BlockIdx/BlockNum、TPipe 等核入口上下文 API 留在 `__aicore__`；
- `__simt_vf__` 参数显式携带工作区间，内部只使用 SIMT thread API 和传入的 GM 指针/标量；
- Bisheng 只报前端 signal 时，用“回退可疑改动后是否仍崩溃”的 A/B 方式缩小作用域。

## 验证信号

来源是同一次多核优化构建中，回退 RegBase 掩码尝试仍失败，以及随后只调整 SIMT caller/callee 边界的重编译。

- 回退 `UnPack` 后 Bisheng 仍以 signal 异常终止（950，证据等级: tool_observed）；
- 将 BlockIdx/BlockNum 消费移到 `__aicore__` caller 并传入 `mStart/mCount` 后，`all_kernels_obj`、`_C` 和 wheel 构建成功（950，证据等级: tool_observed）。

## 失效边界

- 结论只覆盖 CANN 9.0.1 的 950/DAV_3510 编译路径；
- 编译器未输出内部栈，因此没有独立最小复现定位到某个具体 lowering pass；
- A2、A3 未验证。

## 坏实践

为了减少参数数量，把 `GetBlockIdx/GetBlockNum` 下沉到 `__simt_vf__` 内部；这会把块级上下文与线程级 callee 作用域混在一起。

## 误诊

同一阶段同时引入了 RegBase 掩码变化，最初把 Bisheng 崩溃归因于 `UnPack`；回退掩码后错误不变，说明应继续检查 SIMT 作用域。
