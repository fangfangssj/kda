---
type: "Runbook"
title: "Reg 归约结果直接 Duplicate 导致末轴 Softmax 偏差"
description: "仅限 fp32 末轴 Softmax 的安全回退：归约结果经 32B UB 槽 GetValue 后标量 Duplicate；FP16/BF16 不适用，应走 UB 槽 LoadAlign 到 Reg 后广播。"
tags: ["task.debugging", "topic.compute.normalization", "topic.compute.reduction", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["slot_load_reg_broadcast"]
---

# Reg 归约结果直接 Duplicate 导致末轴 Softmax 偏差

## 现象

fp32、`post == 1` 的末轴 Softmax 已能编译和运行，但两个直接 NPU 精度探针分别出现约 `3.17e-1` 与 `8.58e-1` 的 `max_diff`。同一版本的非末轴 fp32 探针接近参考值，故问题收敛到末轴 Reg 归约与广播链路。

## 适用场景

- 算子语义：稳定 Softmax，末轴归约，`post == 1`；
- 数据类型：仅 fp32；
- 工程形态：Ascend950 Reg SIMD direct-launch kernel；
- 验证过的平台与 CANN 版本：Ascend950PR_957b / CANN 9.0.1；
- 适用粒度：`case_specific`；
- 路线优先级：`safety_fallback`；
- 明确未验证：fp16、bfloat16、非末轴以及性能收益。

## 根因

旧实现把 `ReduceMax`/`ReduceSum` 累积得到的单元素 `RegTensor<float> acc` 直接作为 `Duplicate` 的源，试图广播为全向量。该调用形态在这套 Reg 头文件与工具链中没有得到预期的“读取 lane 0 标量并广播”语义，导致后续减最大值和除总和使用了错误广播值。 轨迹没有编译器 overload dump，因此“具体选中了哪个 overload”仍是推断；但隔离 before/after 表明，去掉 RegTensor→Duplicate 直连，并改为 `StoreAlign` 到 32B UB 标量槽、`GetValue(0)` 读回 C++ 标量、再调用标量 `Duplicate` 后，两个末轴 fp32 探针同时恢复。

## 解决配方

把每次标量归约与后续向量广播拆成跨 `asc_vf_call` 的三个阶段：

1. `ReduceMax` 累积后把 lane 0 存入 32B UB slot；
2. caller 在 barrier 后用 `GetValue(0)` 取得 `float maxVal`，以标量 overload 广播，完成 `sub + exp + ReduceSum`，再把 sum 写回 slot；
3. 再次 `GetValue(0)` 取得 `float sumVal`，以标量 overload 广播并执行除法。

### before

代码来源：末轴 fp32 精度失败时的 `operators/softmax/softmax.asc` 快照；symbol `softmax_lastaxis_vf`；函数全体。

```cpp
// ===========================================================================
// Path A: stable softmax over a contiguous fp32 buffer of `count` elems
// (one pre group, post==1). max/sum kept as register scalars; exp retained in
// expBuf. Input/output are fp32 (dtype Cast handled by the kernel class).
// ===========================================================================
__simd_vf__ inline void softmax_lastaxis_vf(__ubuf__ float *xBuf, __ubuf__ float *expBuf,
                                            __ubuf__ float *yBuf, uint32_t count, uint32_t chunkCount)
{
    uint32_t oneE = SCALAR_ELEMS;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneE);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::RegTensor<float> xReg, eReg, oReg, maxBcast, sumBcast, acc, chunkReg;

    // ---- pass 1: reduce max ----
    AscendC::Reg::Duplicate<float>(acc, -INFINITY, scalarMask);
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VF32;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xBuf + off);
        AscendC::Reg::ReduceMax(chunkReg, xReg, mask);
        AscendC::Reg::Max(acc, acc, chunkReg, scalarMask);
    }
    AscendC::Reg::Duplicate<float>(maxBcast, acc, fullMask);

    // ---- pass 2: sub(max) + exp (store to expBuf) + reduce sum ----
    AscendC::Reg::Duplicate<float>(acc, 0.0f, scalarMask);
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VF32;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xBuf + off);
        AscendC::Reg::Sub(eReg, xReg, maxBcast, mask);
        AscendC::Reg::Exp(eReg, eReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(expBuf + off, eReg, mask);
        AscendC::Reg::ReduceSum(chunkReg, eReg, mask);
        AscendC::Reg::Add(acc, acc, chunkReg, scalarMask);
    }
    AscendC::Reg::Duplicate<float>(sumBcast, acc, fullMask);

    // ---- pass 3: div(sum) ----
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VF32;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(eReg, expBuf + off);
        AscendC::Reg::Div(oReg, eReg, sumBcast, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yBuf + off, oReg, mask);
    }
}
```

### after

代码来源：同一文件的 UB 标量槽修复快照；symbols `reduce_max_vf`、`sub_exp_sum_vf`、`div_vf` 与调用方 `Process`；以下均为函数全体。

```cpp
// ===========================================================================
// Path A: stable softmax over a contiguous fp32 buffer of `count` elems
// (one pre group, post==1). The per-group scalar max/sum are reduced to a 32B
// UB slot, read back via GetValue, and rebroadcast with the (unambiguous)
// scalar Duplicate overload. (The reg-reg Duplicate overload is ambiguous with
// the scalar overload in this header set, so we avoid it.)
// ===========================================================================
__simd_vf__ inline void reduce_max_vf(__ubuf__ float *xBuf, __ubuf__ float *slot,
                                      uint32_t count, uint32_t chunkCount)
{
    uint32_t oneE = SCALAR_ELEMS;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneE);
    AscendC::Reg::RegTensor<float> xReg, acc, chunkReg;
    AscendC::Reg::Duplicate<float>(acc, -INFINITY, scalarMask);
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VF32;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xBuf + off);
        AscendC::Reg::ReduceMax(chunkReg, xReg, mask);
        AscendC::Reg::Max(acc, acc, chunkReg, scalarMask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(slot, acc, scalarMask);
}

// sub(max) + exp -> expBuf; reduce sum -> slot.
__simd_vf__ inline void sub_exp_sum_vf(__ubuf__ float *xBuf, float maxVal,
                                       __ubuf__ float *expBuf, __ubuf__ float *slot,
                                       uint32_t count, uint32_t chunkCount)
{
    uint32_t oneE = SCALAR_ELEMS;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneE);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<float> xReg, eReg, maxBcast, acc, chunkReg;
    AscendC::Reg::Duplicate<float>(maxBcast, maxVal, fullMask);   // scalar overload: broadcast maxVal
    AscendC::Reg::Duplicate<float>(acc, 0.0f, scalarMask);
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VF32;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xBuf + off);
        AscendC::Reg::Sub(eReg, xReg, maxBcast, mask);
        AscendC::Reg::Exp(eReg, eReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(expBuf + off, eReg, mask);
        AscendC::Reg::ReduceSum(chunkReg, eReg, mask);
        AscendC::Reg::Add(acc, acc, chunkReg, scalarMask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(slot, acc, scalarMask);
}

// div(sum) expBuf -> yBuf.
__simd_vf__ inline void div_vf(__ubuf__ float *expBuf, float sumVal,
                               __ubuf__ float *yBuf, uint32_t count, uint32_t chunkCount)
{
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<float> eReg, oReg, sumBcast;
    AscendC::Reg::Duplicate<float>(sumBcast, sumVal, fullMask);    // scalar overload: broadcast sumVal
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VF32;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(eReg, expBuf + off);
        AscendC::Reg::Div(oReg, eReg, sumBcast, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yBuf + off, oReg, mask);
    }
}

__aicore__ inline void Process()
{
    uint32_t chunkCount = (rAligned_ + VF32 - 1) / VF32;
    uint32_t rAligned = rAligned_;
    AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(rLen_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};

    for (int64_t g = 0; g < myPre_; ++g) {
        int64_t off = (preStart_ + g) * rLen_;
        auto xLocal = inQueueX_.template AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[off], cp, pad);
        inQueueX_.EnQue(xLocal);
        auto xIn = inQueueX_.template DeQue<T>();
        auto yLocal = outQueueY_.template AllocTensor<T>();

        // get fp32 input pointer
        __ubuf__ float *xAddr;
        if constexpr (std::is_same<T, float>::value) {
            xAddr = (__ubuf__ float *)xIn.GetPhyAddr();
        } else {
            auto xF = xFp32Buf_.template Get<float>();
            AscendC::Cast(xF, xIn, AscendC::RoundMode::CAST_NONE, rAligned);
            AscendC::PipeBarrier<PIPE_V>();
            xAddr = (__ubuf__ float *)xF.GetPhyAddr();
        }
        auto expLocal = expBuf_.template Get<float>();
        auto slotLocal = slotBuf_.template Get<float>();
        __ubuf__ float *expAddr = (__ubuf__ float *)expLocal.GetPhyAddr();
        __ubuf__ float *slotAddr = (__ubuf__ float *)slotLocal.GetPhyAddr();

        // pass 1: reduce max -> slot
        asc_vf_call<reduce_max_vf>(xAddr, slotAddr, static_cast<uint32_t>(rLen_), chunkCount);
        AscendC::PipeBarrier<PIPE_V>();
        float maxVal = slotLocal.GetValue(0);

        // pass 2: sub(max) + exp -> expBuf, reduce sum -> slot
        asc_vf_call<sub_exp_sum_vf>(xAddr, maxVal, expAddr, slotAddr,
                                    static_cast<uint32_t>(rLen_), chunkCount);
        AscendC::PipeBarrier<PIPE_V>();
        float sumVal = slotLocal.GetValue(0);

        // pass 3: div(sum) -> yBuf (fp32), then cast to T
        if constexpr (std::is_same<T, float>::value) {
            __ubuf__ float *yAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
            asc_vf_call<div_vf>(expAddr, sumVal, yAddr, static_cast<uint32_t>(rLen_), chunkCount);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            auto yF = yFp32Buf_.template Get<float>();
            __ubuf__ float *yAddr = (__ubuf__ float *)yF.GetPhyAddr();
            asc_vf_call<div_vf>(expAddr, sumVal, yAddr, static_cast<uint32_t>(rLen_), chunkCount);
            AscendC::PipeBarrier<PIPE_V>();
            constexpr AscendC::RoundMode rm = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(yLocal, yF, rm, rAligned);
        }

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xIn);
        auto yOut = outQueueY_.template DeQue<T>();
        AscendC::DataCopyPad(yGm_[off], yOut, cp);
        outQueueY_.FreeTensor(yOut);
    }
}
```

## 设计决策

1. 归约结果先落 32B UB slot，显式形成可由 `GetValue(0)` 读取的标量边界；
2. 每个 VF 阶段后以 Vector pipe barrier 保证 slot 的生产与消费顺序；
3. 广播仅使用 `float` 标量源的 `Duplicate` 调用形态，避免 RegTensor 源语义歧义；
4. 该路线标记为 `safety_fallback`，因为跨 VF 调用的 UB 落盘与标量回读可能增加开销。

这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。

## 预防

- 不要把“单 lane RegTensor”自动等同于 C++ 标量；
- 对 Reg API overload 不明确的归约广播，先做孤立 NPU 数值探针；
- max 与 sum 两条归约链都要独立验证，避免只修一个广播点。

## 验证信号

来源说明：修改前后均在同一 Ascend950 远端环境直接构建并执行 fp32 Softmax 精度探针。

- 修改前，两个末轴 fp32 探针 `max_diff` 分别为 `0.3168267607688904` 与 `0.8580355644226074`（证据等级：tool_observed；来源平台：950）；
- 修改后完整构建输出 `[100%] Built target op_softmax`（证据等级：tool_observed；来源平台：950）；
- 修改后，同两个探针 `max_diff` 分别为 `2.9802322387695312e-08` 与 `0.0`（证据等级：tool_observed；来源平台：950）。

- 独立复现 R1：修后工具定向精度报告中，fp32 (16,16) 末轴 max_diff=2.9802322387695312e-08、(4,) 末轴 max_diff=0.0；这里只采用与 lane-0 广播修复直接对应的末轴信号（950，tool_observed）。
- 独立复现 R2：在 max/sum sweep 后分别广播 lane 0 后，同轮 pytest 为 34 passed，失败摘要已不含 test_softmax_last_axis；剩余 19 项均为非末轴、special/r1 的独立 507035/device 路径，不纳入本闭环（950，tool_observed）。

## 失效边界

- 只验证 fp32 且 `post == 1` 的末轴路径；同轮 fp16、bfloat16 与部分非末轴 case 仍失败，不得外推；
- 没有 profiling 证据，不能声称该方案性能最优；
- 具体 RegTensor→Duplicate overload 机制没有编译器 dump，根因中的 overload 描述保留为基于隔离实验的推断。

## 坏实践

- 把归约输出 RegTensor 直接当作确定的标量广播源，却不做数值探针；
- 只看构建成功就认为 reduction/broadcast 语义正确；
- 把本卡的 fp32 结论直接复制到 fp16、bfloat16 或非末轴路径。

## 误诊

该偏差不是通用 Softmax 公式错误：相同输入上的末轴结果在替换归约广播边界后恢复，而同轮非末轴 fp32 探针在修复前已经接近参考值。