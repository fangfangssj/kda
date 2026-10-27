---
type: "Runbook"
title: "跨批首行漏比较—续批从 row 0 扫描"
description: "前缀归约续批把 row 0 当作 carry 覆盖，当前输入未参与比较；首批跳过 row 0，续批从 row 0 全量扫描。"
tags: ["task.debugging", "topic.compute.reduction", "topic.implementation.tiling", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 跨批首行漏比较—续批从 row 0 扫描

## 现象

在 `example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp` 调试沿归约轴分批的前缀最小值与索引时，单批长度恰好落在 UB 批容量内可以通过；一旦进入下一批，失败从续批首行开始。最小尾批只有一行时，输出 value/index 仍是前一批 carry，而 reference 会在当前输入更小时更新到续批首行。

失败既不是随机精度漂移，也不是只错索引：在可复现列上，候选 value 保留旧最小值，候选 index 保留旧行号，而 reference 同时更新为当前续批首行。

## 适用场景

- 工程形态：Ascend C kernel 直调精度联调；
- API 路径：RegBase 前缀归约，UB 分批保存 `carry value + carry index`；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；
- 适用粒度：`operator_family`，适用于带跨批状态的 prefix scan / cumulative reduction；
- 路线优先级：`recommended_fastpath`，首批与续批使用不同起始行语义；
- 明确不适用 / 未验证场景：批之间没有 carry 依赖的普通 reduction，以及 A2/A3 未验证。

## 根因

续批沿用“row 0 是初始 carry”的首批约定，先用先前增量 carry 覆盖当前 `yTensor[0]` 再只扫描 `rows - 1` 行，导致续批首个真实输入永远未与 carry 比较。 机制链：首批的 carry 来自本批 row 0，因此扫描范围是 `[1, rows)`；续批的 carry 已在独立临时缓冲中，因此本批 row 0 是新数据，扫描范围必须改为 `[0, rows)`。若尾批 `rows=1`，错误写法得到 `computeTimes=0`，恰好完全跳过该批。

## 解决配方

把“首批初始化”和“续批消费 carry”拆成两条显式分支：

- 首批：从 `yTensor[0]` 初始化 carry，`yAddr=yTensor+A`，`computeTimes=rows-1`，全局 index 起点为 1；
- 续批：保留 `yTensor[0]` 的新输入，`yAddr=yTensor`，`computeTimes=rows`，全局 index 起点为该批的首行；
- 每次 `CopyIn2D` 后统一执行 MTE2→V 同步，再进入 VF。

### before

证据定位：revision=轨迹工具写入前状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`ComputeR`；行段=完整函数；支撑结论=续批覆盖 row 0 且固定只扫描 `rows-1` 行。

```cpp
    __aicore__ inline void ComputeR()
    {
        int64_t R = tiling_.R, N = tiling_.N, A = A_;
        int64_t computeR = tiling_.computeR;
        int64_t splitR = tiling_.splitR;
        int64_t reservedR = tiling_.reservedR;
        int64_t regComputeTimes = tiling_.regComputeTimes;
        int64_t reservedRegTimes = tiling_.reservedRegTimes;
        int64_t dupCount = tiling_.dupCount;

        auto xTmpTensor = xTmpBuf_.Get<X>();
        auto argminTmpTensor = argminTmpBuf_.Get<int32_t>();

        for (int64_t m = 0; m < mCount_; m++) {
            int64_t mOffset = (mStart_ + m) * R * N;
            bool firstBatch = true;
            for (int64_t j = 0; j < splitR; j++) {
                int64_t rows = (j == splitR - 1 && reservedR > 0) ? reservedR : computeR;
                int64_t gmOffset = mOffset + j * computeR * N;

                auto yTensor = yBuf_.template Get<X>();
                auto argminTensor = argminBuf_.template Get<int32_t>();
                CopyIn2D(yTensor, gmOffset, rows, N);
                SyncMte2ToV(*pipe_);

                if (firstBatch) {
                    AscendC::DataCopyParams dcp{1, static_cast<uint16_t>(tiling_.dataCopyBlockLen), 0, 0};
                    AscendC::DataCopy(xTmpTensor, yTensor[0], dcp);
                    SyncMte2ToV(*pipe_);
                    AscendC::Duplicate(argminTmpTensor, static_cast<int32_t>(0), dupCount);
                    // Set row 0 index to 0 (matching ComputeM's pattern)
                    AscendC::Duplicate(argminTensor[0],
                                       static_cast<int32_t>(0), dupCount);
                    firstBatch = false;
                } else {
                    // Write carry from previous batch to row 0 of yTensor and argminTensor.
                    // Use V-pipeline Duplicate for argminTensor to avoid MTE2/V race
                    // with previous batch's Cast in CopyOutArgmin2D.
                    // For yTensor values, DataCopy UB->UB is on V after SyncMte2ToV.
                    AscendC::DataCopyParams dcp{1, static_cast<uint16_t>(tiling_.dataCopyBlockLen), 0, 0};
                    AscendC::DataCopy(yTensor[0], xTmpTensor, dcp);
                    AscendC::DataCopy(argminTensor[0], argminTmpTensor, dcp);
                    SyncMte2ToV(*pipe_);
                }

                __ubuf__ X* selectX = reinterpret_cast<__ubuf__ X*>(xTmpTensor.GetPhyAddr());
                __ubuf__ int32_t* selArg = reinterpret_cast<__ubuf__ int32_t*>(argminTmpTensor.GetPhyAddr());
                __ubuf__ X* yAddr = reinterpret_cast<__ubuf__ X*>(yTensor.GetPhyAddr()) + A;
                __ubuf__ int32_t* argAddr = reinterpret_cast<__ubuf__ int32_t*>(argminTensor.GetPhyAddr()) + A;

                int32_t duplicate = static_cast<int32_t>(j * computeR) + 1;
                asc_vf_call<DoCompareVF<X, (std::is_same_v<X, float> || std::is_same_v<X, half> || std::is_same_v<X, bfloat16_t>)>>(
                    selectX, selArg, yAddr, argAddr,
                    rows - 1, regComputeTimes,
                    static_cast<uint32_t>(reservedRegTimes),
                    A, duplicate, vRegSize_, dSize_);

                SyncVToMte3(*pipe_);
                CopyOut2D(yTensor, yGm_, gmOffset, rows, N);
                CopyOutArgmin2D(argminTensor, argminGm_, gmOffset, rows, N);
                SyncMte3ToMte2(*pipe_);
            }
        }
    }
```

### after

证据定位：revision=轨迹工具写入后状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`ComputeR`；行段=完整函数；支撑结论=首批从 row 1 开始，续批从 row 0 开始并保留当前输入。

```cpp
    __aicore__ inline void ComputeR()
    {
        int64_t R = tiling_.R, N = tiling_.N, A = A_;
        int64_t computeR = tiling_.computeR;
        int64_t splitR = tiling_.splitR;
        int64_t reservedR = tiling_.reservedR;
        int64_t regComputeTimes = tiling_.regComputeTimes;
        int64_t reservedRegTimes = tiling_.reservedRegTimes;
        int64_t dupCount = tiling_.dupCount;

        auto xTmpTensor = xTmpBuf_.Get<X>();
        auto argminTmpTensor = argminTmpBuf_.Get<int32_t>();

        for (int64_t m = 0; m < mCount_; m++) {
            int64_t mOffset = (mStart_ + m) * R * N;
            bool firstBatch = true;
            for (int64_t j = 0; j < splitR; j++) {
                int64_t rows = (j == splitR - 1 && reservedR > 0) ? reservedR : computeR;
                int64_t gmOffset = mOffset + j * computeR * N;

                auto yTensor = yBuf_.template Get<X>();
                auto argminTensor = argminBuf_.template Get<int32_t>();
                CopyIn2D(yTensor, gmOffset, rows, N);
                SyncMte2ToV(*pipe_);

                __ubuf__ X* selectX = reinterpret_cast<__ubuf__ X*>(xTmpTensor.GetPhyAddr());
                __ubuf__ int32_t* selArg = reinterpret_cast<__ubuf__ int32_t*>(argminTmpTensor.GetPhyAddr());

                int64_t computeTimes;
                int32_t duplicate;
                __ubuf__ X* yAddr;
                __ubuf__ int32_t* argAddr;

                if (firstBatch) {
                    // First batch: carry = x[0]. Process rows 1..rows-1.
                    // Row 0 is the carry itself (x[0] with idx 0).
                    AscendC::DataCopyParams dcp{1, static_cast<uint16_t>(tiling_.dataCopyBlockLen), 0, 0};
                    AscendC::DataCopy(xTmpTensor, yTensor[0], dcp);
                    SyncMte2ToV(*pipe_);
                    AscendC::Duplicate(argminTmpTensor, static_cast<int32_t>(0), dupCount);
                    AscendC::Duplicate(argminTensor[0],
                                       static_cast<int32_t>(0), dupCount);
                    yAddr = reinterpret_cast<__ubuf__ X*>(yTensor.GetPhyAddr()) + A;
                    argAddr = reinterpret_cast<__ubuf__ int32_t*>(argminTensor.GetPhyAddr()) + A;
                    computeTimes = rows - 1;
                    duplicate = static_cast<int32_t>(j * computeR) + 1;
                    firstBatch = false;
                } else {
                    // Non-first batch: carry from prev batch is in xTmpTensor.
                    // Process rows 0..rows-1. Row 0 is a data row (not carry).
                    yAddr = reinterpret_cast<__ubuf__ X*>(yTensor.GetPhyAddr());
                    argAddr = reinterpret_cast<__ubuf__ int32_t*>(argminTensor.GetPhyAddr());
                    computeTimes = rows;
                    duplicate = static_cast<int32_t>(j * computeR);
                }

                if (computeTimes > 0) {
                    asc_vf_call<DoCompareVF<X, (std::is_same_v<X, float> || std::is_same_v<X, half> || std::is_same_v<X, bfloat16_t>)>>(
                        selectX, selArg, yAddr, argAddr,
                        computeTimes, regComputeTimes,
                        static_cast<uint32_t>(reservedRegTimes),
                        A, duplicate, vRegSize_, dSize_);
                }

                SyncVToMte3(*pipe_);
                CopyOut2D(yTensor, yGm_, gmOffset, rows, N);
                CopyOutArgmin2D(argminTensor, argminGm_, gmOffset, rows, N);
                SyncMte3ToMte2(*pipe_);
            }
        }
    }
```

## 设计决策

- carry 保存在独立 `xTmpTensor/argminTmpTensor`，不再复制覆盖续批 row 0；
- 首批保留 row 0 直接作为输出，续批则把 row 0 当成普通数据交给同一个 `DoCompareVF`；
- 当前方案属于 `recommended_fastpath`；它没有增加专用单行分支，`rows=1` 由统一续批循环自然覆盖。

## 预防

- 为所有带跨批状态的 scan 写清楚“本批 row 0 是 carry 还是数据”；
- 边界测试至少覆盖单批、恰好满批、满批加一行以及多批带尾批；
- 同时核对 value 与 index，防止旧 carry 数值偶然等于 reference 时掩盖 index 错误；
- 审查 `duplicate` 的全局索引起点是否和 `yAddr` 的起始行一致。

## 验证信号

来源是前缀归约 kernel 的定向批边界实验与修复后的综合上板测试。

- 修复前，单批通过而续批首行起 values/indices 同时失败；单行尾批明确保留旧 carry（950，证据等级: tool_observed）；
- 修复后，批容量边界附近、多个长归约轴与尾批组合全部通过（950，证据等级: tool_observed）；
- 修复后的基础 dtype/语义测试和非对齐、1D、多维、负轴等边界测试全部通过（950，证据等级: tool_observed）。

## 失效边界

- 该配方假定 carry 已由上一批 VF 完整写回独立 UB 临时缓冲；
- 若采用双缓冲或跨核 carry，需要另外建立同步与可见性协议；
- A2、A3 未验证，推断可迁移但仍需上板确认。

## 坏实践

所有 batch 都无条件把 row 0 当 carry，并统一使用 `yTensor + A` / `rows - 1`；这种写法在单批测试中看似正确，进入续批才暴露。

## 误诊

早期尝试把先前增量 carry 直接复制到续批 row 0，能让部分数据通过，却会丢失本批原始 row 0；数据恰好不小于 carry 时会产生假阳性。