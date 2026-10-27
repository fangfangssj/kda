---
type: "Runbook"
title: "ACL 507035 — 完整 a0 统计缓冲超过 UB"
description: "ARA 按完整尾轴乘积分配 fp32 max/sum 缓冲会在大 a0 时超过 UB；须对 a0 再切 tile 并逐 tile 完成三趟归约。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# ACL 507035 — 完整 a0 统计缓冲超过 UB

## 现象

在 `operators/softmax/softmax.asc` 的 ARA 路径通过地址对齐与多 group 精度测试后，pytest 仍在“归约轴之后维度乘积很大”的输入上报 NPU 设备错误 507035；相同 kernel 在较小 `a0` 上正确。查看 `Init` 可见，输入/输出 tile、`maxBuf` 和 `sumBuf` 都按完整 padded `a0` 分配，缓冲需求远大于目标 UB 容量。

## 适用场景

- 工程形态：Ascend C kernel 直调，非末轴 ARA 归约；
- API 路径：`TPipe::InitBuffer` 分配 a0 宽 fp32 统计量，max/sum/output 三趟遍历 r-tile；
- 验证过的平台与 CANN 版本：Ascend950PR_957b，CANN 9.0.1；
- 适用粒度：`operator_family`，适用于保留尾轴列并为每列维护 max、sum、mean 或 variance 的归约；
- 路线优先级：`recommended_fastpath`，a0 超出 UB 时双维 tiling 是已验证的正确路线；较小 a0 仍退化为单 a0-tile；
- 明确不适用：统计量是标量的末轴 AR 路径，或统计缓冲已由其他策略放入可容纳的存储层级。

## 根因

ARA 初版把 `maxBuf` 和 `sumBuf` 各分配为完整 `padA0 * sizeof(float)`，同时输入/输出 tile 也按 `tileR * padA0` 分配，使 UB 需求随完整 `a0` 线性增长；当 `2 * padA0 * sizeof(float)` 加上双队列 tile 超过 UB 时，kernel 在运行期失败。

UB 预算应写为：`2 * tileR * tileA0 * sizeof(T) + 2 * tileA0 * sizeof(float) + queue overhead <= UB`。因此不能只切 r；必须令 `tileA0` 受 UB 预算约束，并对每个 a0-tile 独立执行 max、sum、output 三趟。

## 解决配方

1. host tiling 增加 `tileA0`，以向量宽度对齐并设置容量上限；
2. `Init` 仅按 `tileR * tileA0` 分配输入/输出队列，统计 buffer 仅为 `tileA0` 宽；
3. `ProcessAra` 在 group 内增加 a0-tile 循环，每个 a0-tile 重新初始化 max/sum，然后遍历全部 r-tile 完成三趟；
4. 最后一个 a0-tile 用 `ta0` 计算有效 mask、用 `padTA0` 保持 UB 行对齐。

### before

revision: 未提交工作树快照（修复前）；path: `operators/softmax/softmax.asc`；symbols: `KernelSoftmax<T>::Init`, `KernelSoftmax<T>::ProcessAra`；line_range: 函数全文。代码支撑“完整 a0 分配缓冲且没有 a0 外层切片”的结论。

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t a1, int64_t r, int64_t a0,
                                int64_t groupsPerBlock, int64_t tileR)
    {
        coreIdx_ = AscendC::GetBlockIdx();
        groupStart_ = coreIdx_ * groupsPerBlock;
        groupEnd_ = groupStart_ + groupsPerBlock;
        if (groupEnd_ > a1) groupEnd_ = a1;
        if (groupStart_ > a1) groupStart_ = a1;
        r_ = r;
        a0_ = a0;
        tileR_ = tileR;
        // GM layout: a1 groups, each r*a0 contiguous elements.
        groupElem_ = r * a0;
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        padA0_ = (a0 + VLF - 1) / VLF * VLF;
        padR_ = (r + VLF - 1) / VLF * VLF;
        a0Chunks_ = (a0 + VLF - 1) / VLF;
        isAr_ = (a0 == 1);

        if (isAr_) {
            // AR: xQueue/yQueue hold one group (r elements, padded).
            pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, padR_ * sizeof(T));
            pipe_.InitBuffer(outQueueY_, PIPELINE_DEPTH, padR_ * sizeof(T));
        } else {
            // ARA: xQueue/yQueue hold one r-tile (tileR*a0), maxBuf/sumBuf a0-wide.
            pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, tileR_ * padA0_ * sizeof(T));
            pipe_.InitBuffer(outQueueY_, PIPELINE_DEPTH, tileR_ * padA0_ * sizeof(T));
            pipe_.InitBuffer(maxBuf_, padA0_ * sizeof(float));
            pipe_.InitBuffer(sumBuf_, padA0_ * sizeof(float));
        }
    }

    __aicore__ inline void ProcessAra()
    {
        auto maxBuf = maxBuf_.Get<float>();
        auto sumBuf = sumBuf_.Get<float>();
        int64_t nTiles = (r_ + tileR_ - 1) / tileR_;
        // maxBuf/sumBuf are PER-GROUP: re-init at the start of each group's max pass.
        for (int64_t g = groupStart_; g < groupEnd_; ++g) {
            int64_t gOff = g * groupElem_;
            ComputeAraInit<T>(maxBuf, padA0_, -std::numeric_limits<float>::infinity());
            AscendC::PipeBarrier<PIPE_V>();
            ComputeAraInit<T>(sumBuf, padA0_, 0.0f);
            AscendC::PipeBarrier<PIPE_V>();
            // pass 1: max over this group's r-tiles
            for (int64_t t = 0; t < nTiles; ++t) {
                int64_t rowBase = t * tileR_;
                int64_t rows = MINI64(tileR_, r_ - rowBase);
                int64_t gmBase = gOff + rowBase * a0_;
                auto xLocal = CopyInTileRows(gmBase, rows);
                ComputeAraMax<T>(maxBuf, xLocal, a0_, padA0_, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueueX_.FreeTensor(xLocal);
            }
            // pass 2: sum over this group's r-tiles
            for (int64_t t = 0; t < nTiles; ++t) {
                int64_t rowBase = t * tileR_;
                int64_t rows = MINI64(tileR_, r_ - rowBase);
                int64_t gmBase = gOff + rowBase * a0_;
                auto xLocal = CopyInTileRows(gmBase, rows);
                ComputeAraSum<T>(sumBuf, maxBuf, xLocal, a0_, padA0_, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueueX_.FreeTensor(xLocal);
            }
            // pass 3: output over this group's r-tiles
            for (int64_t t = 0; t < nTiles; ++t) {
                int64_t rowBase = t * tileR_;
                int64_t rows = MINI64(tileR_, r_ - rowBase);
                int64_t gmBase = gOff + rowBase * a0_;
                auto xLocal = CopyInTileRows(gmBase, rows);
                auto yLocal = outQueueY_.AllocTensor<T>();
                ComputeAraOut<T>(yLocal, sumBuf, maxBuf, xLocal, a0_, padA0_, rows);
                outQueueY_.EnQue(yLocal);
                inQueueX_.FreeTensor(xLocal);
                yLocal = outQueueY_.DeQue<T>();
                CopyOutTileRows(gmBase, rows, yLocal);
                outQueueY_.FreeTensor(yLocal);
            }
        }
    }
```

### after

revision: `eabb69e`（轨迹保留的短提交号）；path: `operators/softmax/softmax.asc`；symbols: `KernelSoftmax<T>::Init`, `KernelSoftmax<T>::ProcessAra`；line_range: 函数全文。代码支撑“缓冲仅按 tileA0 分配并逐 a0-tile 执行三趟”的结论。

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t a1, int64_t r, int64_t a0,
                                int64_t groupsPerBlock, int64_t tileR, int64_t tileA0)
    {
        coreIdx_ = AscendC::GetBlockIdx();
        groupStart_ = coreIdx_ * groupsPerBlock;
        groupEnd_ = groupStart_ + groupsPerBlock;
        if (groupEnd_ > a1) groupEnd_ = a1;
        if (groupStart_ > a1) groupStart_ = a1;
        r_ = r;
        a0_ = a0;
        tileR_ = tileR;
        tileA0_ = tileA0;  // per-a0-tile column width (VL-multiple, <= padA0)
        // GM layout: a1 groups, each r*a0 contiguous elements.
        groupElem_ = r * a0;
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        padA0_ = (a0 + VLF - 1) / VLF * VLF;
        padR_ = (r + VLF - 1) / VLF * VLF;
        a0Chunks_ = (a0 + VLF - 1) / VLF;
        isAr_ = (a0 == 1);

        if (isAr_) {
            // AR: xQueue/yQueue hold one group (r elements, padded).
            pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, padR_ * sizeof(T));
            pipe_.InitBuffer(outQueueY_, PIPELINE_DEPTH, padR_ * sizeof(T));
        } else {
            // ARA: xQueue/yQueue hold one (r-tile x a0-tile) block (stride tileA0_ per
            // row); maxBuf/sumBuf are tileA0_-wide (always fit UB even for huge a0).
            pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, tileR_ * tileA0_ * sizeof(T));
            pipe_.InitBuffer(outQueueY_, PIPELINE_DEPTH, tileR_ * tileA0_ * sizeof(T));
            pipe_.InitBuffer(maxBuf_, tileA0_ * sizeof(float));
            pipe_.InitBuffer(sumBuf_, tileA0_ * sizeof(float));
        }
    }

    __aicore__ inline void ProcessAra()
    {
        auto maxBuf = maxBuf_.Get<float>();
        auto sumBuf = sumBuf_.Get<float>();
        int64_t nTiles = (r_ + tileR_ - 1) / tileR_;
        // a0 is tiled into tileA0_-wide columns so maxBuf/sumBuf (tileA0_-wide) always
        // fit UB even for huge a0 (e.g. dim=0 on high-rank tensors). For each a0-tile
        // we run the full 3-pass (max/sum/out) over all r-tiles; maxBuf/sumBuf reset
        // per group and per a0-tile.
        for (int64_t g = groupStart_; g < groupEnd_; ++g) {
            int64_t gOff = g * groupElem_;
            for (int64_t a0t = 0; a0t < a0_; a0t += tileA0_) {
                int64_t ta0 = MINI64(tileA0_, a0_ - a0t);
                int64_t padTA0 = (ta0 + VLF - 1) / VLF * VLF;
                ComputeAraInit<T>(maxBuf, padTA0, -std::numeric_limits<float>::infinity());
                AscendC::PipeBarrier<PIPE_V>();
                ComputeAraInit<T>(sumBuf, padTA0, 0.0f);
                AscendC::PipeBarrier<PIPE_V>();
                // pass 1: max
                for (int64_t t = 0; t < nTiles; ++t) {
                    int64_t rowBase = t * tileR_;
                    int64_t rows = MINI64(tileR_, r_ - rowBase);
                    auto xLocal = CopyInTileRows(gOff, rowBase, rows, a0t, ta0, padTA0);
                    ComputeAraMax<T>(maxBuf, xLocal, ta0, padTA0, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueueX_.FreeTensor(xLocal);
                }
                // pass 2: sum
                for (int64_t t = 0; t < nTiles; ++t) {
                    int64_t rowBase = t * tileR_;
                    int64_t rows = MINI64(tileR_, r_ - rowBase);
                    auto xLocal = CopyInTileRows(gOff, rowBase, rows, a0t, ta0, padTA0);
                    ComputeAraSum<T>(sumBuf, maxBuf, xLocal, ta0, padTA0, rows);
                    AscendC::PipeBarrier<PIPE_V>();
                    inQueueX_.FreeTensor(xLocal);
                }
                // pass 3: output
                for (int64_t t = 0; t < nTiles; ++t) {
                    int64_t rowBase = t * tileR_;
                    int64_t rows = MINI64(tileR_, r_ - rowBase);
                    auto xLocal = CopyInTileRows(gOff, rowBase, rows, a0t, ta0, padTA0);
                    auto yLocal = outQueueY_.AllocTensor<T>();
                    ComputeAraOut<T>(yLocal, sumBuf, maxBuf, xLocal, ta0, padTA0, rows);
                    outQueueY_.EnQue(yLocal);
                    inQueueX_.FreeTensor(xLocal);
                    yLocal = outQueueY_.DeQue<T>();
                    CopyOutTileRows(gOff, rowBase, rows, a0t, ta0, padTA0, yLocal);
                    outQueueY_.FreeTensor(yLocal);
                }
            }
        }
    }
```

## 设计决策

- 同时切 r 和 a0，而不是只缩小 r-tile，因为 max/sum 统计 buffer 的主导项与 a0 成正比；
- 每个 a0-tile 单独完成三趟，避免把完整 a0 的统计量同时驻留 UB；
- 较小 a0 仍为单 tile，不额外改变正常路径；
- 当前方案属 `recommended_fastpath` 的容量安全路线；更激进的存储层级或跨核归约未验证。

## 系统信息

- 工具链：CANN 9.0.1 Ascend C / RegBase；
- 目标：Ascend950PR_957b；
- 相关文件：`operators/softmax/softmax.asc`；
- 相关符号：`Init`、`ProcessAra`、`tileA0_`、`maxBuf_`、`sumBuf_`。

## 预防

- 写 tiling 前列出每个 buffer 的元素数、dtype 字节数、队列深度和总 UB 公式；
- 不只检查输入 tile，还要检查长期驻留的 fp32 统计 buffer；
- 对非末轴归约覆盖“r 小但 a0 很大”和“r、a0 都大”两类容量用例；
- host tiling 与 kernel `InitBuffer` 使用同一个 `tileA0` 契约。

## 验证信号

本知识来自地址与多 group 修复后的剩余真机故障：较小 a0 已通过，而大尾轴乘积用例仍报 507035；加入 a0 tiling 后重建并重跑大 a0 探针与全套 pytest。

- 修复前大 a0 用例在 NPU 上报 507035（Ascend950PR_957b，证据等级: tool_observed）；
- 修复后多个大 a0 探针均通过，fp32 观测最大偏差不超过 `2.384e-07`（Ascend950PR_957b，证据等级: tool_observed）；
- 修复后完整 pytest 为 `88 passed, 6 skipped`，随后干净重建仍为 `88 passed, 6 skipped`（Ascend950PR_957b / CANN 9.0.1，证据等级: tool_observed）。

## 失效边界

- 本卡的具体 tile 上限来自该工程的 UB 预算；迁移时必须重新代入 dtype、队列深度和目标 UB，不能照抄常量；
- 末轴 AR 若规约长度本身超过 UB，需要独立的 r-recompute 路径，本轨迹未验证；
- A2/A3 未验证，不能写入 `platforms`；双维 tiling 原则未验证但推断可迁移。

## 坏实践

- 仅依据单个输入/输出 tile 估算 UB，漏算 fp32 max/sum 常驻缓冲；
- 只减小 `tileR`，却让完整 a0 统计量继续常驻 UB；
- 看到 507035 就只查地址对齐，不重新计算容量预算。

## 误诊

507035 既可能来自地址/pipe，也可能来自资源超限；本轨迹中地址对齐修复后较小 a0 已通过而大 a0 仍失败，这一差异把根因收敛到 UB 容量。
