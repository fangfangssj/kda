---
type: "Runbook"
title: "ARA 多 group 共用 max/sum 污染后续输出"
description: "ARA 三趟归约缓冲若只在 group 循环外初始化，后续 group 会继承前组 max/sum；应逐 group 重置再完成三趟。"
tags: ["task.debugging", "topic.precision", "topic.compute.reduction", "topic.compute.normalization"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# ARA 多 group 共用 max/sum 污染后续输出

## 现象

在 `operators/softmax/softmax.asc` 的 ARA 非末轴三趟实现中，地址对齐修复后，单 group 用例已经正确，但多 group 用例仍出现明显偏差：fp32 最大偏差达到 `2.767e-01`，fp16 与 bf16 也同时失败。失败与 dtype 无关，却与 group 数量相关，说明错误不是 Cast 或单趟向量计算，而是 group 之间共享了本应独立的归约状态。

## 适用场景

- 工程形态：Ascend C kernel 直调，单核循环处理多个独立归约 group；
- API 路径：常驻 UB 的 `maxBuf` / `sumBuf` 跨 r-tile 累加，三趟执行 max、sum、output；
- 验证过的平台与 CANN 版本：Ascend950PR_957b，CANN 9.0.1；
- 适用粒度：`operator_family`，适用于 softmax、归一化及所有按 group 独立归约却复用 UB 状态缓冲的 kernel；
- 路线优先级：`recommended_fastpath`，逐 group 状态隔离是正确性基线；
- 明确不适用：所有 group 本来就共享同一统计量的全局归约，或每个 group 使用独立缓冲区的实现。

## 根因

`maxBuf` 和 `sumBuf` 只在 `group` 循环外初始化一次，而三趟循环分别遍历所有 group，导致后续 group 的 max/sum 从前一 group 的统计量继续累积，破坏 group 独立性。

## 解决配方

把 group 循环提到最外层；进入每个 group 时将 `maxBuf` 重新置为 `-inf`、`sumBuf` 重新置零，然后仅为该 group 完成 max、sum、output 三趟。r-tile 之间允许累计，group 之间禁止累计。

### before

revision: 未提交工作树快照（修复前）；path: `operators/softmax/softmax.asc`；symbol: `KernelSoftmax<T>::ProcessAra`；line_range: 函数全文。代码支撑“初始化在 group 外、每一趟跨所有 group”的结论。

```cpp
    __aicore__ inline void ProcessAra()
    {
        auto maxBuf = maxBuf_.Get<float>();
        auto sumBuf = sumBuf_.Get<float>();
        ComputeAraInit<T>(maxBuf, padA0_, -std::numeric_limits<float>::infinity());
        AscendC::PipeBarrier<PIPE_V>();
        ComputeAraInit<T>(sumBuf, padA0_, 0.0f);
        AscendC::PipeBarrier<PIPE_V>();

        int64_t nTiles = (r_ + tileR_ - 1) / tileR_;
        // pass 1: max over all tiles (maxBuf accumulates across tiles)
        for (int64_t g = groupStart_; g < groupEnd_; ++g) {
            int64_t gOff = g * groupElem_;
            for (int64_t t = 0; t < nTiles; ++t) {
                int64_t rowBase = t * tileR_;
                int64_t rows = MINI64(tileR_, r_ - rowBase);
                int64_t gmBase = gOff + rowBase * a0_;
                auto xLocal = CopyInTileRows(gmBase, rows);
                ComputeAraMax<T>(maxBuf, xLocal, a0_, padA0_, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueueX_.FreeTensor(xLocal);
            }
        }
        // pass 2: sum (sumBuf accumulates across tiles)
        for (int64_t g = groupStart_; g < groupEnd_; ++g) {
            int64_t gOff = g * groupElem_;
            for (int64_t t = 0; t < nTiles; ++t) {
                int64_t rowBase = t * tileR_;
                int64_t rows = MINI64(tileR_, r_ - rowBase);
                int64_t gmBase = gOff + rowBase * a0_;
                auto xLocal = CopyInTileRows(gmBase, rows);
                ComputeAraSum<T>(sumBuf, maxBuf, xLocal, a0_, padA0_, rows);
                AscendC::PipeBarrier<PIPE_V>();
                inQueueX_.FreeTensor(xLocal);
            }
        }
        // pass 3: output
        for (int64_t g = groupStart_; g < groupEnd_; ++g) {
            int64_t gOff = g * groupElem_;
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

revision: 未提交工作树快照（group 隔离修复后、a0 双维 tiling 前）；path: `operators/softmax/softmax.asc`；symbol: `KernelSoftmax<T>::ProcessAra`；line_range: 函数全文。代码支撑“每个 group 重置状态并完成自己的三趟”的结论。

```cpp
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

## 设计决策

- 选择 group 外层、pass 内层的循环次序，使两个常驻 UB 缓冲只表示当前 group；
- 未为每个 group 分配独立 `maxBuf/sumBuf`，因为逐组复用同一组 buffer 已能隔离状态且节省 UB；
- 当前方案属 `recommended_fastpath`：只移动循环和初始化边界，不增加 GM 中间结果。

## 系统信息

- 工具链：CANN 9.0.1 Ascend C / RegBase；
- 目标：Ascend950PR_957b；
- 相关文件：`operators/softmax/softmax.asc`；
- 相关符号：`ProcessAra`、`maxBuf_`、`sumBuf_`。

## 预防

- 为每个长期复用的 UB buffer 写清生命周期键：per-tile、per-group、per-core 或 global；
- 代码评审时把初始化位置与生命周期键对照；
- 同时测试单 group 与多 group，避免单 group 偶然掩盖状态泄漏；
- 三趟算法优先采用 `for group { init; pass1; pass2; pass3; }` 的结构。

## 验证信号

本知识来自 ARA 地址对齐修复后的下一轮真机精度探针：单 group 已通过，多 group 仍失败；只调整 group/三趟循环边界后再次构建和测试。

- 修复前多 group fp32、fp16、bf16 最大偏差分别达到 `2.767e-01`、`7.910e-02`、`4.761e-02`（Ascend950PR_957b，证据等级: tool_observed）；
- 修复后同类多 group 探针三种 dtype 均通过，观测最大偏差不超过 `4.883e-04`（Ascend950PR_957b，证据等级: tool_observed）；
- 修复后工程重新构建成功（Ascend950PR_957b / CANN 9.0.1，证据等级: tool_observed）。

## 失效边界

- 若算法语义要求多个 group 共享统计量，本卡的逐组重置会改变语义；
- 若每 group 已有独立 buffer，则只需验证其索引，不必移动循环；
- A2/A3 未验证，不能写入 `platforms`；状态生命周期原则未验证但推断可迁移。

## 坏实践

- 因 buffer 只由一个 core 使用，就把它误当作只需初始化一次；core 私有不等于 group 私有；
- 按 pass 组织外层循环，却忘记每个 pass 中访问的是不同 group；
- 多 group 失败时先怀疑 dtype 或 `Exp`，不比较单 group 对照。

## 误诊

地址对齐问题修复后仍有精度错，若继续沿对齐方向加 barrier 会偏离根因；单 group 正确、多 group 错是状态生命周期泄漏的更强信号。