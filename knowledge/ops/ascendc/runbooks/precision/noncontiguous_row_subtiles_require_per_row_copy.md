---
type: "Runbook"
title: "跨行子块误作连续 slab 导致分块结果错"
description: "当 nCur 小于物理行宽 N 时，rCur×nCur 子块在 GM 中带 stride，不能从首地址一次连续 DMA，应按行搬运或使用已核实的 2D stride DMA。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["row_dma"]
---

# 跨行子块误作连续 slab 导致分块结果错

## 现象

显式流水同步修复后，完整 pytest 仍固定剩下同一类中间轴 case：连续行或完整行路径都通过，只有把内层宽度切成多个 `nBlock` 的路径失败。定向重复运行显示 `values` 和 `indices` 都 mismatch 且不确定。此时代码从 `mBase + r0*N + n0` 起一次 `DataCopyPad` 搬运 `rCur*nCur` 个元素，并把 UB 解释为 `[rCur][nCur]` 连续二维数组。

## 适用场景

- 工程形态：Ascend C kernel 直调，Host 把逻辑张量折叠为 `[M,R,N]`，kernel 沿 R 扫描并沿 N 分块；
- API 路径：`GlobalTensor` + `DataCopyPad` + UB slab；
- 验证过的平台与版本：Ascend 950（dav-3510）/ CANN 9.0.1，与 frontmatter `platforms` 一致；
- 适用粒度：`operator_family`，适用于归约、扫描、窗口或其他把跨行二维子块搬入 UB 的 kernel；
- 路线优先级：`recommended_fastpath`，完整行走连续 slab，部分行走逐行 DMA，是本轨迹唯一完成全量正确性闭环的分派；
- 明确不适用：`nCur == N` 且 `n0 == 0` 的完整行，此时相邻 R 行在目标子块中确实连续，可保留批量 slab DMA。

## 根因

当 `nCur < N` 时相邻目标行首地址相差 `N` 个元素而 UB 目标行宽仅为 `nCur`，故从首地址连续搬运 `rCur*nCur` 元素会读入行尾间隙并漏掉后续行的目标列区间。

## 解决配方

先用地址公式判定连续性：目标元素地址为 `base + r*N + n`。只有 `n0=0 && nCur=N` 时，`r` 行尾的下一地址恰好等于下一行首，才能把 `rCur*nCur` 当一个连续 slab。否则逐行搬运 `[n0,n0+nCur)`；若使用 2D DMA，必须先从固定版本头文件核实 `srcStride` 的单位和字段语义。

### before

修复前写入快照；path=`operators/cummin/cummin.asc`，symbols=`KernelCummin<T>::ProcessBlock`/`CopyIn`，lines=134-161。两个完整函数支撑“无条件把跨行子块当连续 slab”的结论。

```cpp
    __aicore__ inline void ProcessBlock(int64_t m, int64_t n0, int64_t nCur)
    {
        int64_t rBlk = slabElems_ / nCur;
        if (rBlk < 1) { rBlk = 1; }
        if (rBlk > R_) { rBlk = R_; }

        int64_t mBase = m * R_ * N_;
        bool inited = false;

        for (int64_t r0 = 0; r0 < R_; r0 += rBlk) {
            int64_t rCur = rBlk;
            if (r0 + rCur > R_) { rCur = R_ - r0; }
            int64_t slab = rCur * nCur;
            CopyIn(mBase, r0, n0, slab);
            Compute(mBase, r0, rCur, nCur, slab, inited);
            inited = true;
            CopyOut(mBase, r0, n0, slab);
        }
    }

    __aicore__ inline void CopyIn(int64_t mBase, int64_t r0, int64_t n0, int64_t slab)
    {
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(slab * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        auto xLocal = inQueX_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[mBase + r0 * N_ + n0], cp, pad);
        inQueX_.EnQue(xLocal);
    }
```

### after

修复后写入快照；path=`operators/cummin/cummin.asc`，symbols=`KernelCummin<T>::Process`/`ProcessBlockPerRow`，lines=174-189,244-282。完整函数先按连续性分派，部分行路径每次只搬运一个实际连续行段。

```cpp
    __aicore__ inline void Process()
    {
        for (int64_t wb = workStart_; wb < workEnd_; ++wb) {
            int64_t m = wb / nBlockCount_;
            int64_t nb = wb - m * nBlockCount_;
            int64_t n0 = nb * nBlock_;
            int64_t nCur = nBlock_;
            if (n0 + nCur > N_) { nCur = N_ - n0; }
            if (nCur <= 0) { continue; }
            if (nCur == N_) {
                ProcessBlockContig(m, n0, nCur);
            } else {
                ProcessBlockPerRow(m, n0, nCur);
            }
        }
    }

    __aicore__ inline void ProcessBlockPerRow(int64_t m, int64_t n0, int64_t nCur)
    {
        int64_t mBase = m * R_ * N_;
        bool inited = false;
        for (int64_t r = 0; r < R_; ++r) {
            int64_t off = mBase + r * N_ + n0;
            // CopyIn (one row)
            auto xLocal = inQueX_.AllocTensor<T>();
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(nCur * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm_[off], cp, pad);
            inQueX_.EnQue(xLocal);
            // Compute
            xLocal = inQueX_.DeQue<T>();
            auto outVal = outQueVal_.AllocTensor<T>();
            auto outIdx = outQueIdx_.AllocTensor<int64_t>();
            pipe_barrier(PIPE_ALL);
            __ubuf__ T *xU = (__ubuf__ T *)xLocal.GetPhyAddr();
            __ubuf__ T *ovU = (__ubuf__ T *)outVal.GetPhyAddr();
            __ubuf__ int64_t *oiU = (__ubuf__ int64_t *)outIdx.GetPhyAddr();
            __ubuf__ T *rmU = (__ubuf__ T *)runMinBuf_.Get<T>().GetPhyAddr();
            __ubuf__ int64_t *riU = (__ubuf__ int64_t *)runIdxBuf_.Get<int64_t>().GetPhyAddr();
            cummin_scan_row<T>(xU, ovU, oiU, rmU, riU, r, nCur, inited);
            inited = true;
            pipe_barrier(PIPE_ALL);
            outQueVal_.EnQue(outVal);
            outQueIdx_.EnQue(outIdx);
            inQueX_.FreeTensor(xLocal);
            // CopyOut (one row)
            outVal = outQueVal_.DeQue<T>();
            outIdx = outQueIdx_.DeQue<int64_t>();
            AscendC::DataCopyExtParams cpv{1, static_cast<uint32_t>(nCur * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[off], outVal, cpv);
            AscendC::DataCopyExtParams cpi{1, static_cast<uint32_t>(nCur * sizeof(int64_t)), 0, 0, 0};
            AscendC::DataCopyPad(idxGm_[off], outIdx, cpi);
            outQueVal_.FreeTensor(outVal);
            outQueIdx_.FreeTensor(outIdx);
        }
    }
```

## 设计决策

- 保留 `nCur == N_` 的连续 slab 路径，因为完整行跨 R 拼接在 row-major GM 中确实连续，能减少 DMA 次数；
- `nCur < N_` 改用逐行 DMA，放弃未核实字段语义的 2D `srcStride` 一次搬运，优先保证正确性；
- 放弃把 `nBlock` 强行设为 `N`，因为大 N 会放大 UB running-state 与索引缓冲，且减少并行 work block；
- 当前混合分派属于 `recommended_fastpath`，但逐行路径的性能仍需 profiling 后再优化。

## 系统信息

- toolchain：CANN 9.0.1；
- 目标：Ascend 950 / dav-3510；
- 逻辑布局：row-major `[M,R,N]`，扫描轴 R 不切分，N 可按 `nBlock` 切分；
- 相关 API：`GlobalTensor::operator[]`、`DataCopyPad`、TQue；
- 代码文件：`operators/cummin/cummin.asc`。

## 预防

- 每次把二维/三维子块压成一次 DMA 前，写出相邻行地址差：若 `row_stride != copied_row_width`，它就不是连续块；
- 精度矩阵要同时覆盖 `nCur==N` 与 `nCur<N`，否则完整行测试会掩盖 stride 错误；
- 定向诊断先按 `nBlockCount` 分组，若只有 `nBlockCount>1` 失败，优先查跨行布局而不是 compute 公式。

## 验证信号

这条知识来自显式同步已经修复后的剩余失败，并由连续/逐行分派的单次完整重写闭环。

- 修复前，完整测试仅跨行分块类别失败，值和索引均 mismatch，重复运行也不一致（Ascend 950 / CANN 9.0.1，证据等级: tool_observed）；
- 改成逐行 DMA 后，干净构建的 pytest 为 53/53 通过（Ascend 950 / CANN 9.0.1，证据等级: tool_observed）；
- 最终 cann-bench 精度为 20/20 通过，values 误差为零且 indices exact match（Ascend 950 / CANN 9.0.1，证据等级: tool_observed）。

## 失效边界

- 逐行 DMA 是正确性基线，不保证对大 R 或窄行高效；
- 若采用 2D stride DMA 取代逐行循环，必须重新核实 stride 单位、最大 blockCount、blockLen 与尾块约束；
- 非 row-major 布局或有 padding 的物理布局不能直接套用 `off = m*R*N + r*N + n0`；
- A2/A3 未验证，推断可迁移但必须重新检查 DataCopy 参数契约并上板验证（未验证，推断可迁移）。

## 坏实践

- 看到元素总数是 `rCur*nCur` 就默认这些元素在 GM 连续；
- 用 `memcpy` 思维处理带 stride 的二维子块，却不写出每行首地址；
- 清理构建缓存后错误暂时变化就把根因归为 cache，而不比较失败是否只出现在 `nBlockCount>1`。

## 误诊

轨迹曾把该失败归因于 TQue 同步与 32B 最小搬运长度；显式屏障已经让完整行路径确定正确，而剩余失败严格跟随 `nCur<N`，最终按行修复后全量闭环，说明这里是独立的布局根因。