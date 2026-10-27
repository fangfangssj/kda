---
type: Runbook
title: 分核 partial 的初始化、参与者集合与最终归并
description: 用普通 UnsortedSegmentSum 的 int64 两阶段实现说明分片布局、空核和初始化顺序；按源码观察给出采用检查，不冒充修复轨迹或设备归因结果。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
status: draft
sources: []
created_at: '2026-10-17T18:12:04Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# 分核 partial 的初始化、参与者集合与最终归并

## 适用场景

多个工作项各自产生局部结果，再由独立归并阶段读取全部 partial。以普通 UnsortedSegmentSum 的小输出 int64 路线为结构实例：`total=ns*te`，partial 布局为 `[numCores,total]`。该路线与 DMA 原子写回不同，不假定单核 barrier 能替代跨阶段完成。

## 三个同时成立的合同

1. **已定义值**：归并阶段会读到的每个 partial 元素都有定义。工作核早退时，它的分片也不能留下未初始化值。
2. **一致参与者**：生产的核数、partial 分片 stride 和归并循环的 `numCores` 必须一致；只改变 launch 或分核而不改消费者会破坏布局。
3. **完成顺序**：初始化完成后才能生产，生产完成后才能归并。下面只确认源码把三次 launch 排在同一个 stream 参数下；实际后端执行顺序仍需运行环境合同和验证，不能从任意单核流水屏障推导出来。

## 实际 host 链与设备方法

原 host 的 int64 分支以 `total` 在 `[1,10000]` 内为其小输出候选 gate，随后计算 accum 与 merge 的 tiling。这个固定阈值属于来源实现，不是通用 UB 容量界。下列是分支内部完整的生产、初始化和归并调用单元，删除了注释中的平台及经验值域断言。

```cpp
        constexpr int64_t INT64_SMALL_OUT_MAX = 10000;
        bool useI64Correct = (total >= 1 && total <= INT64_SMALL_OUT_MAX);
        if (useI64Correct) {
            int64_t nb_a, rpc_i, ch_i;
            std::tie(nb_a, rpc_i, ch_i) = calc_uss_i64_accum_tiling(N, te, num_segments);
            int64_t nb_m, slice_m;
            std::tie(nb_m, slice_m) = calc_uss_i64_merge_tiling(total, nb_a);
            auto partial = at::empty({nb_a * total}, data.options().dtype(torch::kInt64));
            auto partial_ptr = (GM_ADDR)partial.data_ptr();
            int64_t nb_p, blk_p;
            std::tie(nb_p, blk_p) = calc_uss_memset0_tiling(partial.nbytes());
            auto call = [=]() -> int {
                launch_uss_memset0(partial_ptr, partial.nbytes(), nb_p, blk_p, stream);
                if (i64) launch_uss_i64_accum_i64(data_ptr, seg_ptr, partial_ptr, N, te, num_segments, nb_a, rpc_i, ch_i, stream);
                else     launch_uss_i64_accum_i32(data_ptr, seg_ptr, partial_ptr, N, te, num_segments, nb_a, rpc_i, ch_i, stream);
                launch_uss_i64_merge(partial_ptr, y_ptr, total, nb_a, nb_m, slice_m, stream);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("UnsortedSegmentSum", call);
            return y;
        }
```

`KernelI64Accum::Init` 把当前核的 `partialGm_` 绑定到 `partial + core*(ns*te)`，分配局部 accumulator 及输入/索引队列。完整 `Process` 如下；读输入与索引的队列交接、局部清零、非法 segment 跳过和最终发布都保留。

```cpp
    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t rowStart = blockIdx * rpc_;
        if (rowStart >= N_) return;
        int64_t rowEnd = rowStart + rpc_;
        if (rowEnd > N_) rowEnd = N_;
        if (rowEnd <= rowStart) return;
        int64_t accElems = ns_ * te_;
        auto acc = accQ_.AllocTensor<int64_t>();
        AscendC::Duplicate(acc.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                           static_cast<int32_t>(2 * accElems));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyPadExtParams<int64_t> dpd{false, 0, 0, 0};
        AscendC::DataCopyPadExtParams<I> spd{false, 0, 0, 0};
        const int64_t CH = dcr_;
        for (int64_t rc0 = rowStart; rc0 < rowEnd; rc0 += CH) {
            int64_t rcn = (rc0 + CH <= rowEnd) ? CH : (rowEnd - rc0);
            auto dataRaw = dataQ_.AllocTensor<int64_t>();
            AscendC::DataCopyExtParams dcp{1, static_cast<uint32_t>(rcn * te_ * 8), 0, 0, 0};
            AscendC::DataCopyPad(dataRaw, dataGm_[rc0 * te_], dcp, dpd);
            dataQ_.EnQue(dataRaw);
            dataRaw = dataQ_.DeQue<int64_t>();
            auto segLocal = segQ_.AllocTensor<I>();
            AscendC::DataCopyExtParams scp{1, static_cast<uint32_t>(rcn * static_cast<int64_t>(sizeof(I))), 0, 0, 0};
            AscendC::DataCopyPad(segLocal, segGm_[rc0], scp, spd);
            segQ_.EnQue(segLocal);
            segLocal = segQ_.DeQue<I>();
            for (int64_t r = 0; r < rcn; ++r) {
                I sg = segLocal.GetValue(static_cast<uint32_t>(r));
                if (sg < 0 || static_cast<int64_t>(sg) >= ns_) continue;
                int64_t base = static_cast<int64_t>(sg) * te_;
                int64_t dbase = r * te_;
                int64_t j = 0;
                for (; j + 4 <= te_; j += 4) {
                    int64_t v0 = dataRaw.GetValue(static_cast<uint32_t>(dbase + j));
                    int64_t v1 = dataRaw.GetValue(static_cast<uint32_t>(dbase + j + 1));
                    int64_t v2 = dataRaw.GetValue(static_cast<uint32_t>(dbase + j + 2));
                    int64_t v3 = dataRaw.GetValue(static_cast<uint32_t>(dbase + j + 3));
                    int64_t c0 = acc.GetValue(static_cast<uint32_t>(base + j));
                    int64_t c1 = acc.GetValue(static_cast<uint32_t>(base + j + 1));
                    int64_t c2 = acc.GetValue(static_cast<uint32_t>(base + j + 2));
                    int64_t c3 = acc.GetValue(static_cast<uint32_t>(base + j + 3));
                    acc.SetValue(static_cast<uint32_t>(base + j), c0 + v0);
                    acc.SetValue(static_cast<uint32_t>(base + j + 1), c1 + v1);
                    acc.SetValue(static_cast<uint32_t>(base + j + 2), c2 + v2);
                    acc.SetValue(static_cast<uint32_t>(base + j + 3), c3 + v3);
                }
                for (; j < te_; ++j) {
                    int64_t v = dataRaw.GetValue(static_cast<uint32_t>(dbase + j));
                    int64_t c = acc.GetValue(static_cast<uint32_t>(base + j));
                    acc.SetValue(static_cast<uint32_t>(base + j), c + v);
                }
            }
            dataQ_.FreeTensor(dataRaw);
            segQ_.FreeTensor(segLocal);
        }
        AscendC::DataCopyExtParams op{1, static_cast<uint32_t>(accElems * 8), 0, 0, 0};
        AscendC::DataCopyPad(partialGm_[0], acc, op);
        accQ_.FreeTensor(acc);
    }
```

`KernelI64Merge::Init` 绑定整个 partial 和输出，设置 `total_/nc_/slice_` 并分配两个 slice 大小的队列。完整归并方法按 `c=0..nc_-1` 读取同一输出 slice，累加后只写一次该 slice。

```cpp
    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t j0 = blockIdx * slice_;
        if (j0 >= total_) return;
        int64_t j1 = j0 + slice_;
        if (j1 > total_) j1 = total_;
        int64_t cnt = j1 - j0;
        if (cnt <= 0) return;
        auto acc = accQ_.AllocTensor<int64_t>();
        AscendC::Duplicate(acc.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                           static_cast<int32_t>(2 * cnt));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyPadExtParams<int64_t> dpd{false, 0, 0, 0};
        for (int64_t c = 0; c < nc_; ++c) {
            auto buf = bufQ_.AllocTensor<int64_t>();
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * 8), 0, 0, 0};
            AscendC::DataCopyPad(buf, partialGm_[c * total_ + j0], cp, dpd);
            bufQ_.EnQue(buf);
            buf = bufQ_.DeQue<int64_t>();
            int64_t j = 0;
            for (; j + 4 <= cnt; j += 4) {
                int64_t v0 = buf.GetValue(static_cast<uint32_t>(j));
                int64_t v1 = buf.GetValue(static_cast<uint32_t>(j + 1));
                int64_t v2 = buf.GetValue(static_cast<uint32_t>(j + 2));
                int64_t v3 = buf.GetValue(static_cast<uint32_t>(j + 3));
                int64_t a0 = acc.GetValue(static_cast<uint32_t>(j));
                int64_t a1 = acc.GetValue(static_cast<uint32_t>(j + 1));
                int64_t a2 = acc.GetValue(static_cast<uint32_t>(j + 2));
                int64_t a3 = acc.GetValue(static_cast<uint32_t>(j + 3));
                acc.SetValue(static_cast<uint32_t>(j), a0 + v0);
                acc.SetValue(static_cast<uint32_t>(j + 1), a1 + v1);
                acc.SetValue(static_cast<uint32_t>(j + 2), a2 + v2);
                acc.SetValue(static_cast<uint32_t>(j + 3), a3 + v3);
            }
            for (; j < cnt; ++j) {
                int64_t v = buf.GetValue(static_cast<uint32_t>(j));
                int64_t a = acc.GetValue(static_cast<uint32_t>(j));
                acc.SetValue(static_cast<uint32_t>(j), a + v);
            }
            bufQ_.FreeTensor(buf);
        }
        AscendC::DataCopyExtParams op{1, static_cast<uint32_t>(cnt * 8), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[j0], acc, op);
        accQ_.FreeTensor(acc);
    }
```

## 空参与者的静态推导

源 tiling 使用 `numBlocks=min(coreNum,N)`（空输入另有最小值处理）及 `rpc=ceil(N/numBlocks)`，设备按 `rowStart=blockIdx*rpc` 早退。即使 `numBlocks<=N`，也可能有空核：令 `N=5,numBlocks=4,rpc=2`，第 3 号核的 `rowStart=6`，不会发布分片；归并仍循环 4 个分片。这是源码整数算式的静态推导，不是本轮设备故障复现。host 的 partial 预清零使未生产分片的加法贡献为零。

同一实现里的清零 kernel 使用零 tile，按真实 byte count 覆盖目标区间，再由 host 在 accum 之前调用；其完整方法为：

```cpp
__global__ __aicore__ __vector__ void uss_memset0_kernel(
    GM_ADDR dst, int64_t nbytes, int64_t blockBytes)
{
    constexpr int64_t TILE_BYTES = 16384;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> q;
    pipe.InitBuffer(q, 1, TILE_BYTES);
    AscendC::LocalTensor<uint8_t> zero_tile = q.AllocTensor<uint8_t>();
    AscendC::Duplicate<int32_t>(zero_tile.ReinterpretCast<int32_t>(), 0,
        static_cast<int32_t>(TILE_BYTES / sizeof(int32_t)));
    q.EnQue(zero_tile);
    AscendC::LocalTensor<uint8_t> z = q.DeQue<uint8_t>();
    AscendC::GlobalTensor<uint8_t> dst_gm;
    dst_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dst));
    int64_t my_start = static_cast<int64_t>(AscendC::GetBlockIdx()) * blockBytes;
    if (my_start >= nbytes) { q.FreeTensor(z); return; }
    int64_t my_end = (my_start + blockBytes <= nbytes) ? (my_start + blockBytes) : nbytes;
    for (int64_t off = my_start; off < my_end; off += TILE_BYTES) {
        int64_t cur = (off + TILE_BYTES <= my_end) ? TILE_BYTES : (my_end - off);
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(cur), 0, 0, 0};
        AscendC::DataCopyPad(dst_gm[off], z, cp);
    }
    q.FreeTensor(z);
}
```

## 验证信号

公开来源见[普通 UnsortedSegmentSum 必要源码与脱敏单条历史记录](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_pr136_uss_evidence_20261019.zip)，ZIP SHA-256 为 `9cd2857dd078ffd847b15acf8787ecea6bc58c7a5e6b53822c6a5e95aaeef260`。包内 `unsorted_segment_sum/source_excerpt.txt` 按原文件及行号保留完整必要类、方法、常量和 host 调用分支；`source_binding.json` 区分完整原件、所选范围和去注释摘录的 SHA。该公开附件提供核验入口，不以 PR #136 自述代替技术证据。

kernel 原件为 `csrc/ops/unsorted_segment_sum/op_kernel/unsorted_segment_sum_kernel.cpp`，完整文件 SHA-256 为 `8dd8ada92f1aec8bca4971ebceafa9ab79b7801309264f869a4ad4f0766d81de`；host 原件为 `csrc/ops/unsorted_segment_sum/op_plugin/unsorted_segment_sum_plugin.cpp`，SHA-256 为 `9dbb6a94f8f905c2f537010446a6bc36942cb8ecad4af30aecda322ff8c71436`。可逐项审阅 `core*(ns*te)` 的分片绑定、`nb_a*total` 的分配长度、merge 的 `nb_a`，以及同一 `stream` 参数从 host 经 launch wrapper 传入清零、accum、merge 的静态链；这不独立证明后端实际完成顺序。

包内 `unsorted_segment_sum/historical_record.json` 的 `/benchmark` 和 `/result` 是普通 `level2/unsorted_segment_sum` 的单条脱敏历史投影：`official-tasks 1.1.0`、`hardware=910c`、`hardware_label=A3`、编译通过，`accuracy_passed_cases=passed_cases=total_cases=80`。原提取回执将该源码内包与记录关联至同一导出提交；`source_binding.json` 的 `/published_historical_record/artifact_binding` 给出来源包、两份源码、原报告及原始单记录的 SHA。原始单记录 SHA 为 `b7d8f3b65d124da9d5ec8a93384cd7b0b2b68df7f62268d44c09268ed5464fbb`，公开脱敏记录文件 SHA 为 `136f5c205a8d45990c043bd73a493b979bfa88a7732d5dcaf253b811dfb2b6f9`；两者不是同一文件。

80/80 只表示原整 artifact 的历史观察，不独立证明本卡 int64 小输出分支、每条索引 dtype 分支或空核输入得到覆盖，也不认证原执行加载的二进制、完整构建依赖或 CANN 版本。本轮没有编译、设备运行或性能实验；`platforms: []` 保留未将整体结果提升为本模式独立平台验证的边界。本卡不声称已复现“漏清零失败→加清零通过”，也不采用源码注释里的硬件缺陷、原子支持或经验值域结论。

## 失效边界与采用检查

partial 的每项值域、分片字节乘法和 int64 求和都须合法；不能以经验上的小数值证明无溢出。源码固定容量阈值不能迁到另一目标；浮点归并还需单独审核误差和归并次序。若改用[DMA 原子写回](../../guides/memory/dma_atomic_write_scope.md)，重新证明输出初始化及状态关闭，不复用这里的“私有分片无冲突”论证。

采用时将完整写者集合与消费者读取集合逐项对照，覆盖 `N<核数`、ceil 分配后出现空段、尾行、空输入、重复调用和非零初始 workspace。设备验证应分别检查中间 partial 与最终输出，再验证阶段顺序；本卡没有把这些建议写成已执行的实验。
