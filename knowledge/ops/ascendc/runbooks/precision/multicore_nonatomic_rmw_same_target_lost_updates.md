---
type: "Runbook"
title: "多核 scatter 同目标非原子 RMW 丢更新，改单核"
description: "reduce 类 scatter 把索引分片到多核并行读-改-写同一 y[target]，重复索引的更新互相覆盖丢失；单核顺序逐元素 RMW 与 golden 的 flat 顺序一致，修复后 3 万重复索引 0 失配。"
tags: [task.debugging, topic.precision, topic.compute.indexing, topic.compute.reduction, topic.implementation.synchronization]
status: stable
sources: []
created_at: "2026-09-14T12:20:54Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# 多核 scatter 同目标非原子 RMW 丢更新，改单核

## 现象

在 Ascend950PR（CANN 9.2.0）上为自定义算子 BenchHarnessScatter（scatter / scatter-reduce）做黑盒失败用例修复：修完拷贝分块竞争后，剩余失败仍集中在 reduce 模式。对 TTK result.csv 按"拷贝是否多块 × scatter 是否多核"二维统计（输出原文）：

```
singleChunkCopy=False singleCoreScatter=False -> F: 72
singleChunkCopy=False singleCoreScatter=False -> P: 3
singleChunkCopy=False singleCoreScatter=True -> F: 213
singleChunkCopy=False singleCoreScatter=True -> P: 75
singleChunkCopy=True singleCoreScatter=True -> P: 36
```

多核 scatter 用例几乎全军覆没（72F/3P）。当时的 kernel 把 indices 的 tile 分给多个核（`GetCoreRange` 按核切 tile），每个核对 GM 里的 `y[target]` 做读-改-写：add 模式下两核同时读到同一个旧值、各自加上自己的 update、先后写回，后写覆盖先写，一次累加就丢了。另外原实现对 bf16 走"批内 gather 全部旧值 → 向量 Add/Mul/Max/Min → scatter 回写"：批内两个相同索引会 gather 到同一个旧值，各自加完再先后回写，同样只留最后一个——批内重复索引也丢更新。全量 399 条里 283 条失败与这两个层面（跨核竞争、批内向量路径）叠加拷贝竞争共同造成。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel 开发 + TTK 黑盒精度评测，golden 为 numpy `np.add.at/np.maximum.at` 语义（重复索引全量累加，处理顺序 = flat 顺序）。
- API 路径：`AscendC::GlobalTensor::GetValue/SetValue`、`DataCopyPad`、`SyncAll`、`GetBlockIdx`；多核切分（`SetBlockDim` + per-core tile range）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——一切"按索引散布并归约"（scatter-reduce / index-put accumulate）类 kernel，golden 要求重复索引全量生效。
- 路线优先级：recommended_fastpath（正确性优先；单核串行的性能代价见性能影响）。
- 明确不适用 / 未验证场景：无重复索引且天然无冲突的散布（如逐元素一一映射）可用多核；原子路径（`SetAtomicAdd`）为替代方案但本会话未验证；其他架构未验证（推断可迁移）。

## 根因

多核并行对同一 GM 地址做非原子读-改-写，重复索引的更新互相覆盖丢失；且批内 gather→向量归约→scatter 回写的实现让批内重复索引读到同一旧值，丢掉除最后一个外的全部更新。

## 解决配方

修复前（原 Process：多核分派 tile + ComputeChunk；verbatim 取自会话中该文件原始内容）：

```cpp
template <typename T, typename I, int64_t RANK>
__aicore__ inline void BenchHarnessScatterKernel<T, I, RANK>::Process()
{
    CopyDataToY();
    AscendC::SyncAll();
    if(td_->totalIndices==0){ AscendC::PipeBarrier<PIPE_ALL>(); return; }

    // Phase 2: scatter loop
    int64_t start,end;
    GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tilesMain, td_->multicore.coresTail, start, end);

    for(int64_t flat=start;flat<end;flat++){
        int64_t aISeg=GetUBSplitRange(flat%td_->split.aO, td_->split.aO, td_->split.aI, td_->split.aITail);
        int64_t chunkBase=flat*td_->split.aI;
        CopyInChunk(chunkBase, aISeg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2_V_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2_V_);
        ComputeChunk(chunkBase, aISeg);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

其中 bf16 批内向量路径（批内重复索引丢更新的坏实践，verbatim）：

```cpp
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        // bf16: vector approach (gather → V reduce → scatter)
        for (int64_t i = 0; i < aISeg; i++) {
            I idxVal = buf_[B0].Get<I>().GetValue(i);
            int64_t flatPos = chunkBase + i;
            UnravelFlatToMultiIndex(flatPos, td_->indicesShape, td_->rank, multiIdx);
            int64_t targetOff = CalcTargetOffset(multiIdx, td_->dim, idxVal, td_->dataStrides, td_->rank);
            T oldVal = gmY_.GetValue(targetOff);
            buf_[B2].Get<T>().SetValue(i, oldVal);
        }
        switch (td_->reduceMode) {
            case 1: AscendC::Add(buf_[B2].Get<T>(), buf_[B2].Get<T>(), buf_[B1].Get<T>(), aISeg); break;
            case 2: AscendC::Mul(buf_[B2].Get<T>(), buf_[B2].Get<T>(), buf_[B1].Get<T>(), aISeg); break;
            case 3: AscendC::Max(buf_[B2].Get<T>(), buf_[B2].Get<T>(), buf_[B1].Get<T>(), aISeg); break;
            case 4: AscendC::Min(buf_[B2].Get<T>(), buf_[B2].Get<T>(), buf_[B1].Get<T>(), aISeg); break;
            default: break;
        }
        for (int64_t i = 0; i < aISeg; i++) {
            I idxVal = buf_[B0].Get<I>().GetValue(i);
            int64_t flatPos = chunkBase + i;
            UnravelFlatToMultiIndex(flatPos, td_->indicesShape, td_->rank, multiIdx);
            int64_t targetOff = CalcTargetOffset(multiIdx, td_->dim, idxVal, td_->dataStrides, td_->rank);
            T result = buf_[B2].Get<T>().GetValue(i);
            gmY_.SetValue(targetOff, result);
        }
    }
```

修复后（scatter 单核化：仅 core 0 顺序处理全部 tile，逐元素读-改-写；verbatim 取自会话中的重写版本）：

```cpp
template <typename T, typename I, int64_t RANK>
__aicore__ inline void BenchHarnessScatterKernel<T, I, RANK>::Process()
{
    CopyDataToY();
    AscendC::SyncAll();
    if(td_->totalIndices==0){ AscendC::PipeBarrier<PIPE_ALL>(); return; }

    // Phase 2: single-core scatter — eliminates multi-core RMW race.
    // Core 0 processes all index tiles sequentially so duplicate-index
    // accumulation is deterministic and matches the golden's flat order.
    if(AscendC::GetBlockIdx()!=0){ AscendC::PipeBarrier<PIPE_ALL>(); return; }

    int64_t totalTiles = td_->split.aO;
    for(int64_t flat=0;flat<totalTiles;flat++){
        int64_t aISeg=GetUBSplitRange(flat, td_->split.aO, td_->split.aI, td_->split.aITail);
        int64_t chunkBase=flat*td_->split.aI;
        CopyInChunk(chunkBase, aISeg);
        // MTE2→S sync: ensure indices/updates loaded into UB before S-pipe GetValue
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evMTE2_S_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evMTE2_S_);
        ScatterChunk(chunkBase, aISeg);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

ScatterChunk 的 reduce 分支（首版为 S-pipe 逐元素 GetValue/SetValue；verbatim）：

```cpp
    } else {
        // Reduce: y[target] = reduce(y[target], updates[p])
        for (int64_t i = 0; i < aISeg; i++) {
            I idxVal = idxBuf.GetValue(i);
            int64_t flatPos = chunkBase + i;
            UnravelFlatToMultiIndex(flatPos, td_->indicesShape, td_->rank, multiIdx);
            int64_t targetOff = CalcTargetOffset(multiIdx, td_->dim, idxVal,
                                                   td_->dataStrides, td_->rank);
            T oldVal = gmY_.GetValue(targetOff);
            T updVal = updBuf.GetValue(i);
            T newVal = ReduceScalar<T>(td_->reduceMode, oldVal, updVal);
            gmY_.SetValue(targetOff, newVal);
        }
    }
```

注：S-pipe `SetValue` 直写 GM 的可见性问题（大规模重复目标下仍丢更新）在同一会话后续被发现，最终访问机制改为"每元素 MTE2 读 → S 归约 → MTE3 写"并配事件同步，单核架构不变；该演化与未决的 2 字节丢更新问题见 [2 字节逐元素 GM RMW 大规模丢写判别卡](two_byte_gm_rmw_large_scale_lost_write.md)。多核拷贝阶段（各核不相交区间）保留不动。

## 设计决策

- 为什么单核而不是多核+原子：CANN 内置 scatter_add 用 `SetAtomicAdd<T>() + DataCopyPad` 做原子累加（内置实现源码可见），但其对 bf16/fp16 的支持（`IsSupportAtomicAddTypeSIMD`）在本会话未验证；单核 + 顺序 RMW 是确定性方案：处理顺序 = flat 顺序 = golden `np.add.at` 的 C-order 顺序，重复索引语义天然逐位对齐。
- 为什么逐元素而不是批内向量：批内 gather 把同一目标的多个重复索引读到不同 UB 槽，向量归约后 scatter 回写只保留最后一个——重复更新丢失（该结论是对原代码的机制分析，未单独实测）。
- 为什么拷贝阶段保留多核：各核拷贝互不相交的 flat 区间，无共享写目标，`SyncAll` 后再进入单核 scatter。
- 原 2 条 multiply PROFILE_CRASH（300s 超时）在重写后不再超时；超时主因按代码结构归因于非 bf16 路径"每 tile 整体搬 y"的 O(dataSize×numTiles) 流量（agent 分析，无 profiling 佐证）。
- 当前方案路线档位：recommended_fastpath（正确性）；多核+原子是未验证的更快候选。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0，TTK aclnn 评测。
- 相关文件：`op_kernel/arch35/bench_harness_scatter_kernel.h`（Process/ComputeChunk/ScatterChunk）、`op_host/arch35/bench_harness_scatter_tiling_arch35.cpp`（tile 切分与 numCores）、`tests/aclnn/repro_failures.cpp`（重复索引 add 复现：data(8,4096)、indices 全 0、dim=0、add，32768 次更新命中同一行）。

## 性能影响

单核串行逐元素 RMW 的绝对性能未测量（本会话无 profiling 数据），不构成可引用结论；相对事实：重写前多核+整体搬 y 的路径在 48.7M 索引用例上 300s 超时（PROFILE_CRASH），单核逐元素路径同用例可完成并出正确结果（来源平台：950；证据等级：tool_observed）。采用前请自行测量单核路径在超大索引规模下的耗时。

## 预防

- 写 scatter-reduce 类 kernel 前先确认 golden 的重复索引语义（`np.add.at` = 全量累加、flat 顺序）；任何"先 gather 一批再回写"的结构都默认对批内重复索引不安全。
- 多核切分只用于**写目标互不相交**的区间；只要两个核可能写同一地址，非原子 RMW 就是错的。

## 验证信号

以下信号来自该算子的一次黑盒失败用例修复会话（拷贝竞争与单核化在同一次重写中落地，矩阵分析区分两者的贡献）：

- 修复前相关性：多核 scatter 用例 72F/3P，单核 scatter 多块用例 75P/213F（后者主要由拷贝竞争贡献）（来源平台：950；证据等级：tool_observed）。
- 修复后：重复索引 add 复现（32768 次更新命中同一行，int32）0/32768 失配；拷贝+update 复现 0/100000 失配（来源平台：950；证据等级：tool_observed）。
- 修复后全量：TTK 正向 114→388/399（97.24%），2 条原超时用例不再 PROFILE_CRASH（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 单核串行在重复目标极密集且规模巨大时暴露另一层问题：2 字节元素（bf16/fp16）逐元素 GM RMW 在 ~2M 次重复命中同一位置时丢更新，任何同步原语无效（见 [2 字节逐元素 GM RMW 大规模丢写判别卡](two_byte_gm_rmw_large_scale_lost_write.md)，该会话未解决）；int32/fp32 4 字节路径万级重复实测正确。
- 低精度 dtype 的累加精度（逐步舍入 vs golden fp32 单次舍入）是另一独立根因，须用 FP32 中间累加解决（见 [scatter 归约 FP32 累加卡](low_precision_scatter_reduction_needs_fp32_accumulator.md)）。
- 多核+`SetAtomicAdd` 替代路线未尝试、未验证。
- 仅 Ascend950PR + CANN 9.2.0 实测（未验证，推断可迁移）。

**未决项**：

- 多核 + `SetAtomicAdd`（内置 scatter_add 模式）能否安全承接 bf16/fp16 add 未尝试。
- 单核路径在超大索引规模下的性能未测量（无 profiling 数据）。
## 坏实践

- reduce 模式下把 indices tile 分给多核做非原子 GM 读-改-写。
- 批内 gather→向量归约→scatter 回写处理含重复索引的批次。
- 认为核数少（如 2 核）就"大概率不撞"——TTK 随机索引下 72/75 失败说明撞是常态。
