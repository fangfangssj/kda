---
type: "Runbook"
title: 纯搬运算子 TBuf+PipeBarrier 逐块搬运骨架
description: "无 Vector 计算的搬运类算子（Transpose 等）可用 TBuf<TPosition::VECCALC> + PipeBarrier<PIPE_ALL> 逐块双向 DataCopyPad 骨架绕开队列流水线与 UB 32B 块错位；TK0 连拷/TK1 二维/TK2 通用 N-D 三分支完整实现，7 dtype 全量 bitwise 验证。"
tags: [task.implementation, topic.data_movement, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-16T10:17:47Z'
platforms: ["950"]
---

# 纯搬运算子 TBuf+PipeBarrier 逐块搬运骨架

## 现象（任务背景）

在 Ascend950PR（arch35，CANN 9.1.0）上从零实现一个自定义 Transpose 算子（7 dtype × 3 调度分支），要求：纯搬运（无计算）、支持任意 perm 化简路由（恒等连拷 / 二维块转置 / 通用 N-D）、空张量安全、bitwise 精确匹配。初版用单 TQue 队列 + 多块 strided read 的写法连续踩中两个平台级陷阱（VECIN 单队列不触发 MTE3 写出、多块 strided read 时 blockLen<32B 每块落 UB 32 字节槽位使有效数据被拉开），最终收敛为一套经全量验证的 TBuf+PipeBarrier 逐块搬运骨架——本卡给出这套完整实现与设计取舍，供同类纯搬运算子直接复用。

## 适用场景

- 工程形态：AscendC 自定义算子开发（registry-invoke，custom opp 打包），纯搬运/数据重排类 kernel（Transpose/Reshape/Gather 拷贝段等），无 Vector 计算。
- API 路径：`TBuf<TPosition::VECCALC>`、`DataCopyPad` + `DataCopyExtParams`、`PipeBarrier<PIPE_ALL>`、`ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL_PARAM`（dtype × SCH_MODE 模板选择）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——骨架与分支路由思路可迁移到任何"读入-重排-写出"型搬运算子；tiling 字段（M/K/组步长）语义需按算子重定义。
- 路线优先级：recommended_fastpath（正确性优先的通用写法；性能敏感场景见设计决策中的对齐替代路线）。
- 明确不适用 / 未验证场景：含 Vector 计算的算子（应回 TQue VECIN/VECOUT 双队列流水）；要求搬运带宽峰值的场景（逐块+每块双屏障未做性能调优，无性能数据）；其他平台/版本（未验证，推断可迁移）。

## 设计决策

- **为什么 TBuf 而不是 TQue**：纯搬运没有 Vector 级，TQue 的队列位置绑定 MTE 方向（VECIN 只进、VECOUT 只出），双向共用一个队列会静默丢搬运；TBuf 不绑定流水线，MTE2/MTE3 都可访问，配 `PipeBarrier<PIPE_ALL>()` 手动同步。
- **为什么逐块搬运**：多块 strided read 在 blockLen<32B 时每块落 UB 的 32 字节槽位、有效数据被拉开，contiguous 回读必错位；块长由算子语义决定不能凑齐 32B，逐块（blockCount=1）是唯一不改语义的写法。代价是每块两次全量屏障 + 两次 DMA 描述，正确性优先。
- **三分支路由**：host `SimplifyPerm` 化简 perm（合轴→输出位置→identity 提取→核心组计数）：numCore==0→TK0 连拷、==2→TK1 二维块转置、>=3→TK2 通用 N-D（odometer 外层 + fast-axis 逐块）。identity 只按位置判断（置换不变量）。
- **M/K 语义约定**：M/K 取**输出序**第一、第二核心组（输入布局 [batch][K][M][blk]、输出布局 [batch][M][K][blk]），kernel 偏移公式与该契约逐字对应——契约不同步即下标互换。
- **UB 预算**：`pipe.InitBuffer(tmpBuf, AlignUp32(字节数))`，tile 上限同时受 `<=尺寸`、`<=4095`（blockCount uint16 上限语义沿用）与 `ubSize/4` 约束（除 4 = 双向缓冲 + 保守余量；本轨迹未单独证明余量必要性，属保守值）。
- **被放弃的路线**：VECIN+VECOUT 双队列 + UB→UB 中转（`DataCopy` 32B 对齐硬约束、`DataCopyPad` 不支持 UB→UB，小块不可行）；单 VECOUT 队列（MTE2 写不进）。
- 当前方案属 recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0，bisheng 编译链，custom_opp_ubuntu_x86_64.run 部署。
- 目录布局与相关文件（与实际读写一一对应）：
  - `op_kernel/arch35/transpose.h`（本卡完整代码，kernel 类）
  - `op_kernel/transpose_arch35.cpp`（入口，ASCENDC_TPL 模板声明）
  - `op_kernel/arch35/transpose_tiling_data.h`（下发字段）/ `transpose_tiling_key.h`（dtype × SCH_MODE 键）
  - `op_host/arch35/transpose_tiling.cpp` / `.h`（SimplifyPerm + 三分支 tiling）
  - `op_api/aclnn_transpose.cpp`（L2 两段式）、`op_api/transpose.cpp`（ADD_TO_LAUNCHER_LIST_AICORE）
  - `tests/st/test_aclnn_transpose.cpp`（ST）、`tests/ut/op_host/test_transpose_tiling.cpp`（UT）
- 编译产物：7 个 .o（每 dtype 一个，各含 3 个 kernelList 实例）= 7 dtype × 3 SCH_MODE = 21 kernel 实例。

## 完整代码

kernel 完整实现（`op_kernel/arch35/transpose.h` 终版，verbatim；host 侧 tiling 要点见上文设计决策）：

```cpp
#ifndef OP_KERNEL_ARCH35_TRANSPOSE_H
#define OP_KERNEL_ARCH35_TRANSPOSE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "transpose_tiling_data.h"
#include "transpose_tiling_key.h"

namespace NsTranspose {

using namespace AscendC;

constexpr int64_t UB_ALIGN_BYTES = 32;

static __aicore__ inline int64_t AlignUp32(int64_t bytes) {
    return (bytes + UB_ALIGN_BYTES - 1) / UB_ALIGN_BYTES * UB_ALIGN_BYTES;
}

template <typename T, int SCH_MODE>
class Transpose {
public:
    __aicore__ inline Transpose(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const TransposeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyProcess();
    __aicore__ inline void Transpose2DProcess();
    __aicore__ inline void GeneralNdProcess();

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    int64_t totalNum_ = 0;
    int64_t elemSize_ = 0;

    int64_t blockFactor_ = 0;
    int64_t ubFactor_ = 0;

    int64_t batch_ = 0;
    int64_t blkSize_ = 0;
    int64_t mSize_ = 0;
    int64_t kSize_ = 0;
    int64_t tileM_ = 0;
    int64_t totalWorkUnits_ = 0;
    int64_t perCoreUnits_ = 0;
    int64_t mChunksPerRow_ = 0;

    int64_t groupNum_ = 0;
    int64_t outerNum_ = 0;
    int64_t fastSize_ = 0;
    int64_t fastInStride_ = 0;
    int64_t fastTile_ = 0;
    int64_t fastChunks_ = 0;
    int64_t perCoreOuter_ = 0;
    int64_t coreTotal_ = 0;
    int64_t groupSizes_[8] = {0};
    int64_t inGroupStrides_[8] = {0};
    int64_t outGroupStrides_[8] = {0};
};

template <typename T, int SCH_MODE>
__aicore__ inline void Transpose<T, SCH_MODE>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                     const TransposeTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    elemSize_ = tilingData->elemSize;

    if (totalNum_ == 0) {
        return;
    }

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    if constexpr (SCH_MODE == 0) {
        blockFactor_ = tilingData->blockFactor;
        ubFactor_ = tilingData->ubFactor;
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t offset = blockIdx * blockFactor_;
        int64_t remainder = totalNum_ - offset;
        blockFactor_ = (remainder > blockFactor_) ? blockFactor_ : remainder;
        if (blockFactor_ < 0) {
            blockFactor_ = 0;
        }
        xGm = xGm[offset];
        yGm = yGm[offset];
        pipe.InitBuffer(tmpBuf, AlignUp32(ubFactor_ * sizeof(T)));
    } else if constexpr (SCH_MODE == 1) {
        batch_ = tilingData->batch;
        blkSize_ = tilingData->blkSize;
        mSize_ = tilingData->mSize;
        kSize_ = tilingData->kSize;
        tileM_ = tilingData->tileM;
        totalWorkUnits_ = tilingData->totalWorkUnits;
        perCoreUnits_ = tilingData->perCoreUnits;
        mChunksPerRow_ = tilingData->mChunksPerRow;
        pipe.InitBuffer(tmpBuf, AlignUp32(tileM_ * blkSize_ * sizeof(T)));
    } else {
        groupNum_ = tilingData->groupNum;
        outerNum_ = tilingData->outerNum;
        fastSize_ = tilingData->fastSize;
        fastInStride_ = tilingData->fastInStride;
        fastTile_ = tilingData->fastTile;
        fastChunks_ = tilingData->fastChunks;
        perCoreOuter_ = tilingData->perCoreOuter;
        coreTotal_ = tilingData->coreTotal;
        batch_ = tilingData->batch;
        blkSize_ = tilingData->blkSize;
        totalWorkUnits_ = tilingData->totalWorkUnits;
        for (int i = 0; i < 8; ++i) {
            groupSizes_[i] = tilingData->groupSizes[i];
            inGroupStrides_[i] = tilingData->inGroupStrides[i];
            outGroupStrides_[i] = tilingData->outGroupStrides[i];
        }
        pipe.InitBuffer(tmpBuf, AlignUp32(fastTile_ * blkSize_ * sizeof(T)));
    }
}

template <typename T, int SCH_MODE>
__aicore__ inline void Transpose<T, SCH_MODE>::Process()
{
    if (totalNum_ == 0) {
        return;
    }
    if constexpr (SCH_MODE == 0) {
        CopyProcess();
    } else if constexpr (SCH_MODE == 1) {
        Transpose2DProcess();
    } else {
        GeneralNdProcess();
    }
}

template <typename T, int SCH_MODE>
__aicore__ inline void Transpose<T, SCH_MODE>::CopyProcess()
{
    int64_t loopCount = (blockFactor_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1) ? (blockFactor_ - ubFactor_ * i) : ubFactor_;
        if (currentNum <= 0) {
            break;
        }
        auto xLocal = tmpBuf.Get<T>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        // GM -> UB (MTE2)
        DataCopyPad(xLocal, xGm[i * ubFactor_], copyParams, {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        // UB -> GM (MTE3)
        DataCopyPad(yGm[i * ubFactor_], xLocal, copyParams);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T, int SCH_MODE>
__aicore__ inline void Transpose<T, SCH_MODE>::Transpose2DProcess()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t uStart = blockIdx * perCoreUnits_;
    int64_t uEnd = uStart + perCoreUnits_;
    if (uEnd > totalWorkUnits_) {
        uEnd = totalWorkUnits_;
    }

    int64_t unitsPerBatch = kSize_ * mChunksPerRow_;

    for (int64_t u = uStart; u < uEnd; ++u) {
        int64_t b = u / unitsPerBatch;
        int64_t rem = u % unitsPerBatch;
        int64_t k = rem / mChunksPerRow_;
        int64_t mChunk = rem % mChunksPerRow_;
        int64_t m0 = mChunk * tileM_;
        int64_t curM = tileM_;
        if (m0 + curM > mSize_) {
            curM = mSize_ - m0;
        }
        if (curM <= 0) {
            continue;
        }

        int64_t inOffset = b * kSize_ * mSize_ * blkSize_ + k * mSize_ * blkSize_ + m0 * blkSize_;
        int64_t outOffset = b * mSize_ * kSize_ * blkSize_ + m0 * kSize_ * blkSize_ + k * blkSize_;

        auto xLocal = tmpBuf.Get<T>();

        // 逐块 strided read + contiguous write（避免 UB 32 字节 padding 问题）
        DataCopyExtParams singleParams;
        singleParams.blockCount = 1;
        singleParams.blockLen = static_cast<uint32_t>(blkSize_ * sizeof(T));
        singleParams.srcStride = 0;
        singleParams.dstStride = 0;

        for (int64_t m = 0; m < curM; m++) {
            int64_t inElem = inOffset + m * blkSize_;
            int64_t outElem = outOffset + m * kSize_ * blkSize_;
            // GM -> UB (single block, MTE2)
            DataCopyPad(xLocal, xGm[inElem], singleParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            // UB -> GM (single block, MTE3)
            DataCopyPad(yGm[outElem], xLocal, singleParams);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <typename T, int SCH_MODE>
__aicore__ inline void Transpose<T, SCH_MODE>::GeneralNdProcess()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t uStart = blockIdx * perCoreOuter_;
    int64_t uEnd = uStart + perCoreOuter_;
    if (uEnd > totalWorkUnits_) {
        uEnd = totalWorkUnits_;
    }

    int64_t numCore = groupNum_;
    int64_t numOuter = numCore - 1;
    int64_t blk = blkSize_;

    for (int64_t u = uStart; u < uEnd; ++u) {
        // Decode workUnit -> (b, outerPos, fastChunk)
        int64_t unitsPerBatch = outerNum_ * fastChunks_;
        int64_t b = u / unitsPerBatch;
        int64_t rem = u % unitsPerBatch;
        int64_t outerPos = rem / fastChunks_;
        int64_t fastChunk = rem % fastChunks_;

        int64_t f0 = fastChunk * fastTile_;
        int64_t curF = fastTile_;
        if (f0 + curF > fastSize_) {
            curF = fastSize_ - f0;
        }
        if (curF <= 0) {
            continue;
        }

        // Decode outerPos -> group indices (output order, groups 0..numOuter-1)
        int64_t gIdx[8] = {0};
        int64_t tmp = outerPos;
        for (int i = numOuter - 1; i >= 0; --i) {
            gIdx[i] = tmp % groupSizes_[i];
            tmp /= groupSizes_[i];
        }

        // Compute base offsets (in core region, element units)
        int64_t baseCoreIn = 0;
        int64_t baseCoreOut = 0;
        for (int i = 0; i < numOuter; ++i) {
            baseCoreIn += gIdx[i] * inGroupStrides_[i];
            baseCoreOut += gIdx[i] * outGroupStrides_[i];
        }

        // Full offsets (including batch and blk)
        int64_t batchInOffset = b * coreTotal_ * blk;
        int64_t inBase = batchInOffset + (baseCoreIn + f0 * fastInStride_) * blk;
        int64_t outBase = batchInOffset + (baseCoreOut + f0) * blk;

        auto xLocal = tmpBuf.Get<T>();

        // 逐块 read + write（避免 UB 32 字节 padding 问题，与 TK1 一致）
        DataCopyExtParams singleParams;
        singleParams.blockCount = 1;
        singleParams.blockLen = static_cast<uint32_t>(blk * sizeof(T));
        singleParams.srcStride = 0;
        singleParams.dstStride = 0;

        for (int64_t f = 0; f < curF; ++f) {
            int64_t inElem = inBase + f * fastInStride_ * blk;
            int64_t outElem = outBase + f * blk;
            DataCopyPad(xLocal, xGm[inElem], singleParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(yGm[outElem], xLocal, singleParams);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

} // namespace NsTranspose
#endif // OP_KERNEL_ARCH35_TRANSPOSE_H
```

要点提炼：单 `TBuf<TPosition::VECCALC>` 共享缓冲（三个分支各按自己 tile 尺寸 `AlignUp32` 初始化）；每次搬运 `blockCount=1`；每次 DataCopyPad 后跟 `PipeBarrier<PIPE_ALL>()`；空张量在 `Init`/`Process` 双重早退；核间按 workUnit 均分（`perCoreUnits`/`perCoreOuter`），`SetBlockDim(usedCoreNum)` 由 host 下发。

## 预防

- 新写纯搬运 kernel 直接从本骨架起步，避免先踩 TQue/多块 strided read 两个坑再回退。
- 复用时把"tiling 字段布局契约"注释（输入 [batch][K][M][blk]、输出 [batch][M][K][blk]）与 kernel 偏移公式一起拷走，改 tiling 语义时两处同步。

## 验证信号

以下信号来自该算子三个迭代的联调与验收会话（Ascend950PR Real 模式）。

- 编译：7 dtype × 3 SCH_MODE = 21 kernel 实例生成（7 个 .o，每个 .json 含 3 个 kernelList 条目），全量重编译可重复、无 error（仅 ge_api_types.h 弃用告警）（来源平台：950；证据等级：tool_observed）。
- 精度：最终版全量 ST 53/53 用例 bitwise 精确匹配通过（L0 5 + 回归 4 + L1 44，覆盖 7 dtype、恒等/全反转/循环 perm、素数维 1009 非对齐、空张量、1 元素边界）（来源平台：950；证据等级：tool_observed）。
- 精度：前版 ST 15/15 bitwise + UT 27/27（含 4 个 TK2 新用例）（来源平台：950；证据等级：tool_observed）。
- 稳定性：多次 `--make_clean` 全量重编 + 重装包回归，结果一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 性能未经调优与测量：逐块 + 每块双 PipeBarrier 是正确性优先写法，大 shape 带宽表现未知（无 profiling 数据，不构成任何性能承诺）；若尾维可保证 ≥32B 对齐，可在 tiling 层恢复批量搬运。
- 组数上限 8（`groupSizes[8]` 固定数组）：更高维 perm 需扩数组，未验证。
- 验证于 950 + CANN 9.1.0 + fp16/int 类/bf16/fp32 dtype 集合；其他平台未验证，推断可迁移。

**未决项**：

- 逐块搬运与"32B 对齐批量搬运"的性能分界未测量（无 profiling 信号，列为后续可选优化方向）。
- `PipeBarrier<PIPE_ALL>` 是否可换 `SetFlag/WaitFlag` 定向同步未验证。
## 坏实践

- 在该骨架上"优化"回多块 strided read（blockCount>1）而不核对块长 ≥32B——直接复现 UB 槽位错位。
- 把 `PipeBarrier<PIPE_ALL>()` 换成更窄的屏障类型以图性能——本轨迹未验证窄屏障的正确性，MTE2/MTE3 跨流同步语义需单独论证。

## 误诊

- 无（骨架为正向实现；初版两处陷阱的误诊记录属于各自故障卡的范围）。
