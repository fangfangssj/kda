---
type: "Runbook"
title: arch35 gather kernel TBuf 手动同步完整实现
description: arch35 多 dtype 随机索引 gather 的正确性优先完整配方：TBuf + MTE2/MTE3 双段搬运 + 每迭代 pipe_barrier + 双缓冲 index 乒乓预取 + 增量坐标分解，18 组合全编译、ST 25/25 bitwise。
tags: [task.implementation, topic.compute.indexing]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# arch35 gather kernel TBuf 手动同步完整实现

## 现象

在 Ascend950PR（DAV_3510/arch35，CANN 9.1.0）上实现 `bench_no_harness_gather`（沿 dim 的随机索引 gather，x ∈ 6 dtype × index ∈ 3 dtype，模板参数 `<D_T_X, BUFFER_MODE>` 经 `ASCENDC_TPL_SEL_PARAM` 路由）。迭代一在该 arch 上连续踩过 MTE 32B 对齐、S pipe 与 MTE 不可见、紧凑循环 flag 失步三个坑（各有独立卡片）；迭代二在此基线上补齐多 TilingKey 全 dtype 实例化、双缓冲模式与增量坐标分解，得到一套**上板验证过的完整 kernel**。本卡是该完整实现与其设计取舍的系统配方，供同类"标量索引 + 逐元素搬运"算子直接复用骨架。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel（`op_kernel/arch35/*.h` 模板类）开发；随机访存/索引类算子（gather/scatter/take 等）。
- API 路径：`TBuf<TPosition::VECCALC>` + `pipe.InitBuffer`；`DataCopyPad`（GM↔UB）；`SetFlag/WaitFlag<HardEvent::MTE2_S / MTE2_MTE3>`；`pipe_barrier(PIPE_ALL)`；`LocalTensor::ReinterpretCast/GetValue`；`ASCENDC_TPL_SEL_PARAM` 模板路由。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510）+ CANN 9.1.0。
- 适用粒度：operator_family（arch35 上标量索引 + 逐元素搬运类 kernel；TBuf 手动同步架构对任何 S/MTE 混用 kernel 可参考）。
- 路线优先级：safety_fallback（正确性优先：每元素 2 次 DMA + 每迭代全管道屏障；相对 TQue 标准流水线路线性能显著更低，本开发会话 无量化性能数据，未验证性能影响）。fallback 声明：本配方以正确性与 arch35 S pipe 同步可靠性为先，采用后应保留全部同步原语；性能优化路线（向量化 gather / 批量对齐分组搬运）见遗留问题。
- 明确不适用/未验证场景：批量规整搬运（应走批量 DataCopy + TQue）；V pipe 向量化 gather（性能更优但本开发会话 未实现）；非 arch35 平台的同步原语组合未验证（推断可迁移，需按平台头文件核对）。

## 根因/解决配方

**配方（单句）**：x/y 数据路径全部走 MTE2→32B 对齐 tempBuf→MTE3 双段搬运（UB 只碰缓冲区头部，天然 32B 对齐）；index 批量 MTE2 搬入 UB 后用 `MTE2_S` 同步给 S pipe `GetValue` 读取；S pipe 算出的地址在每元素迭代前用 `pipe_barrier(PIPE_ALL)` 对全管道可见；双缓冲模式用两个 indexBuf 乒乓预取。

**完整代码（终版 verbatim，取自事件流完整文件写入）**：

```cpp
/*!
 * \file bench_no_harness_gather.h
 * \brief BenchNoHarnessGather 算子 kernel 类定义（arch35）
 *
 * 实现逻辑（纯 MTE2/MTE3，避免 S pipe 不可靠同步问题）：
 *   1. CopyIn:   index(GM) → indexBuf(UB)     [MTE2, 连续读]
 *      MTE2_S 同步确保 index 在 UB 中可被 S pipe 读取
 *   2. Compute:  逐元素 gather:
 *      a. S pipe: 读取 index[j] (GetValue)，计算 srcOffset（增量坐标分解）
 *      b. pipe_barrier(PIPE_ALL): 确保 S pipe 计算的地址对 MTE2 可见
 *      c. DataCopyPad x[srcOffset](GM) → tempBuf(UB)  [MTE2, 32B 对齐 UB]
 *      d. MTE2_MTE3 同步
 *      e. DataCopyPad tempBuf(UB) → y[j](GM)           [MTE3, 32B 对齐 UB]
 *
 * 双缓冲模式（BUFFER_MODE=1）：
 *   - 分配 2 个 indexBuf（indexBufA / indexBufB），batch 间乒乓预取
 *   - MTE2_S 使用 EVENT_ID0 / EVENT_ID1 交替，实现下一批 index 预取与当前批处理重叠
 *   - 注意：arch35 上 S pipe (GetValue) 需要 MTE2_S 显式同步，TQue 的 EnQue/DeQue
 *     仅覆盖 MTE2→V 不覆盖 MTE2→S，故采用 TBuf + 手动 SetFlag/WaitFlag 方案
 *
 * 增量坐标分解（迭代二）：
 *   - 利用 outLinearIdx 连续递增特性，O(1) amortized 更新 coords_ 和 baseOffset_
 *   - 替代完整 O(ndim) 分解
 */

#ifndef BENCH_NO_HARNESS_GATHER_H
#define BENCH_NO_HARNESS_GATHER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bench_no_harness_gather_tiling_data.h"
#include "bench_no_harness_gather_tiling_key.h"

namespace NsBenchNoHarnessGather {

using namespace AscendC;

template <typename D_T_X, int BUFFER_MODE>
class BenchNoHarnessGather {
    static constexpr int32_t MAX_NDIM = 8;
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline BenchNoHarnessGather(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                const BenchNoHarnessGatherTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t GetIndexDtypeSize(int32_t indexDtype);
    __aicore__ inline int64_t ReadIndexValueByDtype(
        const LocalTensor<uint8_t>& indexLocal, int64_t elemIdx);
    __aicore__ inline int64_t ComputeSrcOffset(int64_t outLinearIdx, int64_t indexVal);
    __aicore__ inline void CopyIndexBatch(const LocalTensor<uint8_t>& indexLocal,
                                          int64_t batchIdx, int64_t currentNum,
                                          int64_t indexDtypeSize);
    __aicore__ inline void ProcessBatch(const LocalTensor<uint8_t>& indexLocal,
                                        LocalTensor<D_T_X>& tempLocal,
                                        int64_t globalStartIdx, int64_t currentNum);

private:
    // Tiling 参数
    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubFactor_ = 0;
    int32_t dim_ = 0;
    int32_t ndim_ = 0;
    int32_t indexDtype_ = 0;
    int64_t xStride_[MAX_NDIM] = {0};
    int64_t outShape_[MAX_NDIM] = {0};
    int64_t globalOffset_ = 0;

    // 增量坐标分解状态
    int64_t coords_[MAX_NDIM] = {0};
    int64_t baseOffset_ = 0;       // sum(coords_[d] * xStride_[d] for d != dim_)
    int64_t lastDecompIdx_ = -1;   // 上次分解到的线性索引

    // Buffer
    TPipe pipe;
    TBuf<TPosition::VECCALC> indexBufA;
    TBuf<TPosition::VECCALC> indexBufB;  // 仅 BUFFER_MODE=1 使用
    TBuf<TPosition::VECCALC> tempBuf;    // 32B 对齐临时缓冲区（单元素中转）
    GlobalTensor<D_T_X> xGm;
    GlobalTensor<D_T_X> yGm;
    GlobalTensor<uint8_t> indexGm;
};

template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline int64_t BenchNoHarnessGather<D_T_X, BUFFER_MODE>::GetIndexDtypeSize(int32_t indexDtype)
{
    // ge::DT_INT8=2, ge::DT_INT32=3, ge::DT_INT64=9
    if (indexDtype == 2) return 1;
    else if (indexDtype == 3) return 4;
    else return 8;
}

template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline int64_t BenchNoHarnessGather<D_T_X, BUFFER_MODE>::ReadIndexValueByDtype(
    const LocalTensor<uint8_t>& indexLocal, int64_t elemIdx)
{
    // index dtype 运行时分支：int8/int32/int64 三路
    if (indexDtype_ == 2) {  // DT_INT8
        auto typed = indexLocal.ReinterpretCast<int8_t>();
        return static_cast<int64_t>(typed.GetValue(static_cast<uint32_t>(elemIdx)));
    } else if (indexDtype_ == 3) {  // DT_INT32
        auto typed = indexLocal.ReinterpretCast<int32_t>();
        return static_cast<int64_t>(typed.GetValue(static_cast<uint32_t>(elemIdx)));
    } else {  // DT_INT64
        auto typed = indexLocal.ReinterpretCast<int64_t>();
        return typed.GetValue(static_cast<uint32_t>(elemIdx));
    }
}

template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline int64_t BenchNoHarnessGather<D_T_X, BUFFER_MODE>::ComputeSrcOffset(
    int64_t outLinearIdx, int64_t indexVal)
{
    // 增量坐标分解：O(1) amortized per element
    // 利用 outLinearIdx 连续递增特性，仅在最低维进位时更新
    if (outLinearIdx == 0 || outLinearIdx != lastDecompIdx_ + 1) {
        // 首次或非连续跳转：完整分解 O(ndim)
        int64_t remaining = outLinearIdx;
        baseOffset_ = 0;
        for (int32_t d = ndim_ - 1; d >= 0; d--) {
            coords_[d] = remaining % outShape_[d];
            remaining /= outShape_[d];
            if (d != dim_) {
                baseOffset_ += coords_[d] * xStride_[d];
            }
        }
    } else {
        // 增量更新：从最低维开始进位
        for (int32_t d = ndim_ - 1; d >= 0; d--) {
            coords_[d]++;
            if (d != dim_) {
                baseOffset_ += xStride_[d];
            }
            if (coords_[d] < outShape_[d]) {
                break;  // 未进位，结束
            }
            // 进位：减去整轮增量（coords_[d] 从 0→outShape_[d]，共 outShape_[d] 次）
            if (d != dim_) {
                baseOffset_ -= outShape_[d] * xStride_[d];
            }
            coords_[d] = 0;
            // 继续进位到 d-1
        }
    }
    lastDecompIdx_ = outLinearIdx;

    // 源地址 = baseOffset + indexVal * xStride[dim]
    return baseOffset_ + indexVal * xStride_[dim_];
}

template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessGather<D_T_X, BUFFER_MODE>::CopyIndexBatch(
    const LocalTensor<uint8_t>& indexLocal, int64_t batchIdx, int64_t currentNum,
    int64_t indexDtypeSize)
{
    // CopyIn: index GM → UB (MTE2, 连续读)
    DataCopyParams idxCopyParams;
    idxCopyParams.blockCount = 1;
    idxCopyParams.blockLen = static_cast<uint32_t>(currentNum * indexDtypeSize);
    idxCopyParams.srcStride = 0;
    idxCopyParams.dstStride = 0;
    DataCopyPad(indexLocal, indexGm[batchIdx * ubFactor_ * indexDtypeSize],
                idxCopyParams, {true, 0, 0, 0});
}

template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessGather<D_T_X, BUFFER_MODE>::ProcessBatch(
    const LocalTensor<uint8_t>& indexLocal, LocalTensor<D_T_X>& tempLocal,
    int64_t globalStartIdx, int64_t currentNum)
{
    // 逐元素 gather: MTE2 (GM→UB) + MTE3 (UB→GM) 双段搬运
    for (int64_t j = 0; j < currentNum; j++) {
        int64_t outLinearIdx = globalStartIdx + j;

        // S pipe: 读取 index 值 + 计算源地址偏移（增量坐标分解）
        int64_t indexVal = ReadIndexValueByDtype(indexLocal, j);
        int64_t srcOffset = ComputeSrcOffset(outLinearIdx, indexVal);
        int64_t outOffset = globalStartIdx + j;

        // pipe_barrier(PIPE_ALL): 确保 S pipe 计算的地址对 MTE2/MTE3 可见
        // arch35 上 S pipe 不可靠同步，需全管道屏障
        pipe_barrier(PIPE_ALL);

        // MTE2: xGm[srcOffset] → tempLocal (32B 对齐 UB)
        DataCopyParams elemParams;
        elemParams.blockCount = 1;
        elemParams.blockLen = static_cast<uint32_t>(sizeof(D_T_X));
        elemParams.srcStride = 0;
        elemParams.dstStride = 0;
        DataCopyPad(tempLocal, xGm[srcOffset], elemParams, {true, 0, 0, 0});

        // MTE2 → MTE3 同步
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

        // MTE3: tempLocal → yGm[outOffset] (32B 对齐 UB → GM)
        DataCopyParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(sizeof(D_T_X));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(yGm[outOffset], tempLocal, outParams);
    }
}

template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessGather<D_T_X, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, const BenchNoHarnessGatherTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    ubFactor_ = tilingData->ubFactor;
    dim_ = tilingData->dim;
    ndim_ = tilingData->ndim;
    indexDtype_ = tilingData->indexDtype;

    for (int32_t i = 0; i < MAX_NDIM; i++) {
        xStride_[i] = tilingData->xStride[i];
        outShape_[i] = tilingData->outShape[i];
    }

    globalOffset_ = tilingData->blockFactor * AscendC::GetBlockIdx();
    int64_t remainderLength = totalNum_ - globalOffset_;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) blockLength_ = 0;

    xGm.SetGlobalBuffer((__gm__ D_T_X*)x);
    yGm.SetGlobalBuffer((__gm__ D_T_X*)y);

    int64_t indexDtypeSize = GetIndexDtypeSize(indexDtype_);
    indexGm.SetGlobalBuffer(
        (__gm__ uint8_t*)index + globalOffset_ * indexDtypeSize,
        blockLength_ * indexDtypeSize);

    // UB 缓冲区分配
    int64_t indexUbSize = (ubFactor_ * indexDtypeSize + 31) & ~31;  // 32B 对齐
    int64_t tempBufSize = 32;  // 32 字节，足够放一个元素（最大 int64_t=8B）
    pipe.InitBuffer(indexBufA, indexUbSize);
    if constexpr (BUFFER_MODE) {
        pipe.InitBuffer(indexBufB, indexUbSize);
    }
    pipe.InitBuffer(tempBuf, tempBufSize);
}

template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessGather<D_T_X, BUFFER_MODE>::Process()
{
    // 空 Tensor 处理
    if (blockLength_ == 0) return;

    int64_t indexDtypeSize = GetIndexDtypeSize(indexDtype_);
    LocalTensor<D_T_X> tempLocal = tempBuf.Get<D_T_X>();

    int64_t loopCount = (blockLength_ + ubFactor_ - 1) / ubFactor_;

    if constexpr (BUFFER_MODE == 0) {
        // === 单缓冲模式 ===
        LocalTensor<uint8_t> indexLocal = indexBufA.Get<uint8_t>();
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == loopCount - 1)
                ? (blockLength_ - ubFactor_ * i) : ubFactor_;

            // CopyIn: index GM → UB (MTE2)
            CopyIndexBatch(indexLocal, i, currentNum, indexDtypeSize);

            // MTE2 → S 同步（确保 index 在 UB 中可被 S pipe 读取）
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            // 逐元素 gather
            int64_t globalStartIdx = globalOffset_ + i * ubFactor_;
            ProcessBatch(indexLocal, tempLocal, globalStartIdx, currentNum);
        }
    } else {
        // === 双缓冲模式 ===
        // 使用 2 个 indexBuf 乒乓预取，MTE2_S 用 EVENT_ID0/EVENT_ID1 交替
        LocalTensor<uint8_t> indexLocalA = indexBufA.Get<uint8_t>();
        LocalTensor<uint8_t> indexLocalB = indexBufB.Get<uint8_t>();

        // 预取 batch 0
        int64_t firstNum = (loopCount == 1) ? blockLength_ : ubFactor_;
        CopyIndexBatch(indexLocalA, 0, firstNum, indexDtypeSize);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);  // 通知 batch 0 就绪

        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == loopCount - 1)
                ? (blockLength_ - ubFactor_ * i) : ubFactor_;
            int64_t globalStartIdx = globalOffset_ + i * ubFactor_;

            int32_t curEvent = static_cast<int32_t>(i % 2);       // 当先前增量 event id
            int32_t nextEvent = 1 - curEvent;                      // 下一批 event id

            // 等待当先前增量 index 加载完成
            WaitFlag<HardEvent::MTE2_S>(curEvent);

            // 预取下一批 index（异步 MTE2，与当前批处理重叠）
            if (i + 1 < loopCount) {
                int64_t nextNum = (i + 1 == loopCount - 1)
                    ? (blockLength_ - ubFactor_ * (i + 1)) : ubFactor_;
                const LocalTensor<uint8_t>& nextBuf = (nextEvent == 0) ? indexLocalA : indexLocalB;
                CopyIndexBatch(nextBuf, i + 1, nextNum, indexDtypeSize);
                SetFlag<HardEvent::MTE2_S>(nextEvent);  // 通知下一批就绪
            }

            // 处理当前批
            const LocalTensor<uint8_t>& curBuf = (curEvent == 0) ? indexLocalA : indexLocalB;
            ProcessBatch(curBuf, tempLocal, globalStartIdx, currentNum);
        }
    }
}

} // namespace NsBenchNoHarnessGather
#endif // BENCH_NO_HARNESS_GATHER_H
```

配套机制：`op_host/arch35/bench_no_harness_gather_tiling.cpp` 按 `totalNum > 1024` 阈值选择 `useDoubleBuffer` 并经 `ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(xDtype), useDoubleBuffer)` 路由（TilingKey = dtype + useDoubleBuffer×256，见 [ASCENDC_TPL 模板路由与 TilingKey 实测编码规则](../compilation/ascendc_tpl_routing_tilingkey_encoding.md)）；TilingData 携带 `totalNum/blockFactor/ubFactor/dim/ndim/indexDtype/xStride[8]/outShape[8]`。

## 设计决策

- **TBuf + 手动同步而非 TQue 标准流水线（关键取舍）**：TQue 的 EnQue/DeQue 只同步 MTE2→V 与 V→MTE3，本 kernel 的 Compute 内含 MTE2（读 x）且用 S pipe 读 index，TQue 覆盖不到；迭代一实测 S pipe 不同步即出错，故锁定 TBuf 架构。设计文档原定的 TQue 方案因该 arch 约束被替换（设计偏离已记录进 issue）。
- **双缓冲做成 index 乒乓预取而非三级流水**：Compute 内无 V 操作，MTE2 串行限制了完整流水收益；乒乓让"下一批 index 加载"与"当前批逐元素处理"重叠，是字面满足双缓冲且不引入 S pipe 回归的最小方案。
- **EVENT_ID0/ID1 交替用于乒乓**：相邻两批的 `MTE2_S` 事件不冲突；注意这与逐元素循环内换 EVENT_ID 无效（另一问题）不矛盾——乒乓是批次级、每批只 Set/Wait 一次。
- **增量坐标分解**：outLinearIdx 连续递增，仅在最低维进位时 O(1) 更新 `coords_/baseOffset_`，替代逐元素 O(ndim) 完整分解；batch 边界/核边界跳变时回退完整分解。
- **多 dtype 实例化靠编译系统**：`ASCENDC_TPL_SEL` 声明 12 组模板组合后，`npu_op_kernel_sources` 自动实例化为 18 个 binary（6 x dtype × 3 index dtype，每个 binary 内含单/双缓冲 2 kernel）；index dtype 不进 TilingKey，kernel 内运行时三路分支。
- **性能定位（如实声明）**：本配方是正确性优先的 safety_fallback——每元素 2 次 DMA + 每迭代 `pipe_barrier(PIPE_ALL)` + S pipe 标量读，相对 TQue 向量化路线性能显著更低；本开发会话 无 profiling 数据，未量化。

## 系统信息

- 工具链/环境：x86_64 宿主机，CANN 9.1.0（`ASCEND_HOME_PATH` 指向 CANN 安装根），`bash build.sh --soc=ascend950 -j8`。
- 目标平台：Ascend950PR（DAV_3510/arch35），平台参数 48 核 / UB 253952B。
- 相关文件：`op_kernel/arch35/bench_no_harness_gather.h`（本卡完整代码）、`bench_no_harness_gather_tiling_data.h`、`bench_no_harness_gather_tiling_key.h`、`op_kernel/bench_no_harness_gather_arch35.cpp`（入口）、`op_host/arch35/bench_no_harness_gather_tiling.cpp`、`op_host/bench_no_harness_gather_def.cpp`（18 dtype 组合注册）。
- 验证链：18 binary 全编译 → 打包安装 → UT 38/38 → ST Real 25/25 bitwise。

## 性能影响

本开发会话 无 profiling 信号，相对代价只能定性：相对 TQue 标准流水线（MTE2||V||MTE3 重叠）路线（来源平台：950，同一 arch 的常规实现形态），本配方每元素 2 次 DMA、每迭代一次全管道屏障、逐元素 S pipe 标量读，属显著更慢的正确性优先路线（未验证量化倍数）。采用条件：正确性验证阶段、或同样受 arch35 S pipe 同步约束的 kernel。

## 预防

- 在 arch35 上沿用本骨架时，不要"简化"掉 `pipe_barrier(PIPE_ALL)` 或 `MTE2_S` 同步——三者任缺其一都会复现迭代一的错值/失步。
- kernel 源改动后确认 binary 真的重编（见同仓增量编译卡）再上板。

## 验证信号

实现与验证来自 `bench_no_harness_gather` 迭代二（实现 → 联调验收，Ascend950PR + CANN 9.1.0）：

- 编译：18 个 kernel binary 全部生成（6 x dtype × 3 index dtype，每个含单/双缓冲 2 kernel），无编译错误（来源平台：950 构建链路；证据等级：tool_observed）。
- UT（op_host tiling/infershape）：38/38 通过（含全 dtype TilingKey 断言、双缓冲阈值边界 1024/1025、空 Tensor）（来源平台：950 工程 UT 链路；证据等级：tool_observed）。
- ST Real（NPU 上板，fp32/fp16/bf16/int8/int32/int64 × int8/int32/int64 用例 + 6 穿刺含空 Tensor/负索引）：25/25 全部通过且 **bitwise=YES**（来源平台：950；证据等级：tool_observed）。
- 迭代一回归：迭代一 10 用例（fp32+int32）在迭代二 kernel 上全部通过，无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 同步原语组合验证于 arch35 + CANN 9.1.0；其他代次 arch 需核对 `HardEvent` 枚举与 `PipeBarrier<PIPE_S>` 的 static_assert 情况（未验证，推断可迁移）。
- 双缓冲收益受限于 Compute 内 MTE2 串行；对计算密集 batch 才有明显重叠收益——未做性能剖析。
- `MAX_NDIM=8`、tempBuf 32B、`totalNum>1024` 双缓冲阈值是本工程参数，跨算子复用时按自身 tiling 重定。
- 增量坐标分解假设同一核内 outLinearIdx 连续（多核各管连续段）；若输出索引乱序分发需回退完整分解路径（代码已含该回退）。

**未决项**：

- 向量化 gather（V pipe Gather 指令 / 批量对齐分组搬运）未实现，是本配方的性能接续方向——优先项。
- 双缓冲乒乓的实际重叠收益未 profiling 量化（本开发会话 无性能数据）。
## 坏实践

- 用 TQue 的 EnQue/DeQue 当作 S pipe 同步（不覆盖 MTE2→S）。
- 把逐元素搬运的 UB 目的写成 `yLocal[j]`（非 32B 对齐，errcode 80）。
- 为"提速"删掉每迭代 `pipe_barrier(PIPE_ALL)`——小用例全过、大用例从第 8 元素起零星错值。

## 误诊

- 无（本卡为正向配方；各坑的误诊见对应独立卡片）。
