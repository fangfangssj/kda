---
type: "Runbook"
title: 多遍扫描漏 CopyInChunk，DeQue 读到残数据
description: ColSplit 3 遍扫描的 Pass 2/3 循环体缺少 CopyInChunk，TQue DeQue 返回 Pass 1 最后一个 chunk 的残留数据，输出成片陈旧值（xOut 全为同一错值）；补齐每遍搬入并在 CopyOut 后 FreeTensor 前 pipe_barrier(PIPE_MTE3) 后 11/11 通过。
tags: [task.debugging, topic.data_movement.data_copy, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-16T11:11:42Z'
updated_at: '2026-09-16T11:11:42Z'
platforms: ["950"]
---

# 多遍扫描漏 CopyInChunk，DeQue 读到残数据

## 现象

融合量化算子（Add + RMSNorm + per-token INT8 动态量化）的 ColSplit 分块路径实现后（不再崩溃），N>10240 的三条 ST 用例转为**输出值错误**：y(int8) 逐元素绝对误差高达 40~210、xOut 的 MARE 高达 2.4e+03、scaleOut 相对误差约 3e-03~4.7e-02。最扎眼的特征是 xOut 的不匹配元素**全部等于同一个值**：

```
--- xOut (float16) 比对 ---
不匹配 [0]: golden=0.115723, actual=-1.426758, rel_err=1.33e+01
不匹配 [1]: golden=0.633301, actual=-1.426758, rel_err=3.25e+00
不匹配 [2]: golden=0.868164, actual=-1.426758, rel_err=2.64e+00
不匹配 [3]: golden=0.830078, actual=-1.426758, rel_err=2.72e+00
```

在 kernel 里加 AscendC::printf 探针并让 ST 同时 dump 输入后拿到决定性对照：

```
[DEBUG] input.x1[0]=0.372314, input.x2[0]=0.378418, golden.xOut[0]=0.750732   ← CPU/GM 侧输入正确
[AIV Block 0/56] Pass2 chunk0: x1[0]=0.928223, x2[0]=-0.299561, xOut[0]=0.628662, count=10240  ← Pass2 读到的 x1[0] 不是 GM 的值
[AIV Block 0/56] Pass2 chunk0 before CopyOut: xOutOut[0]=0.628906 (UB)
```

GM 中的 x1[0]=0.372314（与 CPU 输入一致），但 Pass 2 从输入队列 DeQue 出来的 x1[0]=0.928223——是**别的 chunk/别的行的陈旧数据**；而 Pass 1 的 sumSq、rms、invRms 打印值全部合理正确（Pass 1 有逐 chunk 搬入）。

当时的 kernel 状态：3 遍扫描骨架已就位，Pass 1 循环内正确调用了 `CopyInChunk(rowIdx, cStart, cSize)`，但 Pass 2 / Pass 3 的循环体直接调用 `ComputeChunkNormAndCopyOutXOut` / `ComputeChunkAndCopyOutY`，**没有再调 CopyInChunk**。

## 适用场景

- 工程形态：Ascend C kernel 多遍扫描（同一行数据分遍/分 chunk 反复消费）、TQue depth=1 的输入队列在多遍间复用的实现。
- API 路径：`TQue::AllocTensor/EnQue/DeQue/FreeTensor`、`DataCopyPad`（MTE 搬入/搬出）、`pipe_barrier(PIPE_MTE3)`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何"同一输入在 kernel 内被多遍读取"的多遍扫描/两阶段归一化类算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：单遍扫描、每遍使用独立 TBuf（不经 TQue 复用）的实现不触发；depth>1 的队列行为未验证。

## 根因

TQue 是"生产者-消费者"队列：`DeQue` 只是取走**最近一次 EnQue** 的 tensor，不感知"这一遍循环想读哪个 chunk"。Pass 2/3 循环没有重新 `CopyInChunk`（AllocTensor→DataCopyPad→EnQue），DeQue 拿到的是 Pass 1 最后一个 chunk 残留在队列里的旧数据——输入从未按当前 (row, chunk) 刷新，于是 Pass 2 写出的 xOut 全是同一份陈旧输入算出的同一值，y/scaleOut 同理。

次级问题：outQueueXOut/outQueueY 的 `BUFFER_NUM(depth)=1`，`DataCopyPad`（MTE3 搬出）后立即 `FreeTensor`，下一 chunk 的 `AllocTensor` 可能在 MTE3 尚未搬完时复用同一 buffer，被 V（Cast）覆盖——需要在 CopyOut 与 FreeTensor 之间 `pipe_barrier(PIPE_MTE3)` 排空。

## 解决配方

修复前（Pass 2/3 循环体缺少搬入）：

```cpp
    // ========== Pass 2: 计算 absMax + 写出 xOut ==========
    float absMax = 0.0f;
    for (uint32_t c = 0; c < chunkCount_; c++) {
        uint32_t cStart = c * chunkSize_;
        uint32_t cSize = (c == chunkCount_ - 1) ? lastChunkSize_ : chunkSize_;
        float chunkMax = ComputeChunkNormAndCopyOutXOut(rowIdx, cStart, cSize);
        absMax = (chunkMax > absMax) ? chunkMax : absMax;
    }
    float absMaxClamped = (absMax > ABS_MAX_CLAMP_MIN) ? absMax : ABS_MAX_CLAMP_MIN;
    scaleOut_ = absMaxClamped / QUANT_SCALE_DIVISOR;
    AccumulateScaleOut(scaleOut_);

    // ========== Pass 3: 计算 y + 写出 y ==========
    for (uint32_t c = 0; c < chunkCount_; c++) {
        uint32_t cStart = c * chunkSize_;
        uint32_t cSize = (c == chunkCount_ - 1) ? lastChunkSize_ : chunkSize_;
        ComputeChunkAndCopyOutY(rowIdx, cStart, cSize);
    }
```

修复后（每遍循环先搬入当前 chunk；Pass 3 同样补齐）：

```cpp
    // ========== Pass 2: 计算 absMax + 写出 xOut ==========
    float absMax = 0.0f;
    for (uint32_t c = 0; c < chunkCount_; c++) {
        uint32_t cStart = c * chunkSize_;
        uint32_t cSize = (c == chunkCount_ - 1) ? lastChunkSize_ : chunkSize_;
        CopyInChunk(rowIdx, cStart, cSize);
        float chunkMax = ComputeChunkNormAndCopyOutXOut(rowIdx, cStart, cSize);
        absMax = (chunkMax > absMax) ? chunkMax : absMax;
    }
    float absMaxClamped = (absMax > ABS_MAX_CLAMP_MIN) ? absMax : ABS_MAX_CLAMP_MIN;
    scaleOut_ = absMaxClamped / QUANT_SCALE_DIVISOR;
    AccumulateScaleOut(scaleOut_);

    // ========== Pass 3: 计算 y + 写出 y ==========
    for (uint32_t c = 0; c < chunkCount_; c++) {
        uint32_t cStart = c * chunkSize_;
        uint32_t cSize = (c == chunkCount_ - 1) ? lastChunkSize_ : chunkSize_;
        CopyInChunk(rowIdx, cStart, cSize);
        ComputeChunkAndCopyOutY(rowIdx, cStart, cSize);
    }
```

配套修复（depth=1 输出队列的 MTE3 同步，CopyOut 之后、FreeTensor 之前）：

```cpp
    DataCopyPad(outputGMXOut[rowOffset], xOutOut, outCopyParams);
    // ColSplit: 确保 MTE3 (DataCopyPad) 完成后再 FreeTensor
    // 避免 depth=1 时下一个 chunk 的 V (Cast) 覆盖 MTE3 未读完的 buffer
    pipe_barrier(PIPE_MTE3);
    outQueueXOut.FreeTensor(xOutOut);
```

逐 chunk 搬入的完整函数（修复后形态，供复用）：

```cpp
template <typename D_T_X, int32_t IS_COL_SPLIT>
__aicore__ inline void BenchNoHarnessAddRmsNormDynamicQuant<D_T_X, IS_COL_SPLIT>::CopyInChunk(
    uint64_t rowIdx, uint32_t cStart, uint32_t cSize)
{
    uint64_t localRow = rowIdx - rowStart_;
    uint64_t rowOffset = localRow * N_ + cStart;

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = cSize * static_cast<uint32_t>(sizeof(D_T_X));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};

    LocalTensor<D_T_X> x1Local = inQueueX1.template AllocTensor<D_T_X>();
    DataCopyPad(x1Local, inputGMX1[rowOffset], copyParams, padParams);
    inQueueX1.EnQue(x1Local);

    LocalTensor<D_T_X> x2Local = inQueueX2.template AllocTensor<D_T_X>();
    DataCopyPad(x2Local, inputGMX2[rowOffset], copyParams, padParams);
    inQueueX2.EnQue(x2Local);

    LocalTensor<D_T_X> gammaLocal = inQueueGamma.template AllocTensor<D_T_X>();
    DataCopyPad(gammaLocal, inputGMGamma[cStart], copyParams, padParams);
    inQueueGamma.EnQue(gammaLocal);
}
```

判别 / 定位方法（本会话走通）：

1. 输出"成片同一错值"且 Pass 1 中间量（sumSq/rms/invRms）正确 → 先怀疑后遍的输入没刷新，而不是计算逻辑。
2. kernel 内 AscendC::printf 打印 DeQue 后首元素 + ST 侧 dump 同一位置 CPU/GM 输入，做"GM vs 队列"对照——本例 GM 0.372314 vs 队列 0.928223，一步锁定。
3. 检查每一遍循环体的头部是否都有 Alloc/EnQue（CopyIn）；多遍结构里"上一遍的 EnQue"不是本遍的输入。

## 设计决策

- 修复选择"每遍每 chunk 显式 CopyInChunk"而不是"把整行驻留 UB 跨遍复用"：ColSplit 路径本来就是 N 超 UB 的场景，整行驻留不可行；多搬几遍 GM 换正确性。
- `pipe_barrier(PIPE_MTE3)` 放在 DataCopyPad 之后、FreeTensor 之前（而不是循环末尾）：精确覆盖"释放→重分配"窗口，开销最小。
- 保留 TQue 机制（而不是改成裸 TBuf + 手动同步）：TQue 的 Alloc/Free 语义本身没问题，问题在"漏生产"；改动面最小。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510/arch35），CANN 9.1.0。
- 相关文件：`op_kernel/arch35/<op>.h`（ProcessRowColSplit/CopyInChunk/ComputeChunk*）、`op_host/arch35/<op>_tiling.cpp`（chunkSize/chunkCount/lastChunkSize）、`tests/st/test_aclnn_*.cpp`（含 DEBUG dump 探针）。
- 探针用法：kernel 内 `AscendC::printf` 输出 `[AIV Block %d/56] Pass2 chunk0: x1[0]=%f ...`，ST 侧打印 `[DEBUG] input.x1[0]=...`，两者对照。

## 预防

- 多遍扫描 kernel 的每一遍循环体模板固定为"CopyIn(chunk) → Compute(chunk) → CopyOut(chunk)"，review 时逐遍检查循环头有没有搬入。
- 输出队列 depth=1 且循环 CopyOut 时，FreeTensor 前显式 `pipe_barrier(PIPE_MTE3)`。
- 输出"全为同一错误值"是"输入未刷新/读取陈旧 buffer"的强信号，优先做 GM vs 队列的探针对照。

## 验证信号

以下信号来自该算子 ColSplit 联调：探针对照定位 → 补 CopyInChunk + MTE3 barrier → 全量回归。

- 修复前：N=16384/16384/10241 的三条 ColSplit 用例 y abs_err 40~210、xOut MARE 2.4e+03、xOut 不匹配元素全部同一值 -1.426758（来源平台：950；证据等级：tool_observed）。
- 探针：GM x1[0]=0.372314 与 CPU 输入一致；Pass2 DeQue x1[0]=0.928223 为陈旧值；Pass1 sumSq/rms/invRms 合理（来源平台：950；证据等级：tool_observed）。
- 修复后：ST 全 11 条 L0 用例通过（含 ColSplit 三条与 FullLoad 八条，无回归）（来源平台：950；证据等级：tool_observed）。
- 后续汇合：L0 11 + L1 15 共 26/26 通过（含 8192×16384 压力用例）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡断言 TQue"DeQue 取最近一次 EnQue"的行为语义；框架内部对 depth=1 队列的复用实现未读源（行为层面已由探针+修复闭环）。
- `pipe_barrier(PIPE_MTE3)` 的必要性未做单独消融（与 CopyInChunk 同时落地）；若未来框架在 FreeTensor 内已含 MTE3 排空，该 barrier 冗余但无害（未验证）。
- gamma 的 CopyInChunk 以 `inputGMGamma[cStart]` 为源（每行共享同一 gamma 向量，仅按 chunk 偏移）；若算子 gamma 按行独立，偏移需加 row 项。

**未决项**：

- bf16 调试期 CPU/GM dump 值不一致的观测未根因（优先级低：修复后未复现；若复现建议先核查 dump 探针的读取粒度与 dtype 解释）。
- TQue depth=1 下框架对 FreeTensor→AllocTensor 复用窗口的内部实现未读源，pipe_barrier 的最小必要位置未单独验证。
## 坏实践

- 多遍循环里只给第一遍写搬入，后续遍直接 DeQue——队列里是上一遍的残货。
- `pipe_barrier(PIPE_ID_MTE3)` 是不存在的标识符（正确名 `PIPE_MTE3`），写错编译报 `use of undeclared identifier`；以编译器提示为准。
- 在 op_api 层加依赖 `aclTensor::GetDeviceAddr` 的调试打印（编译不过：`const class aclTensor has no member named GetDeviceAddr`），调试 kernel 数据应走 kernel 内 printf + GM dump 对照。

## 误诊

- 曾假设 `l0op::Contiguous`（L2 层输入连续化）创建了副本导致 kernel 读错地址：跳过 Contiguous 直连后失败现象完全不变，排除。
- 曾怀疑 TQue FreeTensor 时机不对（V 计算与队列释放竞争）：把 FreeTensor 提前/调整后错误逐位不变——不是输出侧问题，是输入侧没刷新。
- 曾怀疑 chunkSize=10240 过大触发越界：降到 4096 后 Pass2 读到的 x1[0] 变为 0.917969（仍错，只是换了个陈旧值），排除。
- bf16 用例调试时观察到 CPU 值与 GM dump 值不一致，一度怀疑 GM 数据损坏；该观测未追根因（修复后用例全过，未再复现），不作为结论依据。
