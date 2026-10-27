---
type: "Runbook"
title: "DataCopyPad多行stride单位误用导致非对齐行输出NaN"
description: "多行DataCopyPad把对齐差的字节数直接传给srcStride或dstStride时会按32B块放大行间距；改为gap_bytes/32并覆盖copy-in、prefetch和copy-out可恢复非对齐行精度。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# DataCopyPad多行stride单位误用导致非对齐行输出NaN

## 现象

RMSNorm kernel 从逐行搬运改为多行 tile 后，对齐的 float16 定向用例仍通过，但一个非对齐 bfloat16 用例输出异常，测试报 `MERE nan >= 7.812e-03`。此前一次宽泛批跑还在后续用例观察到 ACL `507035`，但这次因果闭环采用的是可重复的非对齐用例 NaN，而不是把受污染上下文中的运行时错误当成主信号。

故障只在需要非零行对齐差的多行搬运路径出现；对齐路径在同一 kernel 下通过，因此应先检查 `DataCopyExtParams` 的 stride 单位，而不是修改 RMSNorm 数学、容差或测试代码。

## 适用场景

- 工程形态：Ascend C direct-invoke、MemBase AR kernel；
- 数据路径：一次 `DataCopyPad` 搬运多行，GM 行紧密排列，UB 行按 32B pitch 排列；copy-in、下一 tile prefetch 和 copy-out 都设置 `DataCopyExtParams` stride；
- 验证平台：Ascend950PR_957c / `dav-3510`，CANN 9.0.1；frontmatter 因而只标记 `platforms: [950]`；
- 适用粒度：`same_operator`；stride 单位机制可迁移，但其他算子、其他产品或 CANN 版本必须单独复验；
- 未验证：A2/A3、其他 CANN 版本，以及 gap 跨越一个或多个完整 32B block 的布局。

## 根因

旧实现计算： ```cpp rowGapBytes = (numColAlign - numCol) * sizeof(T); ``` 然后把这个“字节数”直接传给 copy-in 的 `dstStride`、prefetch 的 `dstStride` 和 copy-out 的 `srcStride`。但轨迹中固定 rms_norm golden 的多行搬运公式是： ```text stride = (aligned_columns - valid_columns) * element_bytes / 32 ``` 也就是说，这些 stride 字段使用 32B block 数，而 `blockLen` 仍使用字节数。把不足一个 32B block 的 raw gap bytes 直接作为 stride，会被 DMA 按多个 32B block 解释，实际行 pitch 被成倍放大；后续逐行计算仍以 `r * numColAlign` 读取 UB，于是读到错误行内容，平方归约后传播成 NaN。 修复保留 `blockLenBytes`，另算 `gapBlocks = gapBytes / 32`，并在三处多行搬运统一使用 `gapBlocks`。当前验证布局的 gap 都小于 32B，所以 `gapBlocks` 为 0；`DataCopyPad` 对非对齐 `blockLen` 的 32B 自动补齐仍形成正确的 UB 行 pitch。 注意：轨迹读取的 CANN 头文件只显示 arch35 上 stride 字段为 `int64_t`，没有直接写明单位。本卡的单位结论来自固定 golden 公式、单变量代码修正和真机定向/全量闭环，属于经行为验证的机制归因。

## 解决配方

以下是轨迹中该根因修复前后的完整函数，代码保持 verbatim。代码证据：revision 为轨迹内修复前后写入快照（没有记录对应 40 位 Git revision）；path `example/csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp`；symbol `rms_norm_kernel<T>`；行号未在轨迹中记录。

### before

```cpp
template <typename T>
__global__ __aicore__ void rms_norm_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                            int64_t numRow, int64_t numCol,
                                            int64_t blockFactor, int64_t ubFactor,
                                            int64_t numColAlign, float epsilon, float avgFactor)
{
    // SplitD (mode 1) not implemented for this batch (D<=8192 in all cases);
    // fall back to FullLoad single-row loop (correctness over perf).
    // FullLoad (mode 0): m=ubFactor rows per tile.

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, gammaGm, yGm;
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t rowStart = blockIdx * blockFactor;
    if (rowStart >= numRow) return;
    int64_t availRows = numRow - rowStart;
    int64_t rowCount = (blockFactor < availRows) ? blockFactor : availRows;

    xGm.SetGlobalBuffer((__gm__ T*)x + rowStart * numCol);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
    yGm.SetGlobalBuffer((__gm__ T*)y + rowStart * numCol);

    // m = rows per tile. Clamp to [1, rowCount]; honor host ubFactor when it fits.
    int64_t m = (ubFactor >= 1) ? ubFactor : 1;
    if (m > rowCount) m = rowCount;

    // UB buffers: x/gamma/y queues + per-row scratch (reused across rows in a tile).
    // xQueue/yQueue hold m rows per slot (depth2 double buffer); scratch is per-row.
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> xQueue, gammaQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> yQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf, sqFp32Buf, rstdBuf, reduceDstBuf, tmpBuf;
    pipe.InitBuffer(xQueue, PIPELINE_DEPTH, m * numColAlign * sizeof(T));
    pipe.InitBuffer(gammaQueue, 1, numColAlign * sizeof(T));
    pipe.InitBuffer(yQueue, PIPELINE_DEPTH, m * numColAlign * sizeof(T));
    pipe.InitBuffer(xFp32Buf, numColAlign * sizeof(float));
    pipe.InitBuffer(sqFp32Buf, numColAlign * sizeof(float));
    pipe.InitBuffer(rstdBuf, BLOCK_SIZE_B32 * sizeof(float));
    pipe.InitBuffer(reduceDstBuf, BLOCK_SIZE_B32 * sizeof(float));
    // ReduceSum sharedTmp: numColAlign * sizeof(float) is a safe upper bound.
    pipe.InitBuffer(tmpBuf, numColAlign * sizeof(float));

    // gamma 常驻: load once.
    auto gammaLocal = gammaQueue.AllocTensor<T>();
    AscendC::DataCopyExtParams gammaCp{1, static_cast<uint32_t>(numCol * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> gammaPp{false, 0, 0, 0};
    AscendC::DataCopyPad(gammaLocal, gammaGm[0], gammaCp, gammaPp);
    gammaQueue.EnQue(gammaLocal);
    gammaLocal = gammaQueue.DeQue<T>();

    // Copy params for multi-row tiles. dstStride pads each UB row to numColAlign.
    // DataCopyExtParams strides are in BYTES. Construct inline at each call site
    // (the __aicore__ constructors are only callable from __aicore__ context, i.e.
    // the kernel body, not from a lambda). uint32_t stride values match the
    // active variant and widen cleanly to int64_t on arch35.
    uint32_t rowGapBytes = static_cast<uint32_t>((numColAlign - numCol) * static_cast<int64_t>(sizeof(T)));
    uint32_t blockLenBytes = static_cast<uint32_t>(numCol * sizeof(T));
    AscendC::DataCopyPadExtParams<T> rowPp{false, 0, 0, 0};

    auto xFp32Local = xFp32Buf.Get<float>();
    auto sqFp32Local = sqFp32Buf.Get<float>();
    auto rstdLocal = rstdBuf.Get<float>();
    auto reduceDstLocal = reduceDstBuf.Get<float>();
    auto tmpLocal = tmpBuf.Get<uint8_t>();

    int64_t tiles = (rowCount + m - 1) / m;

    // ── Prologue: load tile 0 (issue MTE2 for first tile) ──
    int64_t curM0 = (m < rowCount) ? m : rowCount;
    auto xCur = xQueue.AllocTensor<T>();
    AscendC::DataCopyExtParams copyIn0{static_cast<uint16_t>(curM0), blockLenBytes,
                                       0, rowGapBytes, 0};
    AscendC::DataCopyPad(xCur, xGm[0], copyIn0, rowPp);
    xQueue.EnQue(xCur);

    for (int64_t t = 0; t < tiles; ++t) {
        int64_t curM = (t * m + m <= rowCount) ? m : (rowCount - t * m);
        // DeQue waits for MTE2 of tile t to complete.
        xCur = xQueue.DeQue<T>();

        // Prefetch next tile BEFORE compute so MTE2(t+1) overlaps V(t).
        if (t + 1 < tiles) {
            int64_t nextM = ((t + 1) * m + m <= rowCount) ? m : (rowCount - (t + 1) * m);
            auto xNext = xQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams copyInNext{static_cast<uint16_t>(nextM), blockLenBytes,
                                                  0, rowGapBytes, 0};
            AscendC::DataCopyPad(xNext, xGm[(t + 1) * m * numCol], copyInNext, rowPp);
            xQueue.EnQue(xNext);
        }

        // Allocate y tile (VECOUT); compute each row into it.
        auto yTile = yQueue.AllocTensor<T>();

        for (int64_t r = 0; r < curM; ++r) {
            uint32_t rowOff = static_cast<uint32_t>(r * numColAlign);
            auto xRow = xCur[rowOff];        // VECIN row r (T)
            auto yRow = yTile[rowOff];       // VECOUT row r (T)

            // 1-2. Cast x -> fp32 then sq = x*x. fp32: compute directly on xRow.
            if constexpr (AscendC::IsSameType<T, float>::value) {
                AscendC::Mul(sqFp32Local, xRow, xRow, numCol);
            } else {
                AscendC::Cast(xFp32Local, xRow, AscendC::RoundMode::CAST_NONE, numCol);
                AscendC::Mul(sqFp32Local, xFp32Local, xFp32Local, numCol);
            }
            // 3. Sum sq -> scalar. AR 2D pattern, single row.
            uint32_t srcShape[2] = {1, static_cast<uint32_t>(numCol)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(reduceDstLocal, sqFp32Local,
                                                                   tmpLocal, srcShape, false);
            float sum = reduceDstLocal.GetValue(0);
            // 4. rstd = 1/sqrt(avg*sum + eps), inf/zero guards (scalar C++ math).
            float var = sum * avgFactor + epsilon;
            float rstd;
            if (var == std::numeric_limits<float>::infinity()) {
                rstd = 0.0f;  // var==+inf -> rstd=0
            } else if (var == 0.0f) {
                rstd = std::numeric_limits<float>::infinity();  // var==0 -> rstd=+inf
            } else {
                rstd = 1.0f / std::sqrt(var);
            }
            // 5-6. y = x * rstd * gamma. fp32: compute straight into yRow.
            //      non-fp32: compute in xFp32Local, cast down to yRow.
            if constexpr (AscendC::IsSameType<T, float>::value) {
                AscendC::Muls(yRow, xRow, rstd, numCol);
                AscendC::Mul(yRow, yRow, gammaLocal, numCol);
            } else {
                AscendC::Muls(xFp32Local, xFp32Local, rstd, numCol);
                AscendC::Cast(sqFp32Local, gammaLocal, AscendC::RoundMode::CAST_NONE, numCol);
                AscendC::Mul(xFp32Local, xFp32Local, sqFp32Local, numCol);
                AscendC::Cast(yRow, xFp32Local, AscendC::RoundMode::CAST_ROUND, numCol);
            }
        }

        yQueue.EnQue(yTile);
        auto yOut = yQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyOut{static_cast<uint16_t>(curM), blockLenBytes,
                                           rowGapBytes, 0, 0};
        AscendC::DataCopyPad(yGm[t * m * numCol], yOut, copyOut);

        xQueue.FreeTensor(xCur);
        yQueue.FreeTensor(yOut);
    }

    gammaQueue.FreeTensor(gammaLocal);
}
```

### after

```cpp
template <typename T>
__global__ __aicore__ void rms_norm_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                            int64_t numRow, int64_t numCol,
                                            int64_t blockFactor, int64_t ubFactor,
                                            int64_t numColAlign, float epsilon, float avgFactor)
{
    // SplitD (mode 1) not implemented for this batch (D<=8192 in all cases);
    // fall back to FullLoad single-row loop (correctness over perf).
    // FullLoad (mode 0): m=ubFactor rows per tile.

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, gammaGm, yGm;
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t rowStart = blockIdx * blockFactor;
    if (rowStart >= numRow) return;
    int64_t availRows = numRow - rowStart;
    int64_t rowCount = (blockFactor < availRows) ? blockFactor : availRows;

    xGm.SetGlobalBuffer((__gm__ T*)x + rowStart * numCol);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
    yGm.SetGlobalBuffer((__gm__ T*)y + rowStart * numCol);

    // m = rows per tile. Clamp to [1, rowCount]; honor host ubFactor when it fits.
    int64_t m = (ubFactor >= 1) ? ubFactor : 1;
    if (m > rowCount) m = rowCount;

    // UB buffers: x/gamma/y queues + per-row scratch (reused across rows in a tile).
    // xQueue/yQueue hold m rows per slot (depth2 double buffer); scratch is per-row.
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> xQueue, gammaQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> yQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf, sqFp32Buf, rstdBuf, reduceDstBuf, tmpBuf;
    pipe.InitBuffer(xQueue, PIPELINE_DEPTH, m * numColAlign * sizeof(T));
    pipe.InitBuffer(gammaQueue, 1, numColAlign * sizeof(T));
    pipe.InitBuffer(yQueue, PIPELINE_DEPTH, m * numColAlign * sizeof(T));
    pipe.InitBuffer(xFp32Buf, numColAlign * sizeof(float));
    pipe.InitBuffer(sqFp32Buf, numColAlign * sizeof(float));
    pipe.InitBuffer(rstdBuf, BLOCK_SIZE_B32 * sizeof(float));
    pipe.InitBuffer(reduceDstBuf, BLOCK_SIZE_B32 * sizeof(float));
    // ReduceSum sharedTmp: numColAlign * sizeof(float) is a safe upper bound.
    pipe.InitBuffer(tmpBuf, numColAlign * sizeof(float));

    // gamma 常驻: load once.
    auto gammaLocal = gammaQueue.AllocTensor<T>();
    AscendC::DataCopyExtParams gammaCp{1, static_cast<uint32_t>(numCol * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> gammaPp{false, 0, 0, 0};
    AscendC::DataCopyPad(gammaLocal, gammaGm[0], gammaCp, gammaPp);
    gammaQueue.EnQue(gammaLocal);
    gammaLocal = gammaQueue.DeQue<T>();

    // Copy params for multi-row tiles. UB rows are laid out at numColAlign-element
    // (32B-aligned) pitches; DataCopyPad auto-pads each blockLen up to 32B, so the
    // numColAlign spacing is achieved with ZERO extra stride when
    // (numColAlign-numCol)*sizeof(T) < 32 (always true: gap < one 32B block).
    // srcStride/dstStride units are 32-BYTE BLOCKS (matches golden rms_norm_a5
    // dstStride=(C'-numCol)*s_x/32), NOT bytes. Passing raw gap bytes (e.g. 30)
    // made the API scatter rows by 30*32=960B and corrupt non-aligned cases.
    uint32_t gapBytes = static_cast<uint32_t>((numColAlign - numCol) * static_cast<int64_t>(sizeof(T)));
    uint32_t gapBlocks = gapBytes / 32;   // 32B-block gap; 0 when gap<32 (all current cases)
    uint32_t blockLenBytes = static_cast<uint32_t>(numCol * sizeof(T));
    AscendC::DataCopyPadExtParams<T> rowPp{false, 0, 0, 0};

    auto xFp32Local = xFp32Buf.Get<float>();
    auto sqFp32Local = sqFp32Buf.Get<float>();
    auto rstdLocal = rstdBuf.Get<float>();
    auto reduceDstLocal = reduceDstBuf.Get<float>();
    auto tmpLocal = tmpBuf.Get<uint8_t>();

    int64_t tiles = (rowCount + m - 1) / m;

    // ── Prologue: load tile 0 (issue MTE2 for first tile) ──
    int64_t curM0 = (m < rowCount) ? m : rowCount;
    auto xCur = xQueue.AllocTensor<T>();
    AscendC::DataCopyExtParams copyIn0{static_cast<uint16_t>(curM0), blockLenBytes,
                                       0, gapBlocks, 0};
    AscendC::DataCopyPad(xCur, xGm[0], copyIn0, rowPp);
    xQueue.EnQue(xCur);

    for (int64_t t = 0; t < tiles; ++t) {
        int64_t curM = (t * m + m <= rowCount) ? m : (rowCount - t * m);
        // DeQue waits for MTE2 of tile t to complete.
        xCur = xQueue.DeQue<T>();

        // Prefetch next tile BEFORE compute so MTE2(t+1) overlaps V(t).
        if (t + 1 < tiles) {
            int64_t nextM = ((t + 1) * m + m <= rowCount) ? m : (rowCount - (t + 1) * m);
            auto xNext = xQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams copyInNext{static_cast<uint16_t>(nextM), blockLenBytes,
                                                  0, gapBlocks, 0};
            AscendC::DataCopyPad(xNext, xGm[(t + 1) * m * numCol], copyInNext, rowPp);
            xQueue.EnQue(xNext);
        }

        // Allocate y tile (VECOUT); compute each row into it.
        auto yTile = yQueue.AllocTensor<T>();

        for (int64_t r = 0; r < curM; ++r) {
            uint32_t rowOff = static_cast<uint32_t>(r * numColAlign);
            auto xRow = xCur[rowOff];        // VECIN row r (T)
            auto yRow = yTile[rowOff];       // VECOUT row r (T)

            // 1-2. Cast x -> fp32 then sq = x*x. fp32: compute directly on xRow.
            if constexpr (AscendC::IsSameType<T, float>::value) {
                AscendC::Mul(sqFp32Local, xRow, xRow, numCol);
            } else {
                AscendC::Cast(xFp32Local, xRow, AscendC::RoundMode::CAST_NONE, numCol);
                AscendC::Mul(sqFp32Local, xFp32Local, xFp32Local, numCol);
            }
            // 3. Sum sq -> scalar. AR 2D pattern, single row.
            uint32_t srcShape[2] = {1, static_cast<uint32_t>(numCol)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(reduceDstLocal, sqFp32Local,
                                                                   tmpLocal, srcShape, false);
            float sum = reduceDstLocal.GetValue(0);
            // 4. rstd = 1/sqrt(avg*sum + eps), inf/zero guards (scalar C++ math).
            float var = sum * avgFactor + epsilon;
            float rstd;
            if (var == std::numeric_limits<float>::infinity()) {
                rstd = 0.0f;  // var==+inf -> rstd=0
            } else if (var == 0.0f) {
                rstd = std::numeric_limits<float>::infinity();  // var==0 -> rstd=+inf
            } else {
                rstd = 1.0f / std::sqrt(var);
            }
            // 5-6. y = x * rstd * gamma. fp32: compute straight into yRow.
            //      non-fp32: compute in xFp32Local, cast down to yRow.
            if constexpr (AscendC::IsSameType<T, float>::value) {
                AscendC::Muls(yRow, xRow, rstd, numCol);
                AscendC::Mul(yRow, yRow, gammaLocal, numCol);
            } else {
                AscendC::Muls(xFp32Local, xFp32Local, rstd, numCol);
                AscendC::Cast(sqFp32Local, gammaLocal, AscendC::RoundMode::CAST_NONE, numCol);
                AscendC::Mul(xFp32Local, xFp32Local, sqFp32Local, numCol);
                AscendC::Cast(yRow, xFp32Local, AscendC::RoundMode::CAST_ROUND, numCol);
            }
        }

        yQueue.EnQue(yTile);
        auto yOut = yQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyOut{static_cast<uint16_t>(curM), blockLenBytes,
                                           gapBlocks, 0, 0};
        AscendC::DataCopyPad(yGm[t * m * numCol], yOut, copyOut);

        xQueue.FreeTensor(xCur);
        yQueue.FreeTensor(yOut);
    }

    gammaQueue.FreeTensor(gammaLocal);
}
```

## 预防

构造多行 `DataCopyPad` 参数时，分别标注 `blockLen` 的字节单位与 `srcStride`/`dstStride` 的 32B block 单位；至少用一个非对齐行宽检查相邻两行的 UB 起点。

## 验证信号
均为 Ascend950PR_957c / `dav-3510` 真机上的 `tool_observed` 结果：

1. 修复前，对齐 float16 定向用例通过；非对齐 bfloat16 定向用例失败，报告 `MERE nan >= 7.812e-03`。
2. 只把 copy-in、prefetch、copy-out 的 stride 从 `rowGapBytes` 改为 `gapBlocks` 后，完整构建输出 `BUILD_OK`。
3. 修复后四个定向用例全部通过；其中此前失败的非对齐 bfloat16 路径恢复，`MERE=6.505e-08`、`MARE=7.812e-03`，另一个非对齐 bfloat16 路径也通过。
4. 随后完整精度套件 20 个用例全部通过，原阈值和原 golden 均未放宽。

## 设计决策

- `blockLen=valid_bytes, stride=gap_bytes/32`：选用。与固定 golden 公式一致，并通过非对齐定向、完整精度和最终评测。
- `blockLen=valid_bytes, stride=raw_gap_bytes`：禁用。raw gap 会被按 32B block 数解释，破坏 UB 多行 pitch。
- `退回逐行 blockCount=1`：未采用。它能避开多行 stride，但会丢失本轮批处理优化，且没有必要；正确单位已经闭环。

## 失效边界

本结论只在 950 / CANN 9.0.1 的该重载上完成验证。其他架构或版本未验证，迁移前必须查目标头文件或运行最小多行 DMA probe。

## 坏实践

不要因为 `blockLen` 使用字节，就推断 `DataCopyExtParams` 的 `srcStride`/`dstStride` 也使用字节。多行 `DataCopyPad` 的字段必须逐个核对单位，并让 copy-in、prefetch、copy-out 使用同一套 pitch 约定；只修其中一处会造成输入和输出布局不一致。

## 误诊

- 不是 bfloat16 cast 的一般性精度问题：对齐 bfloat16 路径在相同 cast/reduce 代码下通过。
- 不是 ReduceSum 对非对齐长度必然失效：只改 DMA stride 后原 ReduceSum 代码即通过原失败用例。
- 不是测试容差过严：故障信号是 `MERE=nan`，修复后无需改阈值即可恢复到基线误差水平。
- 不把宽泛批跑中的 ACL `507035` 单独归因于本根因：该批跑同时匹配了更多用例并可能污染设备上下文，主闭环以可重复 NaN 为准。

## 性能影响

该单位修复没有隔离性能 A/B；不得把同时包含多行 tile、双缓冲和多核调整的端到端变化归因于 stride 单位修复。