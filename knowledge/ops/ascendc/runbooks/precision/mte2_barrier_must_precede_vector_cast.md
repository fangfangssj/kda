---
type: "Runbook"
title: "DataCopyPad 后未先同步 Cast 导致低精度 NaN"
description: "MTE2 搬入后把 PipeBarrier 放在 Cast 之后过晚；应先同步，再由 Vector Cast 消费 LocalTensor。"
tags: ["task.debugging", "topic.precision", "topic.implementation.synchronization", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# DataCopyPad 后未先同步 Cast 导致低精度 NaN

## 现象

为 fp32 增加直通分支后，fp32 探针已通过，但 fp16 与 bfloat16 探针开始出现 NaN 或大幅错误。代码在低精度分支中先 `DataCopyPad(gammaLocal, gammaGm, ...)`，紧接着执行 `Cast(gammaFp32, gammaLocal, ...)`，统一的 `PipeBarrier<PIPE_ALL>` 却放在整个 if/else 之后。屏障只能约束后续消费者，无法补救已经提前读取 `gammaLocal` 的 Cast。

## 适用场景

- 工程形态：Ascend C kernel 直调，TBuf/LocalTensor 手动管理，不通过 TQue EnQue/DeQue 自动同步。
- API 路径：GM→UB `DataCopyPad`（MTE2）后由 `Cast`（Vector）立即消费。
- 验证平台与版本：Ascend 950，CANN 9.0.1。
- 适用粒度：`operator_family`，适用于归约/归一化算子中 MTE2 producer→Vector consumer 的手动缓冲链；跨算子族仍需验证。
- 路线优先级：`recommended_fastpath`；producer 与 consumer 之间建立同步是正确性必需项。
- 不适用/未验证：TQue 已通过 EnQue/DeQue 建立依赖的路径、MTE3 方向及其他平台未验证。

## 根因

`DataCopyPad` 的 GM→UB 搬运由 MTE2 异步执行，而 `Cast` 在 Vector pipeline 消费 `gammaLocal`。正确 RAW 依赖是 `MTE2 write gammaLocal → barrier/event → VEC read gammaLocal`。把屏障放到 `Cast` 之后，只保证 Cast 之后的操作等待，不能让已经发出的 Cast 倒退等待 DMA，因此 Vector 读取到未完成/旧数据并传播为 NaN 或大幅偏差。

## 解决配方

### before

**失败状态（屏障晚于 Cast）**：revision=`before_fix_snapshot`；path=`example/csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp`；symbol=`rms_norm_kernel<T>`；lines=29-169。

```cpp
template <typename T>
__global__ __aicore__ void rms_norm_kernel(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t A1, int64_t D, float epsilon, float invD, int64_t blockRows)
{
    int64_t rowBegin = AscendC::GetBlockIdx() * blockRows;
    int64_t rowEnd = rowBegin + blockRows;
    if (rowEnd > A1) rowEnd = A1;
    if (rowBegin >= rowEnd) return;

    // ---- Buffer 尺寸推导（32B 对齐）----
    const uint32_t sizeofT = static_cast<uint32_t>(sizeof(T));
    const uint32_t dAlignBytesT = (static_cast<uint32_t>(D) * sizeofT + 31u) & ~31u;
    const uint32_t dAlignBytesF32 = (static_cast<uint32_t>(D) * 4u + 31u) & ~31u;
    const uint32_t dAlignF32Elem = dAlignBytesF32 / 4u;
    // ReduceSum sharedTmpBuffer 最小尺寸（fp32）
    constexpr uint32_t perRepeat = 256u / 4u;   // 64
    constexpr uint32_t perBlock = 32u / 4u;     // 8
    uint32_t repeats = (dAlignF32Elem + perRepeat - 1u) / perRepeat;
    uint32_t tmpBufSize = ((repeats + perBlock - 1u) / perBlock) * perBlock * 4u;
    if (tmpBufSize < 4096u) tmpBufSize = 4096u;
    // sumBuf 保守取 256B（ReduceSum dst 对齐/中间结果覆盖），32B 对齐
    constexpr uint32_t sumBufSize = 256u;

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, gammaGm, yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaLoadBuf, gammaFp32Buf, xSqBuf, sumBuf, tmpBuf, sqrtBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf; // 仅 fp16/bf16 使用

    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, dAlignBytesT);
    pipe.InitBuffer(outQueueY, PIPELINE_DEPTH, dAlignBytesT);
    pipe.InitBuffer(gammaLoadBuf, dAlignBytesT);
    pipe.InitBuffer(gammaFp32Buf, dAlignBytesF32);
    pipe.InitBuffer(xSqBuf, dAlignBytesF32);
    pipe.InitBuffer(sumBuf, sumBufSize);
    pipe.InitBuffer(tmpBuf, tmpBufSize);
    pipe.InitBuffer(sqrtBuf, 32u); // rstd 标量开方工作区（8 个 fp32，32B 对齐）
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(xFp32Buf, dAlignBytesF32);
    }

    // ---- Init: gamma 一次性搬入 + (非 fp32)Cast 到 fp32（跨行复用）----
    {
        AscendC::DataCopyExtParams gcp{1, static_cast<uint32_t>(D) * sizeofT, 0, 0, 0};
        auto gammaFp32 = gammaFp32Buf.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            // fp32: gamma 已是 float，直接搬入 gammaFp32Buf，无需 Cast（float->float Cast 非法）
            AscendC::DataCopyPadExtParams<float> gppF{false, 0, 0, 0};
            AscendC::DataCopyPad(gammaFp32, gammaGm, gcp, gppF);
        } else {
            AscendC::DataCopyPadExtParams<T> gpp{false, 0, 0, 0};
            auto gammaLocal = gammaLoadBuf.Get<T>();
            AscendC::DataCopyPad(gammaLocal, gammaGm, gcp, gpp);
            AscendC::Cast(gammaFp32, gammaLocal, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(D));
        }
        AscendC::PipeBarrier<PIPE_ALL>(); // MTE2 -> VEC 同步，确保 gammaFp32 就绪
    }
    auto gammaFp32 = gammaFp32Buf.Get<float>();

    const AscendC::DataCopyExtParams inCopy{1, static_cast<uint32_t>(D) * sizeofT, 0, 0, 0};
    const AscendC::DataCopyExtParams outCopy{1, static_cast<uint32_t>(D) * sizeofT, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<T> inPad{false, 0, 0, 0};

    // ---- 软件流水：预取首行 ----
    {
        int64_t off = rowBegin * D;
        auto xl = inQueueX.AllocTensor<T>();
        AscendC::DataCopyPad(xl, xGm[off], inCopy, inPad);
        inQueueX.EnQue(xl);
    }

    for (int64_t r = rowBegin; r < rowEnd; ++r) {
        // 预取下一行（与本行 compute 的 MTE2/Vector 行间重叠）
        if (r + 1 < rowEnd) {
            int64_t off = (r + 1) * D;
            auto xNext = inQueueX.AllocTensor<T>();
            AscendC::DataCopyPad(xNext, xGm[off], inCopy, inPad);
            inQueueX.EnQue(xNext);
        }

        auto xLocal = inQueueX.DeQue<T>();
        int64_t off = r * D;

        if constexpr (std::is_same<T, float>::value) {
            // fp32: xLocal 即 fp32 工作区；xSq 独立，保留原始 x
            auto xSq = xSqBuf.Get<float>();
            AscendC::Mul(xSq, xLocal, xLocal, static_cast<int32_t>(D));
            auto sumLocal = sumBuf.Get<float>();
            auto tmpLocal = tmpBuf.Get<float>();
            AscendC::ReduceSum(sumLocal, xSq, tmpLocal, static_cast<int32_t>(D));
            // rstd = 1/sqrt(meanSq+eps)：全程向量计算规避 libm sqrtf 不可用，
            // 并以 PipeBarrier 在 GetValue 前同步 VEC->标量（GetValue 不自动同步）。
            // sumLocal[0]=sumSq -> Muls(invD) -> Adds(eps) -> Sqrt -> 标量倒数 -> Muls 广播
            AscendC::Muls(sumLocal, sumLocal, invD, 8);
            AscendC::Adds(sumLocal, sumLocal, epsilon, 8);
            auto sqrtLocal = sqrtBuf.Get<float>();
            AscendC::Sqrt(sqrtLocal, sumLocal, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            float rstd = 1.0f / sqrtLocal.GetValue(0);
            AscendC::Muls(xLocal, xLocal, rstd, static_cast<int32_t>(D));
            AscendC::Mul(xLocal, xLocal, gammaFp32, static_cast<int32_t>(D));
            // VEC -> VECOUT 搬运（Adds+0 等价拷贝，避免 UB->UB DataCopy 对齐灰区）
            auto yLocal = outQueueY.AllocTensor<T>();
            AscendC::Adds(yLocal, xLocal, 0.0f, static_cast<int32_t>(dAlignF32Elem));
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
            yLocal = outQueueY.DeQue<T>();
            AscendC::DataCopyPad(yGm[off], yLocal, outCopy);
            outQueueY.FreeTensor(yLocal);
        } else {
            auto xFp32 = xFp32Buf.Get<float>();
            AscendC::Cast(xFp32, xLocal, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(D));
            auto xSq = xSqBuf.Get<float>();
            AscendC::Mul(xSq, xFp32, xFp32, static_cast<int32_t>(D));
            auto sumLocal = sumBuf.Get<float>();
            auto tmpLocal = tmpBuf.Get<float>();
            AscendC::ReduceSum(sumLocal, xSq, tmpLocal, static_cast<int32_t>(D));
            AscendC::Muls(sumLocal, sumLocal, invD, 8);
            AscendC::Adds(sumLocal, sumLocal, epsilon, 8);
            auto sqrtLocal = sqrtBuf.Get<float>();
            AscendC::Sqrt(sqrtLocal, sumLocal, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            float rstd = 1.0f / sqrtLocal.GetValue(0);
            AscendC::Muls(xFp32, xFp32, rstd, static_cast<int32_t>(D));
            AscendC::Mul(xFp32, xFp32, gammaFp32, static_cast<int32_t>(D));
            auto yLocal = outQueueY.AllocTensor<T>();
            AscendC::Cast(yLocal, xFp32, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(D));
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
            yLocal = outQueueY.DeQue<T>();
            AscendC::DataCopyPad(yGm[off], yLocal, outCopy);
            outQueueY.FreeTensor(yLocal);
        }
    }
}
```

### after

**验证通过状态（屏障位于 producer 与 consumer 之间）**：revision=`after_fix_validated_snapshot`；path=`example/csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp`；symbol=`rms_norm_kernel<T>`；lines=29-170。

```cpp
template <typename T>
__global__ __aicore__ void rms_norm_kernel(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t A1, int64_t D, float epsilon, float invD, int64_t blockRows)
{
    int64_t rowBegin = AscendC::GetBlockIdx() * blockRows;
    int64_t rowEnd = rowBegin + blockRows;
    if (rowEnd > A1) rowEnd = A1;
    if (rowBegin >= rowEnd) return;

    // ---- Buffer 尺寸推导（32B 对齐）----
    const uint32_t sizeofT = static_cast<uint32_t>(sizeof(T));
    const uint32_t dAlignBytesT = (static_cast<uint32_t>(D) * sizeofT + 31u) & ~31u;
    const uint32_t dAlignBytesF32 = (static_cast<uint32_t>(D) * 4u + 31u) & ~31u;
    const uint32_t dAlignF32Elem = dAlignBytesF32 / 4u;
    // ReduceSum sharedTmpBuffer 最小尺寸（fp32）
    constexpr uint32_t perRepeat = 256u / 4u;   // 64
    constexpr uint32_t perBlock = 32u / 4u;     // 8
    uint32_t repeats = (dAlignF32Elem + perRepeat - 1u) / perRepeat;
    uint32_t tmpBufSize = ((repeats + perBlock - 1u) / perBlock) * perBlock * 4u;
    if (tmpBufSize < 4096u) tmpBufSize = 4096u;
    // sumBuf 保守取 256B（ReduceSum dst 对齐/中间结果覆盖），32B 对齐
    constexpr uint32_t sumBufSize = 256u;

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, gammaGm, yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaLoadBuf, gammaFp32Buf, xSqBuf, sumBuf, tmpBuf, sqrtBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf; // 仅 fp16/bf16 使用

    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, dAlignBytesT);
    pipe.InitBuffer(outQueueY, PIPELINE_DEPTH, dAlignBytesT);
    pipe.InitBuffer(gammaLoadBuf, dAlignBytesT);
    pipe.InitBuffer(gammaFp32Buf, dAlignBytesF32);
    pipe.InitBuffer(xSqBuf, dAlignBytesF32);
    pipe.InitBuffer(sumBuf, sumBufSize);
    pipe.InitBuffer(tmpBuf, tmpBufSize);
    pipe.InitBuffer(sqrtBuf, 32u); // rstd 标量开方工作区（8 个 fp32，32B 对齐）
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(xFp32Buf, dAlignBytesF32);
    }

    // ---- Init: gamma 一次性搬入 + (非 fp32)Cast 到 fp32（跨行复用）----
    {
        AscendC::DataCopyExtParams gcp{1, static_cast<uint32_t>(D) * sizeofT, 0, 0, 0};
        auto gammaFp32 = gammaFp32Buf.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            // fp32: gamma 已是 float，直接搬入 gammaFp32Buf，无需 Cast（float->float Cast 非法）
            AscendC::DataCopyPadExtParams<float> gppF{false, 0, 0, 0};
            AscendC::DataCopyPad(gammaFp32, gammaGm, gcp, gppF);
            AscendC::PipeBarrier<PIPE_ALL>(); // MTE2 -> VEC 同步
        } else {
            AscendC::DataCopyPadExtParams<T> gpp{false, 0, 0, 0};
            auto gammaLocal = gammaLoadBuf.Get<T>();
            AscendC::DataCopyPad(gammaLocal, gammaGm, gcp, gpp);
            AscendC::PipeBarrier<PIPE_ALL>(); // MTE2 -> VEC 同步
            AscendC::Cast(gammaFp32, gammaLocal, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(D));
        }
    }
    auto gammaFp32 = gammaFp32Buf.Get<float>();

    const AscendC::DataCopyExtParams inCopy{1, static_cast<uint32_t>(D) * sizeofT, 0, 0, 0};
    const AscendC::DataCopyExtParams outCopy{1, static_cast<uint32_t>(D) * sizeofT, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<T> inPad{false, 0, 0, 0};

    // ---- 软件流水：预取首行 ----
    {
        int64_t off = rowBegin * D;
        auto xl = inQueueX.AllocTensor<T>();
        AscendC::DataCopyPad(xl, xGm[off], inCopy, inPad);
        inQueueX.EnQue(xl);
    }

    for (int64_t r = rowBegin; r < rowEnd; ++r) {
        // 预取下一行（与本行 compute 的 MTE2/Vector 行间重叠）
        if (r + 1 < rowEnd) {
            int64_t off = (r + 1) * D;
            auto xNext = inQueueX.AllocTensor<T>();
            AscendC::DataCopyPad(xNext, xGm[off], inCopy, inPad);
            inQueueX.EnQue(xNext);
        }

        auto xLocal = inQueueX.DeQue<T>();
        int64_t off = r * D;

        if constexpr (std::is_same<T, float>::value) {
            // fp32: xLocal 即 fp32 工作区；xSq 独立，保留原始 x
            auto xSq = xSqBuf.Get<float>();
            AscendC::Mul(xSq, xLocal, xLocal, static_cast<int32_t>(D));
            auto sumLocal = sumBuf.Get<float>();
            auto tmpLocal = tmpBuf.Get<float>();
            AscendC::ReduceSum(sumLocal, xSq, tmpLocal, static_cast<int32_t>(D));
            // rstd = 1/sqrt(meanSq+eps)：全程向量计算规避 libm sqrtf 不可用，
            // 并以 PipeBarrier 在 GetValue 前同步 VEC->标量（GetValue 不自动同步）。
            // sumLocal[0]=sumSq -> Muls(invD) -> Adds(eps) -> Sqrt -> 标量倒数 -> Muls 广播
            AscendC::Muls(sumLocal, sumLocal, invD, 8);
            AscendC::Adds(sumLocal, sumLocal, epsilon, 8);
            auto sqrtLocal = sqrtBuf.Get<float>();
            AscendC::Sqrt(sqrtLocal, sumLocal, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            float rstd = 1.0f / sqrtLocal.GetValue(0);
            AscendC::Muls(xLocal, xLocal, rstd, static_cast<int32_t>(D));
            AscendC::Mul(xLocal, xLocal, gammaFp32, static_cast<int32_t>(D));
            // VEC -> VECOUT 搬运（Adds+0 等价拷贝，避免 UB->UB DataCopy 对齐灰区）
            auto yLocal = outQueueY.AllocTensor<T>();
            AscendC::Adds(yLocal, xLocal, 0.0f, static_cast<int32_t>(dAlignF32Elem));
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
            yLocal = outQueueY.DeQue<T>();
            AscendC::DataCopyPad(yGm[off], yLocal, outCopy);
            outQueueY.FreeTensor(yLocal);
        } else {
            auto xFp32 = xFp32Buf.Get<float>();
            AscendC::Cast(xFp32, xLocal, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(D));
            auto xSq = xSqBuf.Get<float>();
            AscendC::Mul(xSq, xFp32, xFp32, static_cast<int32_t>(D));
            auto sumLocal = sumBuf.Get<float>();
            auto tmpLocal = tmpBuf.Get<float>();
            AscendC::ReduceSum(sumLocal, xSq, tmpLocal, static_cast<int32_t>(D));
            AscendC::Muls(sumLocal, sumLocal, invD, 8);
            AscendC::Adds(sumLocal, sumLocal, epsilon, 8);
            auto sqrtLocal = sqrtBuf.Get<float>();
            AscendC::Sqrt(sqrtLocal, sumLocal, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            float rstd = 1.0f / sqrtLocal.GetValue(0);
            AscendC::Muls(xFp32, xFp32, rstd, static_cast<int32_t>(D));
            AscendC::Mul(xFp32, xFp32, gammaFp32, static_cast<int32_t>(D));
            auto yLocal = outQueueY.AllocTensor<T>();
            AscendC::Cast(yLocal, xFp32, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(D));
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
            yLocal = outQueueY.DeQue<T>();
            AscendC::DataCopyPad(yGm[off], yLocal, outCopy);
            outQueueY.FreeTensor(yLocal);
        }
    }
}
```

## 设计决策

- fp32 直通和低精度转换分别放置同步点，保证每条编译期分支自身闭合 producer→consumer 依赖。
- 低精度分支顺序固定为 `DataCopyPad → PipeBarrier → Cast`；不使用“分支外统一屏障”掩盖分支内部依赖。
- 若改用 TQue，应依赖 EnQue/DeQue 的同步语义并删除重复手工屏障；本次 TBuf 路线保留显式屏障。

## 预防

- 画出每个 LocalTensor 的 producer pipeline 与 consumer pipeline，屏障必须位于二者之间。
- 审查时拒绝“操作完成后补一个 barrier”的形式化同步；检查屏障前后实际指令。
- dtype 分支修改后必须回归所有分支，不能只复测刚修复的 fp32。

## 验证信号

来源于一次 Ascend 950 直调 kernel 的分 dtype 回归；屏障位置修改前后使用同一远程构建与探针路径。

- 屏障放在 Cast 之后时，fp32 探针通过，但 fp16/bfloat16 多组探针出现 NaN 或大幅错误（证据等级: tool_observed；来源平台: 950）。
- 把屏障移到低精度 `DataCopyPad` 与 `Cast` 之间后，远程构建成功，fp32/fp16/bfloat16 探针全部通过（证据等级: tool_observed；来源平台: 950）。
- 随后的完整评测达到 20/20 精度通过（证据等级: tool_observed；来源平台: 950）。

## 失效边界

- 结论针对 TBuf 上的 MTE2→VEC 手动依赖；TQue 或其他 API 可能自带同步，不能无条件叠加 `PIPE_ALL`。
- A2/A3 未验证，推断可迁移但必须在目标平台确认同步原语。
- 本卡不证明 `PIPE_ALL` 是最小成本方案，只证明“同步必须发生在 producer 与 consumer 之间”。

## 坏实践

- 在 if/else 之后放一个统一屏障，却让分支内部的 Vector consumer 已经先执行。
- 只回归 fp32 直通分支，忽略低精度 Cast 分支的同步变化。

## 误诊

错误在修复 fp32 Cast 后出现，容易认为新 dtype 分支本身有数值问题；但 NaN 仅发生在仍需 MTE2→VEC 的低精度路径，且移动屏障立即闭环，指向同步位置而非舍入精度。