---
type: "Runbook"
title: "bisheng 不接受 device 侧 sqrtf—改用向量 Sqrt"
description: device kernel 中直接调用 sqrtf 导致 bisheng 编译失败；两条已验证路线：32B 对齐的向量 Sqrt 链 + 显式同步，与 SIMT 标量场景的 bisheng CCE 内建 sqrt（__builtin_cce_sqrtf 单条指令）。复现增量：TBE/opc 算子仓工具链形态报 use of undeclared identifier sqrtf（did you mean sqrt）——标量场景直接用 sqrt（double 版）+ static_cast 收回 float 即可编译通过，与内建路线同名同效。复现增量：GroupNorm（TBE 链，CANN 9.1.0）同型报错——编译器 note 直指 CCE 内建 float sqrt 声明，直接小写 sqrt 即过，无须 double 往返还原。
tags: ["task.debugging", "topic.compute.elementwise.math_function", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: '2026-09-17T05:05:01Z'
platforms: ["950"]
---

# bisheng 不接受 device 侧 sqrtf—改用向量 Sqrt

## 现象

在 kernel 直调工程中编译归约后开方的 device 代码时，`op_kernel/rms_norm_kernel.cpp` 在 `-xasc` 编译阶段失败。错误尾部指向 `/usr/include/math.h` 的宏展开与 `sqrtf`，并报告 `3 errors generated`，随后 `all_kernels_obj` 构建失败。当时代码在 `__aicore__` 函数内直接执行 `1.0f / sqrtf(sumSq / D + epsilon)`。

## 适用场景

- 工程形态：Ascend C kernel 直调工程，bisheng 编译 device 源文件。
- API 路径：归约结果经 `AscendC::Sqrt`、`PipeBarrier<PIPE_ALL>`、`LocalTensor::GetValue` 转成标量倒数。
- 验证平台与版本：Ascend 950，活动 CANN toolkit 路径为 9.0.1。
- 适用粒度：`operator_family`，适用于需要在 device kernel 中对归约标量做平方根/倒数的归一化与统计类算子。
- 路线优先级：`recommended_fastpath`；这是本轨迹中唯一完成编译与全量精度验证的实现形态，不代表其他平台上的全局最优实现。
- 不适用场景：Host 侧普通 C++ 可正常链接 libm 的代码；RegBase 及其他平台工具链未验证；SIMT 标量场景已在 950/CANN 9.2.0 经 CCE 内建 sqrt 验证（见下方增量与验证信号）。
- SIMT/标量路线（独立复现 R3，BenchHarnessEngramGateFusion，CANN 9.2.0）：`AscendC::Sqrt` 仅重载 LocalTensor 张量签名（3 参/5 参、返回 `void` 不能赋给 `float`），标量场景 no matching；可用入口是 bisheng CCE 内建头 `__clang_cce_aicore_functions.h` 的标量 `sqrt`（`CCE_INTRINSIC CALLEE_ATTR float sqrt(float in) { return __builtin_cce_sqrtf(in); }`）。
- 选用条件（标量 vs 向量链）：开方作用在标量归约结果上（每行一次）时优先 CCE 内建 `sqrt`，免取向量链往返（`sed` 删 `<cmath>`、`sqrtf(`→`sqrt(`）；张量级开方仍直接用 `AscendC::Sqrt(LocalTensor, LocalTensor, count)` 向量 API。判别探针：`grep -rn "sqrt\b" <bisheng 安装目录>/lib/clang/15.0.5/include/__clang_cce*.h`，确认存在 `CCE_INTRINSIC`/`CALLEE_ATTR` 声明再用；glibc `<cmath>`/`math.h` 的声明是 `__host__`，一律不可在 `__aicore__` 调用。

## 根因

bisheng 的 device 编译环境不是普通 Host C++ 链接环境，`__aicore__` 内直接使用 libc/libm 的 `sqrtf` 会在头文件宏展开/设备符号解析阶段失败；应把 `mean+epsilon` 写入 LocalTensor，通过 Ascend C Vector API 完成开方。计算链必须明确为 `meanSq = sumSq × invD`、`stabilized = meanSq + epsilon`、`sqrt = Sqrt(stabilized)`、`rstd = 1 / sqrt`，其中 8 个 fp32 元素恰为 `8×4B=32B`，满足该实现使用的对齐粒度。

## 解决配方

### before

**失败状态**：revision=`before_fix_snapshot`；path=`example/csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp`；symbol=`rms_norm_kernel<T>`；lines=30-152。

```cpp
template <typename T>
__global__ __aicore__ void rms_norm_kernel(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t A1, int64_t D, float epsilon, int64_t blockRows)
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaLoadBuf, gammaFp32Buf, xSqBuf, sumBuf, tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf; // 仅 fp16/bf16 使用

    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, dAlignBytesT);
    pipe.InitBuffer(outQueueY, PIPELINE_DEPTH, dAlignBytesT);
    pipe.InitBuffer(gammaLoadBuf, dAlignBytesT);
    pipe.InitBuffer(gammaFp32Buf, dAlignBytesF32);
    pipe.InitBuffer(xSqBuf, dAlignBytesF32);
    pipe.InitBuffer(sumBuf, sumBufSize);
    pipe.InitBuffer(tmpBuf, tmpBufSize);
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(xFp32Buf, dAlignBytesF32);
    }

    // ---- Init: gamma 一次性搬入 + Cast 到 fp32（跨行复用）----
    {
        AscendC::DataCopyExtParams gcp{1, static_cast<uint32_t>(D) * sizeofT, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> gpp{false, 0, 0, 0};
        auto gammaLocal = gammaLoadBuf.Get<T>();
        AscendC::DataCopyPad(gammaLocal, gammaGm, gcp, gpp);
        AscendC::PipeBarrier<AscendC::PIPE_ALL>(); // MTE2 -> VEC 同步
        auto gammaFp32 = gammaFp32Buf.Get<float>();
        AscendC::Cast(gammaFp32, gammaLocal, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(D));
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
            float sumSq = sumLocal.GetValue(0);
            float rstd = 1.0f / sqrtf(sumSq / static_cast<float>(D) + epsilon);
            AscendC::Muls(xLocal, xLocal, rstd, static_cast<int32_t>(D));
            AscendC::Mul(xLocal, xLocal, gammaFp32, static_cast<int32_t>(D));
            // VEC -> VECOUT 搬运（对齐 count），经 outQueueY 同步后 CopyOut
            auto yLocal = outQueueY.AllocTensor<T>();
            AscendC::DataCopy(yLocal, xLocal, static_cast<int32_t>(dAlignF32Elem));
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
            float sumSq = sumLocal.GetValue(0);
            float rstd = 1.0f / sqrtf(sumSq / static_cast<float>(D) + epsilon);
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

**验证通过状态**：revision=`after_fix_validated_snapshot`；path=`example/csrc/ops/rms_norm/op_kernel/rms_norm_kernel.cpp`；symbol=`rms_norm_kernel<T>`；lines=29-170。

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

- 不在构建系统中尝试为 device kernel 强行补 Host libm 链接，而是改用目标平台支持的 `AscendC::Sqrt`。
- `invD` 由 Host 计算后传入，device 侧用 `Muls` 与 `Adds` 形成完整 Vector 链。
- `Sqrt` 后保留 `PipeBarrier<PIPE_ALL>` 再 `GetValue`；同一轨迹的删屏障对照实验出现大面积精度失败，说明不能用“同为 VEC 流水”推断标量读天然可见。
- 当前路线是已验证可用实现；没有证据支持把 `sqrtf` 失败归因为缺少普通链接参数。

## 性能影响

轨迹没有隔离测量“向量 Sqrt 替代 sqrtf”本身的性能成本，因此不宣称收益。后续批量标量同步实验未形成整体净收益，不能作为本卡的性能结论。

## 预防

- 在设计阶段把 device 数学函数逐个映射到 Ascend C API，不把 Host libc/libm 可用性外推到 `__aicore__`。
- 标量结果若来自 Vector 指令，先验证同步契约，再读取 `GetValue`。
- 对标量工作区按 API 粒度分配，当前验证形态为 32B 对齐。

## 验证信号

来源于一次 Ascend 950 直调归一化 kernel 的远程构建与上板评测；代码状态由轨迹中的完整写入与逐次编辑重建。

- 初始版本在 `sqrtf` 的 `math.h` 宏展开处报告 3 个编译错误，`all_kernels_obj` 失败（证据等级: tool_observed；来源平台: 950）。
- 改为向量计算链并完成同步/Host 参数配套后，远程 CMake 构建到 `_C` 目标成功（证据等级: tool_observed；来源平台: 950）。
- 最终实现通过完整 20/20 精度用例，覆盖 fp32、fp16 与 bfloat16（证据等级: tool_observed；来源平台: 950）。

- 独立复现 R1：工具报告 [100%] Built target _C。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3（BenchHarnessEngramGateFusion，CANN 9.2.0）：`AscendC::Sqrt(scalar)` 报 `no matching function for call to 'Sqrt'` + `cannot initialize a variable of type 'float' with an rvalue of type 'void'`；`sqrtf` + `<cmath>` 报 `call to __host__ function from __aicore__ function`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3（BenchHarnessEngramGateFusion，CANN 9.2.0）：判别探针 grep 命中 `__clang_cce_aicore_functions.h:3517:CCE_INTRINSIC CALLEE_ATTR float sqrt(float in) { return __builtin_cce_sqrtf(in); }`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3（BenchHarnessEngramGateFusion，CANN 9.2.0）：改为 CCE 内建 `sqrt`（删 `<cmath>`、`sqrtf(`→`sqrt(`）后开方相关错误消失、编译推进到下一独立问题，最终 `bash build.sh` 输出 `SUCCESS`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只验证 Ascend 950 与本次活动 CANN 9.0.1 工具链；A2/A3 是否具有相同 `sqrtf` 限制未验证，推断可迁移但不得直接视为已验证。
- `count=8` 是本实现对 32B fp32 工作区的代码形态，不应在其他 dtype/接口签名上机械照搬。
- 其他数学函数不能仅凭本卡推断同样不可用，需分别查 API/编译验证。
- SIMT/内建路线验证于 950 + CANN 9.2.0（bisheng clang 15.0.5）；内建头路径与版本强相关，跨版本需重新探针确认。只实测了 `sqrt`/`sqrtf`；exp/log 等其他函数按『先查内建头』判别方法推广，未逐一验证。kernel 头文件不要 `#include <cmath>`——host 数学头会持续引入 `__host__` 声明，其他调用点还会踩同类错误。
- 相邻卡片：[__aicore__ 内 std::min/max 编译失败卡](aicore_avoid_host_std_minmax.md)——同族『host 标准库函数不能进 `__aicore__`』，min/max 用显式条件比较替代。[dav-2201 Rsqrt 精度不足卡](../precision/rsqrt_fast_approximation_use_sqrt_reciprocal.md)——开方可用之后的精度层问题：Rsqrt 近似偏差改用 Sqrt+标量倒数。

## 坏实践

- 在 `__aicore__` 里直接包含 `<cmath>` 并调用 `sqrtf`，然后把失败归咎于普通 Host 链接配置。
- 只把 `sqrtf` 换成 `AscendC::Sqrt`，却遗漏 LocalTensor 对齐、Vector→Scalar 同步和 Host 参数传递。

## 误诊

最初容易把报错理解成“缺少 `-lm`”；轨迹中的 device 编译上下文与后续 Ascend C API 修复表明，正确切入层是 kernel API 路径，不是通用链接参数。

## 增量（2026-09-17，先前增量 · TBE/opc 算子仓工具链形态：sqrtf 未声明，标量 sqrt + 显式收回 float）

另一独立轨迹并入第三条独立 trace（融合算子 bench_no_harness_add_rms_norm_dynamic_quant 的 RMSNorm 标量计算，Ascend950PR + CANN 9.1.0，`bash build.sh --soc=ascend950 -j8` 的 TBE 单算子编译链），补齐"错误形态 + 标量双精度路线"：

**1. 错误形态（undeclared identifier + 编译器自荐 sqrt）**：`__aicore__` 设备代码的 reduce 标量推导里按主机 C 习惯写 `sqrtf`：

```cpp
    float rms = sqrtf(variance + epsilon_);
```

kernel 编译（TBE CompileStage.INFERCHANNEL）报：

```
op_kernel/arch35/bench_no_harness_add_rms_norm_dynamic_quant.h:247:17:
error: use of undeclared identifier 'sqrtf'; did you mean 'sqrt'?
1 error generated.
```

与 bisheng 直调路线的 `math.h` 宏展开错误形态不同：此处错误直指标识符未声明，且编译器提示的 `sqrt`（double 版）**实测可用**——float 实参传入 double 形参自动提升，结果需显式收回 float。

**2. 修复（标量场景第三条路线，verbatim before/after）**：

```cpp
// before
    float variance = sumSq * invN_;
    float rms = sqrtf(variance + epsilon_);
    float invRms = (rms > 0.0f) ? (1.0f / rms) : 0.0f;
```

```cpp
// after
    float variance = sumSq * invN_;
    float rms = static_cast<float>(sqrt(variance + epsilon_));
    float invRms = (rms > 0.0f) ? (1.0f / rms) : 0.0f;
```

选 `sqrt` + `static_cast<float>` 而不是手写位级平方根：与 golden 的 `std::sqrt`（double 语义、RNE 舍回 float）语义一致，精度路径最短；备选"把 RMS 也搬进向量 API（Sqrt LocalTensor）"被放弃——reduce 输出是单标量，为此再起一次向量调用不划算。该写法与本卡 R3（CCE 内建 `sqrt`）同名同效：SIMT/直调场景按内建头探针确认后用 `sqrt`，算子仓 TBE 场景直接按编译器提示换 `sqrt`。

**3. 判别要点**：设备侧报 `use of undeclared identifier '<func>f'` 时直接按编译器提示换 `<func>`（double 版）并显式 `static_cast<float>`，不要去翻 math.h、也不要写 `f` 后缀单精度变体（logf/powf 等推断同不可用，未逐一实测）。

### 复现增量验证信号

- 修复前：kernel 编译报 `error: use of undeclared identifier 'sqrtf'; did you mean 'sqrt'?`（op_kernel/arch35 头 247 行，TBE CompileStage.INFERCHANNEL 失败）（来源平台：950；证据等级：tool_observed）。
- 修复后：`sqrt` + `static_cast` 写入后重建，sqrtf 错误消失，编译推进到 DataCopyPad 参数检查阶段，最终全工程编译成功（来源平台：950；证据等级：tool_observed）。
- 同会话对照：同 kernel 一直使用 `<cmath>` 的 `sqrt(variance + epsilon_)` 编译正常（同 trace 的 std::max 修复记录佐证），印证"sqrt 可用、sqrtf 不可用"的符号级差异（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- float→double→float 的往返在极端值域（次正规边界）的舍入差异未逐位验证——本算子 golden 同为 double 路径，未见偏差。
- 本卡限定设备侧（`__aicore__`）标量代码；host 侧 sqrtf 正常可用。
- 未验证，推断可迁移：其它 AscendC 目标 SoC（910b 等）同一设备侧编译器族。

## 增量（2026-09-17，先前增量 · TBE 链 GroupNorm 形态：note 直指 CCE 内建 float sqrt）

独立复现（BenchNoHarnessGroupNorm 算子全载 kernel，Ascend950PR/arch35 + CANN 9.1.0，`bash build.sh --soc=ascend950 -j8` 的 TBE op 编译器 INFERCHANNEL 阶段）：按公式 `rstd = 1/sqrt(var + ε)` 计算标准差倒数时按 host 侧习惯写 `sqrtf`，kernel 编译报错原文：

```text
op_kernel/arch35/bench_no_harness_group_norm.h:208:25:
  error: use of undeclared identifier 'sqrtf'; did you mean 'sqrt'?
    208 |     float rstd = 1.0f / sqrtf(var + epsilon_);
<bisheng 安装目录>/lib/clang/15.0.5/include/__clang_cce_aicore_functions.h:3519:33:
  note: 'sqrt' declared here
CCE_INTRINSIC CALLEE_ATTR float sqrt(float in) { return __builtin_cce_sqrtf(in); }
```

与本卡先前增量 形态同错误文本，但本开发会话 编译器 note 指向的声明是 **CCE 内建 float 版 `sqrt`**（而非经 host 头文件链可自动提升的 double 版）——直接改小写 `sqrt` 即可，无须 double 往返收窄。该开发会话 还实走了 R3 已记录的弯路：先改成 `AscendC::Sqrt(var + epsilon_)`（张量 Level2 API 误用于标量 float，错误方向），再依编译器 note 二次修正。

修复（before/after verbatim，kernel ProcessOneGroup 内）：

```cpp
// before（libc 名字）
    // Step 5: 标准差倒数 rstd = 1/sqrt(var + eps)
    float rstd = 1.0f / sqrtf(var + epsilon_);
```

```cpp
// 误诊中间态（张量 API 误用于标量，错误方向）
    float rstd = 1.0f / AscendC::Sqrt(var + epsilon_);
```

```cpp
// after（CCE 内建 sqrt，小写）
    // Step 5: 标准差倒数 rstd = 1/sqrt(var + eps)
    float rstd = 1.0f / sqrt(var + epsilon_);
```

判别补充：报错 note 指向的 `sqrt` 声明取决于该 TU 的包含链——本形态直指 `__clang_cce_aicore_functions.h` 的 CCE 内建 float 版（无须 static_cast）；先前增量 形态指向可自动提升的 double 版（结果需 `static_cast<float>` 收回）。两者同名同效，以当次编译器 note 为准，不要去翻 math.h、也不要写 `f` 后缀单精度变体。

### 复现增量验证信号

- 修复前：kernel 编译报 `error: use of undeclared identifier 'sqrtf'; did you mean 'sqrt'?`，TBE 编译框架层同时报 `Exception: An error occurred during compile phases of CompileStage.INFERCHANNEL`、`Op[BenchNoHarnessGroupNorm] of index[0] compile failed`、`.json not generated`（来源平台：950；证据等级：tool_observed）。
- 修复后：同一命令重跑 `[INFO] Build completed successfully!`，3 个 kernel `.o`（BenchNoHarnessGroupNorm_*.o）与 `custom_opp_ubuntu_x86_64.run` 生成（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- 本形态与本卡先前增量 的 note 定位差异（CCE 内建 float 版 vs host 链 double 版）来自包含链差异，未做控制变量验证；处置一致——用编译器自荐的 `sqrt`，并按其形参类型决定是否显式收窄。
- 仍只实测 `sqrt`/`sqrtf`；exp/log/pow 等其他数学函数按"先查内建头"判别方法推广，未逐一验证。
- 误诊方向"以为 AscendC 命名空间下有标量封装"与 R3 一致：标量数学在 CCE 内建层（小写、无命名空间），`AscendC::Sqrt` 仅重载 LocalTensor 张量签名。
