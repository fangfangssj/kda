---
type: "Runbook"
title: "Reg 同 wrapper 回读 UB 引发尺寸相关漂移"
description: "同一 asc_vf_call 内跨 pass 写回并立即回读 UB 会形成无法显式隔离的流水 RAW 风险，拆分 wrapper 并在其间加 PIPE_V barrier 可恢复确定性。"
tags: ["task.debugging", "topic.compute.normalization", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-09-14T08:44:28Z"
platforms: ["950"]
---

# Reg 同 wrapper 回读 UB 引发尺寸相关漂移

## 现象

在 Ascend950 上为连续归约轴 Softmax 编写 `softmax.asc` 的 Reg 路径时，工程已经能够构建和加载，但 NPU 精度结果随归约长度剧烈变化。短行输出会出现负概率、NaN，或整行归一化和显著偏离 1；较长行又可能完全正确。继续扫描相邻长度时，结果并不随长度单调改善：一个长度通过，紧邻长度仍可能失败。这种“数据量大反而偶然正确”的信号出现在 `ar_full_vf` 把三次 pass 放在同一个 `asc_vf_call`、第二次 pass 把指数值写入 UB、第三次 pass 再从同一 UB 地址读取的代码状态下。

## 适用场景

- 工程形态：AscendC kernel 直调，连续归约轴的多 pass Reg 实现。
- API 路径：`__simd_vf__`、`asc_vf_call`、`Reg::StoreAlign`、`Reg::LoadAlign` 与 `PipeBarrier<PIPE_V>`。
- 已验证平台与版本：Ascend950PR_957c（dav-3510），CANN 9.0.1；与 frontmatter 的 `platforms: [950]` 一致。
- 适用粒度：`same_operator`。证据只覆盖 Softmax 的 AR 路径，其他多 pass reduction 尚未实测。
- 路线优先级：`safety_fallback`。这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。最终组合实现的平均速度约为评测参考实现的 0.29 倍，且没有隔离测出本修复自身的性能增量。
- 不适用或未验证：单次 pass 不跨 UB 写后读的 kernel、能够在同一 wrapper 内提供等价显式依赖保证的未来 API、A2/A3 与其他 CANN 版本均未验证。
- 增量适用（独立复现 R2，CANN 9.2.0）：跨算子族证据——BenchHarnessEngramGateFusion 融合算子（aclnn 算子仓 regbase 范式）的两阶段 Σx² 归约（单 VF 内 partial store 后紧接 load 合并）复现同一根因；屏障变体扩展为 `PipeBarrier<PIPE_ALL>`（R1 用 PIPE_V）；失败形态扩展——小 D（rep=1）用例同样翻车，单 VF 内 store→load 不因块小而安全。

## 根因

同一个 `__simd_vf__` 中第二个 pass 用 `StoreAlign` 写 `expAddr`、第三个 pass 随即用 `LoadAlign` 回读，而 `__aicore__` 层无法在这个单一 `asc_vf_call` 内插入 `PipeBarrier<PIPE_V>`，导致 vector store 尚未排空时 load 可能读到旧 UB 数据。 机制链是：`pass 2 StoreAlign(expAddr)` 发出写入 → wrapper 尚未返回，host 侧无 barrier 插入点 → `pass 3 LoadAlign(expAddr)` 可在写入完成前取数 → 除法使用过期指数值。较长循环给写流水更多自然排空时间，所以“大长度通过”不能证明实现正确；相邻长度成败非单调正是时序竞争而不是公式误差的信号。

增量（R2）：该“无内存序保证”的机制解释为调试者归因；客观证据是成对对照信号——“无屏障单 VF 版 2/23”与“拆分 + PipeBarrier 版 12/23”两次构建仅差调用拆分与屏障。

## 解决配方

把三次 pass 拆成三个完整的 `__simd_vf__` 函数和三个独立的 `asc_vf_call` wrapper。max/sum 只通过 VL 大小的 UB scalar slot 跨 pass 传递，调用之间由 `PipeBarrier<PIPE_V>` 建立可见性；第三个 pass 重新计算 `exp(x-max)`，不再回读第二个 pass 写出的整行指数 UB。

### before：单 wrapper 内写后读

- revision: trace-before-ar-split
- path: `softmax.asc`
- symbol: `ar_full_vf`
- line_range: `L132-L190`（修复前替换区域，含完整函数与原注释）

```cpp
// AR full softmax for one contiguous row of `count` fp32 elements.
// Numerically stable: y = exp(x-max) / sum(exp(x-max)). The per-row reduced
// max/sum scalars stay in registers and are broadcast to full vectors with the
// broadcast overload of Reg::Duplicate (template defaults HighLowPart::LOWEST,
// MaskMergeMode::ZEROING) — no UB scalar round-trip (GetValue is rejected by
// the AIV backend as a scalar instruction in the vector loop). Per-chunk
// Reduce + scalar combine makes the tail-chunk mask correct: the Reduce mask
// is a participation predicate (vcmax/vcadd exclude masked-off elements as the
// identity, not zero them), so all-negative rows with a partial tail chunk are
// handled correctly.
__simd_vf__ inline void ar_full_vf(__ubuf__ float *yAddr, __ubuf__ float *xAddr,
                                  __ubuf__ float *expAddr, uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReg, tmpSca, maxSca, maxBcast, eReg, sumSca, sumBcast;
    uint32_t one = 1;
    uint32_t full = vectorLength;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(one);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(full);
    const float negInf = -std::numeric_limits<float>::infinity();

    // pass 1: row max (per-chunk reduce + scalar combine)
    AscendC::Reg::Duplicate<float>(maxSca, negInf, scalarMask);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xAddr + offset);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float, float>(tmpSca, xReg, mask);
        AscendC::Reg::Max(maxSca, maxSca, tmpSca, scalarMask);
    }
    // broadcast the lane-0 scalar to a full vector for elementwise subtraction
    AscendC::Reg::Duplicate<float>(maxBcast, maxSca, fullMask);

    // pass 2: exp(x - max) + sum (store exp to expAddr for the div pass)
    AscendC::Reg::Duplicate<float>(sumSca, 0.0f, scalarMask);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xAddr + offset);
        AscendC::Reg::Sub(xReg, xReg, maxBcast, mask);
        AscendC::Reg::Exp(xReg, xReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(expAddr + offset, xReg, mask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float, float>(tmpSca, xReg, mask);
        AscendC::Reg::Add(sumSca, sumSca, tmpSca, scalarMask);
    }
    AscendC::Reg::Duplicate<float>(sumBcast, sumSca, fullMask);

    // pass 3: normalize (exp / sum) — broadcast sum vector, elementwise div
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(eReg, expAddr + offset);
        AscendC::Reg::Div(eReg, eReg, sumBcast, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, eReg, mask);
    }
}
```

- revision: trace-before-ar-split
- path: `softmax.asc`
- symbol: `ArFull`
- line_range: `L321-L332`（该编辑事件 old side 的完整函数）

```cpp
__aicore__ inline void ArFull(const AscendC::LocalTensor<float> &y,
                              const AscendC::LocalTensor<float> &x,
                              const AscendC::LocalTensor<float> &exp, int64_t count)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    __ubuf__ float *yA = (__ubuf__ float *)y.GetPhyAddr();
    __ubuf__ float *xA = (__ubuf__ float *)x.GetPhyAddr();
    __ubuf__ float *eA = (__ubuf__ float *)exp.GetPhyAddr();
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t chunks = (cnt + vectorLength - 1) / vectorLength;
    asc_vf_call<ar_full_vf>(yA, xA, eA, cnt, chunks);
}
```

- revision: trace-before-ar-split
- path: `softmax.asc`
- symbol: `KernelSoftmaxAR<T>::Compute`
- line_range: `L453-L473`（该编辑事件 old side 的完整函数）

```cpp
    __aicore__ inline void Compute()
    {
        auto xLocal = inQueue_.template DeQue<T>();
        auto expLocal = expBuf_.template Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            auto yLocal = outQueue_.template AllocTensor<T>();
            ArFull(yLocal, xLocal, expLocal, r_);
            outQueue_.EnQue(yLocal);
        } else {
            auto xF = xFp32_.template Get<float>();
            auto yF = yFp32_.template Get<float>();
            CastToFp32<T>(xF, xLocal, r_);
            AscendC::PipeBarrier<PIPE_V>();
            ArFull(yF, xF, expLocal, r_);
            AscendC::PipeBarrier<PIPE_V>();
            auto yLocal = outQueue_.template AllocTensor<T>();
            CastFromFp32<T>(yLocal, yF, r_);
            outQueue_.EnQue(yLocal);
        }
        inQueue_.FreeTensor(xLocal);
    }
```

### after：每个 pass 一个 wrapper，并在 host 层隔离

- revision: trace-after-ar-split
- path: `softmax.asc`
- symbols: `ar_max_vf`, `ar_sum_vf`, `ar_out_vf`
- line_range: `L149-L217`

```cpp
// AR pass 1: row max -> maxSlot[0].
__simd_vf__ inline void ar_max_vf(__ubuf__ float *maxSlot, __ubuf__ float *xAddr,
                                  uint32_t count, uint32_t chunkCount)
{
    AscendC::Reg::RegTensor<float> xReg, tmpSca, maxSca;
    uint32_t one = 1;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(one);
    const float negInf = -std::numeric_limits<float>::infinity();
    AscendC::Reg::Duplicate<float>(maxSca, negInf, scalarMask);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * (AscendC::VECTOR_REG_WIDTH / sizeof(float));
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xAddr + offset);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float, float>(tmpSca, xReg, mask);
        AscendC::Reg::Max(maxSca, maxSca, tmpSca, scalarMask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(maxSlot, maxSca, scalarMask);
}

// AR pass 2: sum of exp(x - max) -> sumSlot[0]. Reloads+broadcasts max.
__simd_vf__ inline void ar_sum_vf(__ubuf__ float *sumSlot, __ubuf__ float *xAddr,
                                  __ubuf__ float *maxSlot, uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReg, tmpSca, maxSca, maxBcast, sumSca;
    uint32_t one = 1;
    uint32_t full = vectorLength;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(one);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(full);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(maxSca, maxSlot);
    AscendC::Reg::Duplicate<float>(maxBcast, maxSca, fullMask);
    AscendC::Reg::Duplicate<float>(sumSca, 0.0f, scalarMask);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xAddr + offset);
        AscendC::Reg::Sub(xReg, xReg, maxBcast, mask);
        AscendC::Reg::Exp(xReg, xReg, mask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float, float>(tmpSca, xReg, mask);
        AscendC::Reg::Add(sumSca, sumSca, tmpSca, scalarMask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(sumSlot, sumSca, scalarMask);
}

// AR pass 3: y = exp(x - max) / sum. Reloads+broadcasts max and sum, recomputes exp.
__simd_vf__ inline void ar_out_vf(__ubuf__ float *yAddr, __ubuf__ float *xAddr,
                                  __ubuf__ float *maxSlot, __ubuf__ float *sumSlot,
                                  uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReg, maxSca, maxBcast, sumSca, sumBcast;
    uint32_t full = vectorLength;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(full);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(maxSca, maxSlot);
    AscendC::Reg::Duplicate<float>(maxBcast, maxSca, fullMask);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sumSca, sumSlot);
    AscendC::Reg::Duplicate<float>(sumBcast, sumSca, fullMask);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xReg, xAddr + offset);
        AscendC::Reg::Sub(xReg, xReg, maxBcast, mask);
        AscendC::Reg::Exp(xReg, xReg, mask);
        AscendC::Reg::Div(xReg, xReg, sumBcast, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, xReg, mask);
    }
}
```

- revision: trace-after-ar-split
- path: `softmax.asc`
- symbols: `ArMax`, `ArSum`, `ArOut`
- line_range: `L321-L358`

```cpp
__aicore__ inline void ArMax(const AscendC::LocalTensor<float> &maxSlot,
                              const AscendC::LocalTensor<float> &x, int64_t count)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    __ubuf__ float *slot = (__ubuf__ float *)maxSlot.GetPhyAddr();
    __ubuf__ float *xA = (__ubuf__ float *)x.GetPhyAddr();
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t chunks = (cnt + vectorLength - 1) / vectorLength;
    asc_vf_call<ar_max_vf>(slot, xA, cnt, chunks);
}

__aicore__ inline void ArSum(const AscendC::LocalTensor<float> &sumSlot,
                             const AscendC::LocalTensor<float> &x,
                             const AscendC::LocalTensor<float> &maxSlot, int64_t count)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    __ubuf__ float *slot = (__ubuf__ float *)sumSlot.GetPhyAddr();
    __ubuf__ float *xA = (__ubuf__ float *)x.GetPhyAddr();
    __ubuf__ float *m = (__ubuf__ float *)maxSlot.GetPhyAddr();
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t chunks = (cnt + vectorLength - 1) / vectorLength;
    asc_vf_call<ar_sum_vf>(slot, xA, m, cnt, chunks);
}

__aicore__ inline void ArOut(const AscendC::LocalTensor<float> &y,
                             const AscendC::LocalTensor<float> &x,
                             const AscendC::LocalTensor<float> &maxSlot,
                             const AscendC::LocalTensor<float> &sumSlot, int64_t count)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    __ubuf__ float *yA = (__ubuf__ float *)y.GetPhyAddr();
    __ubuf__ float *xA = (__ubuf__ float *)x.GetPhyAddr();
    __ubuf__ float *m = (__ubuf__ float *)maxSlot.GetPhyAddr();
    __ubuf__ float *s = (__ubuf__ float *)sumSlot.GetPhyAddr();
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t chunks = (cnt + vectorLength - 1) / vectorLength;
    asc_vf_call<ar_out_vf>(yA, xA, m, s, cnt, chunks);
}
```

- revision: trace-after-ar-split
- path: `softmax.asc`
- symbol: `KernelSoftmaxAR<T>::Compute`
- line_range: `L453-L482`

```cpp
    __aicore__ inline void Compute()
    {
        auto xLocal = inQueue_.template DeQue<T>();
        auto maxSlot = maxSlot_.template Get<float>();
        auto sumSlot = sumSlot_.template Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            auto yLocal = outQueue_.template AllocTensor<T>();
            ArMax(maxSlot, xLocal, r_);
            AscendC::PipeBarrier<PIPE_V>();
            ArSum(sumSlot, xLocal, maxSlot, r_);
            AscendC::PipeBarrier<PIPE_V>();
            ArOut(yLocal, xLocal, maxSlot, sumSlot, r_);
            outQueue_.EnQue(yLocal);
        } else {
            auto xF = xFp32_.template Get<float>();
            auto yF = yFp32_.template Get<float>();
            CastToFp32<T>(xF, xLocal, r_);
            AscendC::PipeBarrier<PIPE_V>();
            ArMax(maxSlot, xF, r_);
            AscendC::PipeBarrier<PIPE_V>();
            ArSum(sumSlot, xF, maxSlot, r_);
            AscendC::PipeBarrier<PIPE_V>();
            ArOut(yF, xF, maxSlot, sumSlot, r_);
            AscendC::PipeBarrier<PIPE_V>();
            auto yLocal = outQueue_.template AllocTensor<T>();
            CastFromFp32<T>(yLocal, yF, r_);
            outQueue_.EnQue(yLocal);
        }
        inQueue_.FreeTensor(xLocal);
    }
```

buffer 初始化也必须同步改变；slot 按完整 vector load 的访问宽度分配，不能只给 4 字节 lane 0。

- revision: trace-before-ar-split
- path: `softmax.asc`
- symbol: `KernelSoftmaxAR<T>::Init`
- line_range: `L407-L427`（由相邻版本与同一编辑事件的 old side 函数级重建）

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t a1, int64_t r, int64_t numBlocks)
    {
        constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        int64_t rowsPerCore = (a1 + numBlocks - 1) / numBlocks;
        int64_t bid = AscendC::GetBlockIdx();
        startRow_ = MinI64(a1, bid * rowsPerCore);
        endRow_ = MinI64(a1, startRow_ + rowsPerCore);
        r_ = r;
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        yGm_.SetGlobalBuffer((__gm__ T *)y);
        // Buffers are ceil-aligned to vectorLength so a tail-chunk LoadAlign
        // never overruns the allocation.
        int64_t ceilR = ((r + vectorLength - 1) / vectorLength) * vectorLength;
        pipe_.InitBuffer(inQueue_, 1, ceilR * sizeof(T));
        pipe_.InitBuffer(outQueue_, 1, ceilR * sizeof(T));
        pipe_.InitBuffer(expBuf_, ceilR * sizeof(float));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFp32_, ceilR * sizeof(float));
            pipe_.InitBuffer(yFp32_, ceilR * sizeof(float));
        }
    }
```

- revision: trace-after-ar-split
- path: `softmax.asc`
- symbol: `KernelSoftmaxAR<T>::Init`
- line_range: `L407-L431`

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t a1, int64_t r, int64_t numBlocks)
    {
        constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        int64_t rowsPerCore = (a1 + numBlocks - 1) / numBlocks;
        int64_t bid = AscendC::GetBlockIdx();
        startRow_ = MinI64(a1, bid * rowsPerCore);
        endRow_ = MinI64(a1, startRow_ + rowsPerCore);
        r_ = r;
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        yGm_.SetGlobalBuffer((__gm__ T *)y);
        // Buffers are ceil-aligned to vectorLength so a tail-chunk LoadAlign
        // never overruns the allocation.
        int64_t ceilR = ((r + vectorLength - 1) / vectorLength) * vectorLength;
        pipe_.InitBuffer(inQueue_, 1, ceilR * sizeof(T));
        pipe_.InitBuffer(outQueue_, 1, ceilR * sizeof(T));
        // One-vector-lane scalar slots for the per-row reduced max/sum. VL-sized
        // so LoadAlign never overruns; only lane 0 holds the scalar, broadcast
        // back via Duplicate(LOWEST) each pass.
        pipe_.InitBuffer(maxSlot_, vectorLength * sizeof(float));
        pipe_.InitBuffer(sumSlot_, vectorLength * sizeof(float));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFp32_, ceilR * sizeof(float));
            pipe_.InitBuffer(yFp32_, ceilR * sizeof(float));
        }
    }
```

类成员也从整行中间 buffer 换成两个 scalar slot；以下声明为编辑事件的 verbatim before/after，虽然不是函数，仍是上述完整函数能编译所需的配套定义。

- revision: trace-before-ar-split
- path: `softmax.asc`
- symbol: `KernelSoftmaxAR<T>` members
- line_range: `L496-L499`（该编辑事件 old side 的完整成员声明）

```cpp
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32_, yFp32_;
    int64_t startRow_ = 0, endRow_ = 0, r_ = 0;
};
```

- revision: trace-after-ar-split
- path: `softmax.asc`
- symbol: `KernelSoftmaxAR<T>` members
- line_range: `L496-L499`

```cpp
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxSlot_, sumSlot_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32_, yFp32_;
    int64_t startRow_ = 0, endRow_ = 0, r_ = 0;
};
```

### 增量：两阶段归约的负结果与恢复形态（CANN 9.2.0，独立复现 R2）

同一根因在两阶段 Σx² 归约上的成对对照 artifact。banned（单 VF 内 store→load 无屏障的写法，实测 2/23，verbatim——勿用）：

```cpp
// Two-phase reduce: per-rep partial sums stored to work[], then merged.
__simd_vf__ inline void ReduceSumSqVF(__ubuf__ float* dst, __ubuf__ float* src, __ubuf__ float* work, uint16_t n, uint16_t rep) {
    constexpr uint32_t VL=64; AscendC::Reg::RegTensor<float> rs,sq,ps,ms; AscendC::Reg::MaskReg m,m1;
    m1=AscendC::Reg::CreateMask<float,AscendC::Reg::MaskPattern::VL1>();
    uint32_t r=n;
    for(uint16_t i=0;i<rep;++i){uint16_t o=i*VL; m=AscendC::Reg::UpdateMask<float>(r);
    AscendC::Reg::LoadAlign(rs,src+o); AscendC::Reg::Mul(sq,rs,rs,m);
    AscendC::Reg::Reduce<AscendC::ReduceType::SUM,float>(ps,sq,m);
    AscendC::Reg::StoreAlign<float,AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(work+i,ps,m1);
    r=(r>VL)?r-VL:0;}
    // Phase 2: load partial sums and final-reduce（与阶段1同函数 —— 读不到刚写入的 work[]）
    uint32_t repRem=rep; AscendC::Reg::MaskReg fm=AscendC::Reg::UpdateMask<float>(repRem);
    AscendC::Reg::LoadAlign(ms,work);
    AscendC::Reg::Reduce<AscendC::ReduceType::SUM,float>(ps,ms,fm);
    AscendC::Reg::StoreAlign<float,AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dst,ps,m1);
}
```

恢复形态（verbatim——拆成两次独立 `asc_vf_call`，中间 `PipeBarrier<PIPE_ALL>`）：

```cpp
// Per-rep partial reduce: stores one scalar per rep to work[0..rep-1].
// Follow with PipeBarrier then asc_vf_call<MergeReduceVF> to get final sum.
__simd_vf__ inline void ReduceSumSqPartialsVF(__ubuf__ float* work, __ubuf__ float* src, uint16_t n, uint16_t rep) {
    constexpr uint32_t VL=64; AscendC::Reg::RegTensor<float> rs,sq,ps; AscendC::Reg::MaskReg m,m1;
    m1=AscendC::Reg::CreateMask<float,AscendC::Reg::MaskPattern::VL1>();
    uint32_t r=n;
    for(uint16_t i=0;i<rep;++i){uint16_t o=i*VL; m=AscendC::Reg::UpdateMask<float>(r);
    AscendC::Reg::LoadAlign(rs,src+o); AscendC::Reg::Mul(sq,rs,rs,m);
    AscendC::Reg::Reduce<AscendC::ReduceType::SUM,float>(ps,sq,m);
    AscendC::Reg::StoreAlign<float,AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(work+i,ps,m1);
    r=(r>VL)?r-VL:0;}
}
// Merge partial sums: load work[0..rep-1], reduce to single scalar at dst[0].
__simd_vf__ inline void MergeReduceVF(__ubuf__ float* dst, __ubuf__ float* work, uint16_t rep) {
    AscendC::Reg::RegTensor<float> ms,ps; AscendC::Reg::MaskReg fm,m1;
    m1=AscendC::Reg::CreateMask<float,AscendC::Reg::MaskPattern::VL1>();
    uint32_t repRem=rep; fm=AscendC::Reg::UpdateMask<float>(repRem);
    AscendC::Reg::LoadAlign(ms,work);
    AscendC::Reg::Reduce<AscendC::ReduceType::SUM,float>(ps,ms,fm);
    AscendC::Reg::StoreAlign<float,AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dst,ps,m1);
}
// 调用侧（大 D / dRep>1 路径）：
//   asc_vf_call<ReduceSumSqPartialsVF>(work, src, n, rep);
//   AscendC::PipeBarrier<PIPE_ALL>();
//   asc_vf_call<MergeReduceVF>(dst, work, rep);
```

路线档位：单 VF 内 store→load 中转为 `banned`；拆分 + 中间屏障为 `safety_fallback`——功能正确，相比纯寄存器内累加多一次 UB 往返与一次全流水屏障，本算子内未测得可感知性能差异，性能敏感场景需实测。`MergeReduceVF` 无 for 循环、单次掩码归并，传 rep=13 正常。跨块循环上提到 `__aicore__` wrapper 并在调用间加 Vector barrier 的完整形态，见 [AR 跨块循环放在 __simd_vf__ 只执行首块卡](simd_vf_chunk_loop_must_be_driven_by_aicore_wrapper.md)——区分条件：该卡是循环上移，本节是中间结果中转拆分；固定地址累加器的逐块 StoreAlign→LoadAlign 同样需逐 VF barrier，见 [Reg 固定地址跨块累加需逐 VF PipeBarrier 卡](fixed_address_vf_accumulator_needs_per_call_barrier.md)——区分条件：跨块同址读改写 vs 单调用内两阶段中转。

## 系统信息

- 硬件：Ascend950PR_957c，dav-3510。
- 软件：CANN 9.0.1，Bisheng/AscendC 编译链。
- 代码文件：`softmax.asc`。
- 关键符号：`ar_full_vf`、`ar_max_vf`、`ar_sum_vf`、`ar_out_vf`、`ArFull`、`ArMax`、`ArSum`、`ArOut`、`KernelSoftmaxAR<T>::Compute`。

## 设计决策

选择“三 wrapper + 两个 scalar slot + 输出 pass 重算 exp”，是为了在 `__aicore__` 层获得明确的 barrier 插入点，并彻底移除整行 `expAddr` 的写后读依赖。保留一个 `ar_full_vf` 而只希望大循环自然排空是被禁止的竞态方案；尝试从 Reg 标量直接 `GetValue` 绕过 UB 也被 AIV backend 以 vector loop 中不支持 scalar instruction 拒绝。当前方案属于 `safety_fallback`，优先保证结果确定性，不声称性能最优。

## 性能影响

输出 pass 会重新执行一次 `Exp`，并增加 wrapper 边界与 barrier。轨迹没有提供仅切换本修复前后的成对性能结果；最终组合实现的平均速度约为评测参考实现的 0.29 倍，所以该方案只能作为正确性兜底，不能据此推导通用性能收益。

## 预防

- 审查多 pass Reg kernel 时，画出每个 `asc_vf_call` 的 UB 读写集合；同一地址跨 pass 出现 `StoreAlign → LoadAlign` 时，必须确认存在 host 层可执行的同步边界。
- 不把“大尺寸多次通过”当竞态消失的证据；至少覆盖短行、vector 边界两侧和相邻非单调长度。
- 跨 wrapper 只传小标量时，使用 VL 大小的 UB slot，避免 `LoadAlign` 越过分配范围。
- 如果必须保留整行中间结果，另行验证能建立可见性的 API 路线和相对性能，不能沿用本卡的自然排空假设。
- 增量：拆分 + 屏障只恢复基线，不修其他根因——R2 会话中该修复相对基线无增益（12/23 → 12/23），剩余失败由 rep 截断另行闭环，见 [__simd_vf__ 多 rep 截断致大 D 尾部为零卡](simd_vf_multi_rep_truncation_large_d.md)——区分条件：尾部元素保持零/旧值的大面积失配属 rep 截断，不属本卡。

## 验证信号

来源说明：本卡由单条 AscendC 开发轨迹中的文件编辑、构建输出、NPU pytest 与 cann-bench 工具结果重建；内部事件锚点只保存在同目录 `manifest.json`，正文不暴露会话标识。下列前四条信号为 R1（Softmax AR 路径，CANN 9.0.1）；R2 为 BenchHarnessEngramGateFusion 融合算子 kernel（CANN 9.2.0，aclnn 算子仓形态、不同算子与不同会话）的独立复现：两阶段归约改造的成对对比实验。

- 修复前，目标长度扫描可观察到负输出、NaN、归一化和偏离 1，以及相邻长度成败非单调。（证据等级: tool_observed；来源平台: Ascend950）
- 拆分 wrapper 并插入 `PIPE_V` barrier 后，易失败长度、特殊值、float16 与 bfloat16 的定向 NPU 扫描均通过，输出 `FAILS: 0`。（证据等级: tool_observed；来源平台: Ascend950）
- 修复后完整 NPU pytest 首次为 53/53，同一轨迹随后连续两次 53/53，清理构建后仍为 53/53。（证据等级: tool_observed；来源平台: Ascend950）
- 最终新评测进程覆盖的 20 个用例全部精度通过；最终组合实现平均速度约为参考实现的 0.29 倍。（证据等级: tool_observed；来源平台: Ascend950）

- 独立复现 R2（BenchHarnessEngramGateFusion，CANN 9.2.0，与 R1 不同算子、不同会话）：把两阶段归约写进单个 `__simd_vf__`（阶段 1 逐 rep StoreAlign 部分和到 work[]、阶段 2 同函数内紧接 LoadAlign 读回合并）后，同一 ACLNN ST 测试集从基线 12/23 跌到 2/23——F1/F2/F5/F6/F8 从通过变为全错（F1 64/64、F2 8/8、F6 16/16 元素错），E4/E5/E6/E7/E8 全部 64/64 错，仅最小调用路径与全零期望用例通过；小 D（rep=1）用例也一并失败（来源平台：950；证据等级：tool_observed）。
- R2 恢复形态：拆成 `ReduceSumSqPartialsVF` + `PipeBarrier<PIPE_ALL>` + `MergeReduceVF` 两次调用后恢复 12/23，F1/F2/F5/F6/F8/E5/E6 均 PASSED——与负结果版仅差调用拆分与屏障（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 证据只覆盖 Ascend950 与 CANN 9.0.1；A2/A3 和其他版本未验证，推断可迁移但不得直接采用。
- 该卡不能证明所有同 wrapper 的 store/load 都必然出错；它只证明本数据流缺少显式 barrier 时结果不确定。
- 若平台或 API 能在单 wrapper 内提供经验证的依赖控制，重算 exp 的 fallback 可能不再必要。
- 最终性能是两个修复和整条 kernel 的组合结果，不能归因到本修复单点。
- 增量（R2）：屏障变体扩展为 `PipeBarrier<PIPE_ALL>`（R1 为 PIPE_V），两者在本场景均能建立可见性；单 VF 内 store→load 对小 D（rep=1）同样不安全。
- 增量（R2）：拆分 + 屏障只恢复基线、不修 rep 截断（大 D 单调用多 rep 截断）等其他根因——12/23 相对基线无增益，仅恢复原状；剩余失败的归因见 [__simd_vf__ 多 rep 截断致大 D 尾部为零卡](simd_vf_multi_rep_truncation_large_d.md)。
- 增量（R2）：`LocalMemBar` 是否可替代“拆分 + PipeBarrier”未验证，更细粒度同步的开销差异未测；采用前先核对 `LocalMemBar` 只能在 `__simd_vf__` 调用上下文内使用、从 `__aicore__` 直接调用会 507035 的约束，见 [arch35 __aicore__ 调 LocalMemBar 报 507035 卡](../compilation/arch35_aicore_localmembar_507035.md)。

## 坏实践

- 在一个 `asc_vf_call` 中写完整 `expAddr` 后立即回读，并把大长度偶然通过解释成实现正确。
- 仅增加循环工作量或重复跑通过用例，试图让流水线“自然稳定”。
- 把 VL load 对应的 scalar slot 只分配为单个 `float`。
- 通过放宽精度阈值掩盖负概率、NaN 或归一化和错误；本轨迹没有采用这种捷径。
- 增量：在一个 `__simd_vf__` 函数里“先 StoreAlign 中间结果到 UB、再 LoadAlign 读回”做数据中转——无内存序保证，读旧值且无任何告警（R2 实测 12/23→2/23，小 D 也翻车）。
- 增量：见到“重写后更糟”的全面倒退时不要急于继续叠加改动；先用最小对照（加/拆屏障）把变量隔离。

## 误诊

早期曾关注 tail mask、低精度 cast 与 UB 残留，但这些假设不能解释 fp32 同样失败以及长度非单调；真正闭环来自改变 wrapper 边界和数据依赖。`GetValue` 路线只产生编译器拒绝，不是运行时精度修复。

增量（R2）：R2 的两阶段重写本身是一次误诊产物——把 statK≈golden/2 归因于“`Reg::Reduce` 多 rep 寄存器累加错误”。实际根因在上游 cast 截断（见 [__simd_vf__ 多 rep 截断致大 D 尾部为零卡](simd_vf_multi_rep_truncation_large_d.md)）——归约算法本身没有错；本卡闭合的是重写过程中暴露的“单 VF 内 store→load 无内存序”这一独立问题。
