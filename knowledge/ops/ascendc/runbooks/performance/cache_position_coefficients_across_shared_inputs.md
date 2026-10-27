---
type: "Runbook"
title: "按序列位缓存 cos/sin 并让 q/k 共享搬运"
description: "逐行重复搬入和展开相同 cos/sin 会拖慢小行宽 Vector kernel；按序列位缓存并在 q/k 间共享可减少 GM→UB 事务。"
tags: ["task.optimization", "topic.data_movement.data_copy", "topic.memory", "topic.implementation.pipeline", "topic.performance.optimization.optimization_overview"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
aliases: ["q_k_share"]
---

# 按序列位缓存 cos/sin 并让 q/k 共享搬运

## 现象

正确性冻结后，profiling 显示目标 Vector kernel 的主要时间仍耗在逐行重复搬入和展开 cos/sin；源码先完整处理 q，再完整处理 k，相同序列位置的只读系数被多次搬运。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的同负载性能评测。
- API 路径：`EnsureCosSinLoaded`、`cachedS_`、`TQue`、`TBuf` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

原路径每处理一行输入就执行两次 cos/sin GM→UB 搬入和一次展开；q、k 分成两个全量循环，使相同行索引的系数至少重复一次。layout=0 下相邻 head 的 `s` 还相同，重复次数进一步增加。固定行宽很小时，DMA 发起、队列交接和展开开销无法被足够计算掩盖。把 `cosFullF/sinFullF` 留在 TBuf 中，以 `cachedS_` 只在 `s` 变化时重载，并在同一行循环中先后计算 q/k，可同时消除跨 head 与 q/k 的冗余系数事务。

## 解决配方

### before

- path: `operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`
- symbol: `KernelApplyRotaryPosEmb<T>`
- claim: q/k 分两个循环，每个 ComputeRow 都重新搬入、Cast 和展开 cos/sin

```cpp
template <typename T>
class KernelApplyRotaryPosEmb {
public:
    __aicore__ inline KernelApplyRotaryPosEmb() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR cos, GM_ADDR sin,
                               GM_ADDR queryOut, GM_ADDR keyOut,
                               uint32_t totalRows, uint32_t headDim,
                               uint32_t rowsPerBlock, uint32_t layout, uint32_t rotaryMode,
                               uint32_t seqLen, uint32_t numHeads) {
        D_ = headDim;
        halfD_ = headDim / 2;
        layout_ = layout;
        mode_ = rotaryMode;          // 0 = half, 1 = interleaved
        seqLen_ = seqLen;
        numHeads_ = numHeads;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t startRow = blockIdx * rowsPerBlock;
        if (startRow >= totalRows) {
            rowCount_ = 0;
            startRow_ = 0;
        } else {
            startRow_ = startRow;
            uint32_t remaining = totalRows - startRow;
            rowCount_ = (rowsPerBlock < remaining) ? rowsPerBlock : remaining;
        }

        // 32B-aligned allocation counts (padding tail never read).
        constexpr uint32_t alignT = 32 / sizeof(T);      // 16 fp16/bf16, 8 fp32
        constexpr uint32_t alignF = 32 / sizeof(float);   // 8
        DAlignT_ = (headDim + alignT - 1) / alignT * alignT;
        DAlignF_ = (headDim + alignF - 1) / alignF * alignF;
        halfDAlignT_ = (halfD_ + alignT - 1) / alignT * alignT;
        halfDAlignF_ = (halfD_ + alignF - 1) / alignF * alignF;

        // GM buffers (per-core row range for q/k; full cos/sin).
        qGm.SetGlobalBuffer((__gm__ T*)query + (uint64_t)startRow_ * headDim,
                            (uint64_t)rowCount_ * headDim);
        kGm.SetGlobalBuffer((__gm__ T*)key + (uint64_t)startRow_ * headDim,
                            (uint64_t)rowCount_ * headDim);
        qOutGm.SetGlobalBuffer((__gm__ T*)queryOut + (uint64_t)startRow_ * headDim,
                               (uint64_t)rowCount_ * headDim);
        kOutGm.SetGlobalBuffer((__gm__ T*)keyOut + (uint64_t)startRow_ * headDim,
                               (uint64_t)rowCount_ * headDim);
        cosGm.SetGlobalBuffer((__gm__ T*)cos, (uint64_t)seqLen * halfD_);
        sinGm.SetGlobalBuffer((__gm__ T*)sin, (uint64_t)seqLen * halfD_);

        // UB buffers (per-row compute; all initialized always to avoid conditional-Init hazard).
        pipe.InitBuffer(inQueueX, 1, DAlignT_ * sizeof(T));
        pipe.InitBuffer(inQueueCos, 1, halfDAlignT_ * sizeof(T));
        pipe.InitBuffer(inQueueSin, 1, halfDAlignT_ * sizeof(T));
        pipe.InitBuffer(outQueueY, 1, DAlignT_ * sizeof(T));
        pipe.InitBuffer(xFp32Buf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(cosHalfFBuf, halfDAlignF_ * sizeof(float));
        pipe.InitBuffer(sinHalfFBuf, halfDAlignF_ * sizeof(float));
        pipe.InitBuffer(cosFullFBuf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(sinFullFBuf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(rotFBuf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(mulFBuf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(yFp32Buf, DAlignF_ * sizeof(float));
        // interleaved offset/sign tables (built once; used only in interleaved mode).
        pipe.InitBuffer(offExpandBuf, DAlignF_ * sizeof(uint32_t));
        pipe.InitBuffer(offRotBuf, DAlignF_ * sizeof(uint32_t));
        pipe.InitBuffer(signFullBuf, DAlignF_ * sizeof(float));

        if (mode_ == 1) {
            BuildInterleavedTables();
        }
    }

    __aicore__ inline void Process() {
        if (rowCount_ == 0) {
            return;
        }
        // Process q rows then k rows (same row range; cos/sin reloaded per row by s index).
        ProcessTensor(qGm, qOutGm);
        ProcessTensor(kGm, kOutGm);
    }

private:
    __aicore__ inline void ProcessTensor(GlobalTensor<T>& xGm, GlobalTensor<T>& yGm) {
        for (uint32_t r = 0; r < rowCount_; r++) {
            ComputeRow(r, xGm, yGm);
        }
    }

    __aicore__ inline uint32_t RowToSeqIdx(uint32_t r) {
        // r is the per-core local row index; absolute row = startRow_ + r.
        uint32_t absRow = startRow_ + r;
        if (layout_ == 0) {
            // (B,S,N,D): absRow = b*(S*N) + s*N + n  -> s = (absRow / N) % S
            return (absRow / numHeads_) % seqLen_;
        } else {
            // (B,N,S,D): absRow = b*(N*S) + n*S + s  -> s = absRow % S
            return absRow % seqLen_;
        }
    }

    __aicore__ inline void ComputeRow(uint32_t r, GlobalTensor<T>& xGm, GlobalTensor<T>& yGm) {
        uint32_t s = RowToSeqIdx(r);

        // --- CopyIn x ---
        LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyX{1, static_cast<uint32_t>(D_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padX{false, 0, 0, 0};
        DataCopyPad(xUb, xGm[(uint64_t)r * D_], copyX, padX);
        inQueueX.EnQue(xUb);
        LocalTensor<T> xUbDeq = inQueueX.DeQue<T>();

        // --- CopyIn cos / sin (half-D rows) ---
        LocalTensor<T> cosUb = inQueueCos.AllocTensor<T>();
        DataCopyExtParams copyC{1, static_cast<uint32_t>(halfD_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padC{false, 0, 0, 0};
        DataCopyPad(cosUb, cosGm[(uint64_t)s * halfD_], copyC, padC);
        inQueueCos.EnQue(cosUb);
        LocalTensor<T> cosUbDeq = inQueueCos.DeQue<T>();

        LocalTensor<T> sinUb = inQueueSin.AllocTensor<T>();
        DataCopyPad(sinUb, sinGm[(uint64_t)s * halfD_], copyC, padC);
        inQueueSin.EnQue(sinUb);
        LocalTensor<T> sinUbDeq = inQueueSin.DeQue<T>();

        // --- Cast x -> fp32 (fp32 input: DataCopy; fp16/bf16: Cast CAST_NONE) ---
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(xFp32, xUbDeq, DAlignF_);
        } else {
            Cast(xFp32, xUbDeq, RoundMode::CAST_NONE, D_);
        }
        PipeBarrier<PIPE_V>();

        // --- Cast cos/sin -> fp32 ---
        LocalTensor<float> cosHalfF = cosHalfFBuf.Get<float>();
        LocalTensor<float> sinHalfF = sinHalfFBuf.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(cosHalfF, cosUbDeq, halfDAlignF_);
            DataCopy(sinHalfF, sinUbDeq, halfDAlignF_);
        } else {
            Cast(cosHalfF, cosUbDeq, RoundMode::CAST_NONE, halfD_);
            Cast(sinHalfF, sinUbDeq, RoundMode::CAST_NONE, halfD_);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> cosFullF = cosFullFBuf.Get<float>();
        LocalTensor<float> sinFullF = sinFullFBuf.Get<float>();
        LocalTensor<float> rotF = rotFBuf.Get<float>();

        if (mode_ == 0) {
            // half: cos_full = [cos_half | cos_half] (repeat-2x)
            DataCopy(cosFullF, cosHalfF, halfD_);
            DataCopy(cosFullF[halfD_], cosHalfF, halfD_);
            DataCopy(sinFullF, sinHalfF, halfD_);
            DataCopy(sinFullF[halfD_], sinHalfF, halfD_);
            PipeBarrier<PIPE_V>();
            // rotate_half = concat(-x2, x1): rot[0..h) <- x[h..D) then negate; rot[h..D) <- x[0..h)
            DataCopy(rotF, xFp32[halfD_], halfD_);
            DataCopy(rotF[halfD_], xFp32, halfD_);
            PipeBarrier<PIPE_V>();
            Muls(rotF, rotF, -1.0f, halfD_);   // negate first half -> [-x2 | x1]
            PipeBarrier<PIPE_V>();
        } else {
            // interleaved: cos_full[2k]=cos_full[2k+1]=cos_half[k] via Gather(offExpand[i]=i/2 bytes)
            Gather(cosFullF, cosHalfF, offExpandBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            Gather(sinFullF, sinHalfF, offExpandBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            PipeBarrier<PIPE_V>();
            // rotate_half interleaved: rot[i]=x[i^1], then *signFull[-1,1,-1,1..]
            Gather(rotF, xFp32, offRotBuf.Get<uint32_t>(),
                   (uint32_t)0, D_);
            PipeBarrier<PIPE_V>();
            Mul(rotF, rotF, signFullBuf.Get<float>(), D_);
            PipeBarrier<PIPE_V>();
        }

        // --- y = x*cos + rot*sin ---
        LocalTensor<float> mulF = mulFBuf.Get<float>();
        LocalTensor<float> yFp32 = yFp32Buf.Get<float>();
        Mul(mulF, xFp32, cosFullF, D_);
        PipeBarrier<PIPE_V>();
        Mul(yFp32, rotF, sinFullF, D_);
        PipeBarrier<PIPE_V>();
        Add(yFp32, yFp32, mulF, D_);
        PipeBarrier<PIPE_V>();

        // --- Cast y -> input dtype (fp32: DataCopy; fp16/bf16: Cast CAST_RINT = round-to-nearest-even) ---
        LocalTensor<T> yUb = outQueueY.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(yUb, yFp32, DAlignF_);
        } else {
            Cast(yUb, yFp32, RoundMode::CAST_RINT, D_);
        }
        outQueueY.EnQue<T>(yUb);
        LocalTensor<T> yUbDeq = outQueueY.DeQue<T>();

        DataCopyExtParams copyY{1, static_cast<uint32_t>(D_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[(uint64_t)r * D_], yUbDeq, copyY);

        inQueueX.FreeTensor(xUbDeq);
        inQueueCos.FreeTensor(cosUbDeq);
        inQueueSin.FreeTensor(sinUbDeq);
        outQueueY.FreeTensor(yUbDeq);
    }

    __aicore__ inline void BuildInterleavedTables() {
        LocalTensor<uint32_t> offExpand = offExpandBuf.Get<uint32_t>();
        LocalTensor<uint32_t> offRot = offRotBuf.Get<uint32_t>();
        LocalTensor<float> signFull = signFullBuf.Get<float>();
        for (uint32_t i = 0; i < D_; i++) {
            offExpand.SetValue(i, (i / 2) * sizeof(float));           // byte offset within src
            offRot.SetValue(i, (i ^ 1) * sizeof(float));              // byte offset within src
            signFull.SetValue(i, ((i & 1U) == 0) ? -1.0f : 1.0f);
        }
        PipeBarrier<PIPE_V>();
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueCos;
    TQue<TPosition::VECIN, 1> inQueueSin;
    TQue<TPosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> cosHalfFBuf;
    TBuf<TPosition::VECCALC> sinHalfFBuf;
    TBuf<TPosition::VECCALC> cosFullFBuf;
    TBuf<TPosition::VECCALC> sinFullFBuf;
    TBuf<TPosition::VECCALC> rotFBuf;
    TBuf<TPosition::VECCALC> mulFBuf;
    TBuf<TPosition::VECCALC> yFp32Buf;
    TBuf<TPosition::VECCALC> offExpandBuf;
    TBuf<TPosition::VECCALC> offRotBuf;
    TBuf<TPosition::VECCALC> signFullBuf;
    GlobalTensor<T> qGm;
    GlobalTensor<T> kGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> qOutGm;
    GlobalTensor<T> kOutGm;
    uint32_t D_;
    uint32_t halfD_;
    uint32_t DAlignT_;
    uint32_t DAlignF_;
    uint32_t halfDAlignT_;
    uint32_t halfDAlignF_;
    uint32_t rowCount_;
    uint32_t startRow_;
    uint32_t layout_;
    uint32_t mode_;
    uint32_t seqLen_;
    uint32_t numHeads_;
};
```

### after

- path: `operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`
- symbol: `KernelApplyRotaryPosEmb<T>`
- claim: cachedS_ 控制系数重载，同一行的 q/k 共用 resident cosFullF/sinFullF

```cpp
template <typename T>
class KernelApplyRotaryPosEmb {
public:
    __aicore__ inline KernelApplyRotaryPosEmb() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR cos, GM_ADDR sin,
                               GM_ADDR queryOut, GM_ADDR keyOut,
                               uint32_t totalRows, uint32_t headDim,
                               uint32_t rowsPerBlock, uint32_t layout, uint32_t rotaryMode,
                               uint32_t seqLen, uint32_t numHeads) {
        D_ = headDim;
        halfD_ = headDim / 2;
        layout_ = layout;
        mode_ = rotaryMode;          // 0 = half, 1 = interleaved
        seqLen_ = seqLen;
        numHeads_ = numHeads;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t startRow = blockIdx * rowsPerBlock;
        if (startRow >= totalRows) {
            rowCount_ = 0;
            startRow_ = 0;
        } else {
            startRow_ = startRow;
            uint32_t remaining = totalRows - startRow;
            rowCount_ = (rowsPerBlock < remaining) ? rowsPerBlock : remaining;
        }

        // 32B-aligned allocation counts (padding tail never read).
        constexpr uint32_t alignT = 32 / sizeof(T);      // 16 fp16/bf16, 8 fp32
        constexpr uint32_t alignF = 32 / sizeof(float);   // 8
        DAlignT_ = (headDim + alignT - 1) / alignT * alignT;
        DAlignF_ = (headDim + alignF - 1) / alignF * alignF;
        halfDAlignT_ = (halfD_ + alignT - 1) / alignT * alignT;
        halfDAlignF_ = (halfD_ + alignF - 1) / alignF * alignF;

        // GM buffers (per-core row range for q/k; full cos/sin).
        qGm.SetGlobalBuffer((__gm__ T*)query + (uint64_t)startRow_ * headDim,
                            (uint64_t)rowCount_ * headDim);
        kGm.SetGlobalBuffer((__gm__ T*)key + (uint64_t)startRow_ * headDim,
                            (uint64_t)rowCount_ * headDim);
        qOutGm.SetGlobalBuffer((__gm__ T*)queryOut + (uint64_t)startRow_ * headDim,
                               (uint64_t)rowCount_ * headDim);
        kOutGm.SetGlobalBuffer((__gm__ T*)keyOut + (uint64_t)startRow_ * headDim,
                               (uint64_t)rowCount_ * headDim);
        cosGm.SetGlobalBuffer((__gm__ T*)cos, (uint64_t)seqLen * halfD_);
        sinGm.SetGlobalBuffer((__gm__ T*)sin, (uint64_t)seqLen * halfD_);

        // UB buffers (per-row compute; all initialized always to avoid conditional-Init hazard).
        pipe.InitBuffer(inQueueX, 1, DAlignT_ * sizeof(T));
        pipe.InitBuffer(inQueueCos, 1, halfDAlignT_ * sizeof(T));
        pipe.InitBuffer(inQueueSin, 1, halfDAlignT_ * sizeof(T));
        pipe.InitBuffer(outQueueY, 1, DAlignT_ * sizeof(T));
        pipe.InitBuffer(xFp32Buf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(cosHalfFBuf, halfDAlignF_ * sizeof(float));
        pipe.InitBuffer(sinHalfFBuf, halfDAlignF_ * sizeof(float));
        pipe.InitBuffer(cosFullFBuf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(sinFullFBuf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(rotFBuf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(mulFBuf, DAlignF_ * sizeof(float));
        pipe.InitBuffer(yFp32Buf, DAlignF_ * sizeof(float));
        // interleaved offset/sign tables (built once; used only in interleaved mode).
        pipe.InitBuffer(offExpandBuf, DAlignF_ * sizeof(uint32_t));
        pipe.InitBuffer(offRotBuf, DAlignF_ * sizeof(uint32_t));
        pipe.InitBuffer(signFullBuf, DAlignF_ * sizeof(float));

        if (mode_ == 1) {
            BuildInterleavedTables();
        }
        cachedS_ = ~0u;  // invalid -> first row always reloads cos/sin
    }

    __aicore__ inline void Process() {
        if (rowCount_ == 0) {
            return;
        }
        // AFTER (P1): single row loop; cos/sin cached per-s (reload only on s-change) and shared
        // between q and k. Layout=0: s stable across N heads -> N x fewer cos/sin GM DMA;
        // layout=1: s changes every row but q/k share -> halve. Math identical to BEFORE.
        for (uint32_t r = 0; r < rowCount_; r++) {
            uint32_t s = RowToSeqIdx(r);
            EnsureCosSinLoaded(s);
            ComputeOne(r, qGm, qOutGm);
            ComputeOne(r, kGm, kOutGm);
        }
    }

private:
    __aicore__ inline uint32_t RowToSeqIdx(uint32_t r) {
        // r is the per-core local row index; absolute row = startRow_ + r.
        uint32_t absRow = startRow_ + r;
        if (layout_ == 0) {
            // (B,S,N,D): absRow = b*(S*N) + s*N + n  -> s = (absRow / N) % S
            return (absRow / numHeads_) % seqLen_;
        } else {
            // (B,N,S,D): absRow = b*(N*S) + n*S + s  -> s = absRow % S
            return absRow % seqLen_;
        }
    }

    // Load+expand cos/sin for sequence index s into resident cosFullF/sinFullF; reload only on s change.
    // First MTE2->V sync via TQue DeQue (avoids D26b bare-TBuf hazard); resident TBufs are V-pipe only.
    __aicore__ inline void EnsureCosSinLoaded(uint32_t s) {
        if (s == cachedS_) {
            return;
        }
        DataCopyExtParams copyC{1, static_cast<uint32_t>(halfD_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padC{false, 0, 0, 0};

        LocalTensor<T> cosUb = inQueueCos.AllocTensor<T>();
        DataCopyPad(cosUb, cosGm[(uint64_t)s * halfD_], copyC, padC);
        inQueueCos.EnQue(cosUb);
        LocalTensor<T> cosUbDeq = inQueueCos.DeQue<T>();

        LocalTensor<T> sinUb = inQueueSin.AllocTensor<T>();
        DataCopyPad(sinUb, sinGm[(uint64_t)s * halfD_], copyC, padC);
        inQueueSin.EnQue(sinUb);
        LocalTensor<T> sinUbDeq = inQueueSin.DeQue<T>();

        LocalTensor<float> cosHalfF = cosHalfFBuf.Get<float>();
        LocalTensor<float> sinHalfF = sinHalfFBuf.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(cosHalfF, cosUbDeq, halfDAlignF_);
            DataCopy(sinHalfF, sinUbDeq, halfDAlignF_);
        } else {
            Cast(cosHalfF, cosUbDeq, RoundMode::CAST_NONE, halfD_);
            Cast(sinHalfF, sinUbDeq, RoundMode::CAST_NONE, halfD_);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> cosFullF = cosFullFBuf.Get<float>();
        LocalTensor<float> sinFullF = sinFullFBuf.Get<float>();
        if (mode_ == 0) {
            // half: cos_full = [cos_half | cos_half] (repeat-2x)
            DataCopy(cosFullF, cosHalfF, halfD_);
            DataCopy(cosFullF[halfD_], cosHalfF, halfD_);
            DataCopy(sinFullF, sinHalfF, halfD_);
            DataCopy(sinFullF[halfD_], sinHalfF, halfD_);
        } else {
            // interleaved: cos_full[2k]=cos_full[2k+1]=cos_half[k] via Gather(offExpand byte offsets)
            Gather(cosFullF, cosHalfF, offExpandBuf.Get<uint32_t>(), (uint32_t)0, D_);
            Gather(sinFullF, sinHalfF, offExpandBuf.Get<uint32_t>(), (uint32_t)0, D_);
        }
        PipeBarrier<PIPE_V>();

        inQueueCos.FreeTensor(cosUbDeq);
        inQueueSin.FreeTensor(sinUbDeq);
        cachedS_ = s;
    }

    // Compute y = x*cos_full + rotate_half(x)*sin_full for one row, using cached cosFullF/sinFullF.
    __aicore__ inline void ComputeOne(uint32_t r, GlobalTensor<T>& xGm, GlobalTensor<T>& yGm) {
        // --- CopyIn x ---
        LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyX{1, static_cast<uint32_t>(D_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padX{false, 0, 0, 0};
        DataCopyPad(xUb, xGm[(uint64_t)r * D_], copyX, padX);
        inQueueX.EnQue(xUb);
        LocalTensor<T> xUbDeq = inQueueX.DeQue<T>();

        // --- Cast x -> fp32 (fp32: DataCopy; fp16/bf16: Cast CAST_NONE) ---
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(xFp32, xUbDeq, DAlignF_);
        } else {
            Cast(xFp32, xUbDeq, RoundMode::CAST_NONE, D_);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> cosFullF = cosFullFBuf.Get<float>();
        LocalTensor<float> sinFullF = sinFullFBuf.Get<float>();
        LocalTensor<float> rotF = rotFBuf.Get<float>();

        if (mode_ == 0) {
            // rotate_half = concat(-x2, x1): rot[0..h) <- x[h..D) then negate; rot[h..D) <- x[0..h)
            DataCopy(rotF, xFp32[halfD_], halfD_);
            DataCopy(rotF[halfD_], xFp32, halfD_);
            PipeBarrier<PIPE_V>();
            Muls(rotF, rotF, -1.0f, halfD_);   // negate first half -> [-x2 | x1]
            PipeBarrier<PIPE_V>();
        } else {
            // rotate_half interleaved: rot[i]=x[i^1], then *signFull[-1,1,-1,1..]
            Gather(rotF, xFp32, offRotBuf.Get<uint32_t>(), (uint32_t)0, D_);
            PipeBarrier<PIPE_V>();
            Mul(rotF, rotF, signFullBuf.Get<float>(), D_);
            PipeBarrier<PIPE_V>();
        }

        // --- y = x*cos + rot*sin ---
        LocalTensor<float> mulF = mulFBuf.Get<float>();
        LocalTensor<float> yFp32 = yFp32Buf.Get<float>();
        Mul(mulF, xFp32, cosFullF, D_);
        PipeBarrier<PIPE_V>();
        Mul(yFp32, rotF, sinFullF, D_);
        PipeBarrier<PIPE_V>();
        Add(yFp32, yFp32, mulF, D_);
        PipeBarrier<PIPE_V>();

        // --- Cast y -> input dtype (fp32: DataCopy; fp16/bf16: Cast CAST_RINT = round-to-nearest-even) ---
        LocalTensor<T> yUb = outQueueY.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(yUb, yFp32, DAlignF_);
        } else {
            Cast(yUb, yFp32, RoundMode::CAST_RINT, D_);
        }
        outQueueY.EnQue<T>(yUb);
        LocalTensor<T> yUbDeq = outQueueY.DeQue<T>();

        DataCopyExtParams copyY{1, static_cast<uint32_t>(D_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[(uint64_t)r * D_], yUbDeq, copyY);

        inQueueX.FreeTensor(xUbDeq);
        outQueueY.FreeTensor(yUbDeq);
    }

    __aicore__ inline void BuildInterleavedTables() {
        LocalTensor<uint32_t> offExpand = offExpandBuf.Get<uint32_t>();
        LocalTensor<uint32_t> offRot = offRotBuf.Get<uint32_t>();
        LocalTensor<float> signFull = signFullBuf.Get<float>();
        for (uint32_t i = 0; i < D_; i++) {
            offExpand.SetValue(i, (i / 2) * sizeof(float));           // byte offset within src
            offRot.SetValue(i, (i ^ 1) * sizeof(float));              // byte offset within src
            signFull.SetValue(i, ((i & 1U) == 0) ? -1.0f : 1.0f);
        }
        PipeBarrier<PIPE_V>();
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueCos;
    TQue<TPosition::VECIN, 1> inQueueSin;
    TQue<TPosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> cosHalfFBuf;
    TBuf<TPosition::VECCALC> sinHalfFBuf;
    TBuf<TPosition::VECCALC> cosFullFBuf;
    TBuf<TPosition::VECCALC> sinFullFBuf;
    TBuf<TPosition::VECCALC> rotFBuf;
    TBuf<TPosition::VECCALC> mulFBuf;
    TBuf<TPosition::VECCALC> yFp32Buf;
    TBuf<TPosition::VECCALC> offExpandBuf;
    TBuf<TPosition::VECCALC> offRotBuf;
    TBuf<TPosition::VECCALC> signFullBuf;
    GlobalTensor<T> qGm;
    GlobalTensor<T> kGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> qOutGm;
    GlobalTensor<T> kOutGm;
    uint32_t D_;
    uint32_t halfD_;
    uint32_t DAlignT_;
    uint32_t DAlignF_;
    uint32_t halfDAlignT_;
    uint32_t halfDAlignF_;
    uint32_t rowCount_;
    uint32_t startRow_;
    uint32_t layout_;
    uint32_t mode_;
    uint32_t seqLen_;
    uint32_t numHeads_;
    uint32_t cachedS_;   // AFTER (P1): cached sequence index for cos/sin resident reuse
};
```

正式 AFTER 与成对比较命令（trace 中原样执行）：

## 设计决策

- 选用“按 s 缓存 + q/k 共享”作为一个可回滚的系数搬运削减候选：layout=0 可跨连续 head 复用，layout=1 即使 s 每行变化，仍能在 q/k 间复用一次。
- 首次 MTE2→V 同步仍通过 `TQue` 的 EnQue/DeQue 建立；常驻的 `TBuf` 只承载 V-pipe 后续读取，避免把同步假设藏在裸 TBuf 搬运中。
- 未同时加入双缓冲、多行 tile 或删除 PipeBarrier，避免多机制叠加导致 A/B 无法归因。
- 设计评审曾担心“缓存 + q/k 共享”不是最小改动；保留 q/k 共享是因为 layout=1 下单纯按 s 缓存没有跨行复用。当前方案属于 `recommended_fastpath`。

## 性能影响

- 同一 a3 平台、同一正式用例集与冻结测试下，AFTER 相对 BEFORE 的平均 speedup 比值提升 `27.7%`（来源平台：a3；证据等级：artifact_verified）。
- 20 项成对用例全部变快；按 layout 分组，layout=0 平均 elapsed 下降约 `22.0%`，layout=1 下降约 `33.3%`（参照路线：未缓存且 q/k 分离的 BEFORE；来源平台：a3；证据等级：artifact_verified）。
- 这是一次 BEFORE/AFTER 配对，没有重复采样、方差或置信区间；结果支持该机制，但不是统计因果证明，也不应外推为通用收益。

## 预防

- 为逐行 kernel 画出只读系数的索引：若多个输入或连续行映射到同一索引，先找可驻留的 UB 表。
- 缓存状态必须用无效哨兵初始化，保证第一行必加载。
- 把 MTE2→V 的同步责任留在明确的队列协议中；不要因为 TBuf 可复用就删除首次同步。
- 性能改动前后冻结源码、构建和测试 blob，并做逐用例成对比较与正确性等价检查。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：冻结基线正确性为 20/20。（来源平台：a3；证据等级: tool_observed）。
- 修复后：AFTER 正确性仍为 20/20，同负载平均 speedup 相对冻结基线提升 27.7%，无正确性回退。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- layout=1 下 s 每行变化，跨行缓存无收益，主要收益来自 q/k 共享；若只有单输入则这一部分也消失。
- 只验证了当前 UB 容量足以常驻一行展开系数的行宽；更大行宽必须重新计算 UB 占用。
- 当前结果仍低于参照路径，不能把该候选宣称为全局性能最优；多行 tile、双缓冲等方向未在本闭环验证。
- 其他平台未验证；推断可迁移前须重跑 profiler 与成对正式评测。

## 坏实践

- 在 q 和 k 两个独立全量循环中无条件重复搬入相同 cos/sin。
- 同时引入缓存、双缓冲、多行 tile 与屏障删除，再只用一组 A/B 把全部收益归给其中一项。

## 误诊

仅凭 kernel 占比高不能断言是算力瓶颈；本次结论还结合了源码中的逐行重复 DMA、Wait Time 和修改后的成对结果。