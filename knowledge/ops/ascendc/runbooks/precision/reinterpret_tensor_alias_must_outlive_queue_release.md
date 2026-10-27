---
type: "Runbook"
title: "FreeTensor早释使队列别名被下一轮DMA覆盖"
description: "fp32路径让计算张量别名队列张量时，compute完成前FreeTensor会触发UB复用，造成非确定性NaN和有限值错误。"
tags: ["task.debugging", "topic.implementation.memory_management", "topic.implementation.synchronization", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# FreeTensor早释使队列别名被下一轮DMA覆盖

## 现象

`apply_rotary_pos_emb.asc` 在本地小用例已经通过后，正式 a3 评测仍有少量 fp32 用例失败：一个大幅值相消用例出现异常相对误差，另一个 NaN 混合输入出现 NaN 位置不一致。决定性信号不是单次误差大小，而是同一 seed、同一已编译共享库重复执行时，NPU 输出的 NaN 数量在不同进程中变化；CPU 参考的 NaN 数量保持不变。

排障先把复用的 `t1/t2` 改成四个独立 scratch 并增加 V 屏障，构建成功后非确定性仍然存在。这证明 scratch RAW/WAW 不是主根因。随后检查输入队列生命周期，发现 fp32 路径令 `xT` 直接别名 `xIO`，却在任何 `Mul` 读取 `xT` 之前调用了 `xInQ.FreeTensor(xIO)`。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`FreeTensor`、`TQue`、`ReinterpretCast`、`LocalTensor` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

fp32 路径执行：

1. `xIO = xInQ.DeQue<T>()`；
2. `xT = xIO.ReinterpretCast<CT>()`，这里 `T` 与 `CT` 都是 fp32，`xT` 仍指向 `xIO` 的 UB；
3. 立即 `xInQ.FreeTensor(xIO)`；
4. 之后才用 `Mul`、`Gather` 等 V 指令读取 `xT`。

`FreeTensor` 把该队列槽标为可复用，不表示所有持有它的视图已经完成消费。下一子 tile 的 `AllocTensor + DataCopyPad` 可以用 MTE2 覆盖同一 UB，而前一 tile 的 V 流水仍在读取，于是产生真正的数据竞争：有时读到 NaN，有时读到错误有限值，表现随运行时序变化。

低精度路径先 Cast 到独立 `xFBuf`，后续 `xT` 不再别名 `xIO`，所以该错误对 fp32 更敏感。

## 解决配方

把 `FreeTensor(xIO)` 从别名创建后移动到本 tile 所有 V 计算完成之后、输出搬运之前。生命周期边界应围绕“最后一次实际消费”，而不是围绕“最后一次 C++ 变量赋值”。

### before

来源状态：四 scratch 误诊修正后、生命周期修复前的工作树快照；路径 `apply_rotary_pos_emb.asc`；符号 `apply_rotary_pos_emb_kernel<T>`；行 24–247。

```cpp
template <typename T>  // T = IO dtype (float / half / bfloat16_t)
__global__ __aicore__ void apply_rotary_pos_emb_kernel(
    GM_ADDR qGm, GM_ADDR kGm, GM_ADDR cosGm, GM_ADDR sinGm,
    GM_ADDR qOutGm, GM_ADDR kOutGm,
    int64_t totalGroups, int64_t S, int64_t N, int64_t D, int64_t rotaryMode,
    int64_t groupsPerCore, int64_t subRows)
{
    using CT = float;
    const int64_t half = D / 2;
    const int64_t dBytes = D * (int64_t)sizeof(CT);
    const int64_t hBytes = half * (int64_t)sizeof(CT);
    const bool isHalf = (rotaryMode == 0);
    const bool isCast = !std::is_same<T, CT>::value;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> xInQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outOutQ;
    pipe.InitBuffer(xInQ, PIPELINE_DEPTH, subRows * dBytes);
    pipe.InitBuffer(outOutQ, PIPELINE_DEPTH, subRows * dBytes);

    AscendC::TBuf xFBuf;
    pipe.InitBuffer(xFBuf, subRows * dBytes);
    AscendC::TBuf outFBuf;
    pipe.InitBuffer(outFBuf, subRows * dBytes);

    AscendC::TBuf cosFBuf, sinFBuf;
    pipe.InitBuffer(cosFBuf, dBytes);
    pipe.InitBuffer(sinFBuf, dBytes);
    AscendC::LocalTensor<CT> cosF;
    AscendC::LocalTensor<CT> sinF;

    // TQue 承接 cos/sin 的 MTE2 载入：DeQue 自动同步 MTE2→V（§2.9 TBuf MTE2→V 骨架，
    // 避免 Vector 读到 DMA 未完成/被覆盖的 TBuf）。fp32 直接作 cosF；fp16/bf16 作 cosIO 供 Cast。
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> cosQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> sinQue;
    pipe.InitBuffer(cosQue, 1, dBytes);
    pipe.InitBuffer(sinQue, 1, dBytes);

    AscendC::TBuf t1Buf, t2Buf, t3Buf, t4Buf;
    pipe.InitBuffer(t1Buf, hBytes);
    pipe.InitBuffer(t2Buf, hBytes);
    pipe.InitBuffer(t3Buf, hBytes);
    pipe.InitBuffer(t4Buf, hBytes);
    auto t1 = t1Buf.Get<CT>();
    auto t2 = t2Buf.Get<CT>();
    auto t3 = t3Buf.Get<CT>();
    auto t4 = t4Buf.Get<CT>();

    AscendC::TBuf catBuf, xeBuf, xoBuf, offEvenBuf, offOddBuf, offInterBuf;
    AscendC::LocalTensor<uint32_t> offEven;
    AscendC::LocalTensor<uint32_t> offOdd;
    AscendC::LocalTensor<uint32_t> offInter;
    AscendC::LocalTensor<CT> xe;
    AscendC::LocalTensor<CT> xo;
    AscendC::LocalTensor<CT> cat;
    if (!isHalf) {
        pipe.InitBuffer(xeBuf, hBytes);
        pipe.InitBuffer(xoBuf, hBytes);
        pipe.InitBuffer(catBuf, dBytes);
        xe = xeBuf.Get<CT>();
        xo = xoBuf.Get<CT>();
        cat = catBuf.Get<CT>();
        pipe.InitBuffer(offEvenBuf, half * (int64_t)sizeof(uint32_t));
        pipe.InitBuffer(offOddBuf, half * (int64_t)sizeof(uint32_t));
        pipe.InitBuffer(offInterBuf, D * (int64_t)sizeof(uint32_t));
        offEven = offEvenBuf.Get<uint32_t>();
        offOdd = offOddBuf.Get<uint32_t>();
        offInter = offInterBuf.Get<uint32_t>();
        const uint32_t sz = (uint32_t)sizeof(CT);
        for (int64_t k = 0; k < half; ++k) {
            offEven.SetValue(k, (uint32_t)(2 * k) * sz);
            offOdd.SetValue(k, (uint32_t)(2 * k + 1) * sz);
            offInter.SetValue(2 * k, (uint32_t)k * sz);
            offInter.SetValue(2 * k + 1, (uint32_t)(half + k) * sz);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::GlobalTensor<T> qG, kG, qOutG, kOutG, cosG, sinG;
    qG.SetGlobalBuffer((__gm__ T *)qGm);
    kG.SetGlobalBuffer((__gm__ T *)kGm);
    qOutG.SetGlobalBuffer((__gm__ T *)qOutGm);
    kOutG.SetGlobalBuffer((__gm__ T *)kOutGm);
    cosG.SetGlobalBuffer((__gm__ T *)cosGm);
    sinG.SetGlobalBuffer((__gm__ T *)sinGm);

    AscendC::DataCopyExtParams copyCSio{1, static_cast<uint32_t>(half * (int64_t)sizeof(T)), 0, 0, 0};
    AscendC::DataCopyExtParams copyCSf{1, static_cast<uint32_t>(hBytes), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padCS{false, 0, 0, 0};

    const int64_t coreId = AscendC::GetBlockIdx();
    const int64_t gStart = coreId * groupsPerCore;
    const int64_t gEnd = (gStart + groupsPerCore < totalGroups) ? (gStart + groupsPerCore) : totalGroups;

    for (int64_t g = gStart; g < gEnd; ++g) {
        const int64_t s = g % S;
        const int64_t grpBase = g * N * D;

        // 构建 cosFull=[c,c]、sinFull=[-s,+s]（fp32）。cos/sin 经 TQue 载入，DeQue 同步 MTE2→V。
        if constexpr (isCast) {
            auto cosIO = cosQue.AllocTensor<T>();
            auto sinIO = sinQue.AllocTensor<T>();
            AscendC::DataCopyPad(cosIO, cosG[s * half], copyCSio, padCS);
            AscendC::DataCopyPad(sinIO, sinG[s * half], copyCSio, padCS);
            cosQue.EnQue(cosIO);
            sinQue.EnQue(sinIO);
            cosIO = cosQue.DeQue<T>();
            sinIO = sinQue.DeQue<T>();
            cosF = cosFBuf.Get<CT>();
            sinF = sinFBuf.Get<CT>();
            AscendC::Cast(cosF, cosIO, AscendC::RoundMode::CAST_NONE, half);
            AscendC::Cast(cosF[half], cosIO, AscendC::RoundMode::CAST_NONE, half);
            AscendC::Cast(sinF, sinIO, AscendC::RoundMode::CAST_NONE, half);
            AscendC::Cast(sinF[half], sinIO, AscendC::RoundMode::CAST_NONE, half);
            AscendC::PipeBarrier<PIPE_V>();
            cosQue.FreeTensor(cosIO);
            sinQue.FreeTensor(sinIO);
        } else {
            cosF = cosQue.AllocTensor<CT>();
            sinF = sinQue.AllocTensor<CT>();
            AscendC::DataCopyPad(cosF, cosG[s * half], copyCSf, padCS);
            AscendC::DataCopyPad(cosF[half], cosG[s * half], copyCSf, padCS);
            AscendC::DataCopyPad(sinF, sinG[s * half], copyCSf, padCS);
            AscendC::DataCopyPad(sinF[half], sinG[s * half], copyCSf, padCS);
            cosQue.EnQue(cosF);
            sinQue.EnQue(sinF);
            cosF = cosQue.DeQue<CT>();
            sinF = sinQue.DeQue<CT>();
        }
        if (isHalf) {
            AscendC::Muls(sinF, sinF, CT(-1.0), half);  // 首半取负 → [-s, +s]
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (int64_t which = 0; which < 2; ++which) {
            AscendC::GlobalTensor<T> xG = (which == 0) ? qG : kG;
            AscendC::GlobalTensor<T> oG = (which == 0) ? qOutG : kOutG;
            for (int64_t base = 0; base < N; base += subRows) {
                int64_t rows = (base + subRows < N) ? subRows : (N - base);
                AscendC::DataCopyExtParams copyIn{1, static_cast<uint32_t>(rows * D * (int64_t)sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, 0};

                auto xIO = xInQ.AllocTensor<T>();
                AscendC::DataCopyPad(xIO, xG[grpBase + base * D], copyIn, padIn);
                xInQ.EnQue(xIO);
                xIO = xInQ.DeQue<T>();

                AscendC::LocalTensor<CT> xT;
                if constexpr (isCast) {
                    auto xF = xFBuf.Get<CT>();
                    AscendC::Cast(xF, xIO, AscendC::RoundMode::CAST_NONE, rows * D);
                    AscendC::PipeBarrier<PIPE_V>();
                    xT = xF;
                } else {
                    xT = xIO.template ReinterpretCast<CT>();
                }
                xInQ.FreeTensor(xIO);

                auto outIO = outOutQ.AllocTensor<T>();
                AscendC::LocalTensor<CT> outT;
                if constexpr (isCast) {
                    outT = outFBuf.Get<CT>();
                } else {
                    outT = outIO.template ReinterpretCast<CT>();
                }

                if (isHalf) {
                    for (int64_t r = 0; r < rows; ++r) {
                        const int64_t rb = r * D;
                        // 4 独立 scratch（写一次读一次），避免 t1/t2 复用致 V 管线 RAW/WAW 竞争。
                        AscendC::Mul(t1, xT[rb], cosF, half);            // x[:h]*c
                        AscendC::Mul(t2, xT[rb + half], sinF, half);      // x[h:]*(-s)
                        AscendC::Mul(t3, xT[rb + half], cosF[half], half); // x[h:]*c
                        AscendC::Mul(t4, xT[rb], sinF[half], half);        // x[:h]*(+s)
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(outT[rb], t1, t2, half);            // out[:h]=x[:h]*c - x[h:]*s
                        AscendC::Add(outT[rb + half], t3, t4, half);     // out[h:]=x[h:]*c + x[:h]*s
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    for (int64_t r = 0; r < rows; ++r) {
                        const int64_t rb = r * D;
                        AscendC::Gather(xe, xT[rb], offEven, (uint32_t)0, half);
                        AscendC::Gather(xo, xT[rb], offOdd, (uint32_t)0, half);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(t1, xe, cosF, half);     // c*xe
                        AscendC::Mul(t2, xo, sinF, half);     // s*xo
                        AscendC::Mul(t3, xo, cosF, half);     // c*xo
                        AscendC::Mul(t4, xe, sinF, half);     // s*xe
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Sub(cat, t1, t2, half);      // cat[:h]=ye=c*xe - s*xo
                        AscendC::Add(cat[half], t3, t4, half); // cat[h:]=yo=c*xo + s*xe
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Gather(outT[rb], cat, offInter, (uint32_t)0, D); // 重交错
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }

                AscendC::DataCopyExtParams copyOut{1, static_cast<uint32_t>(rows * D * (int64_t)sizeof(T)), 0, 0, 0};
                if (isCast) {
                    AscendC::Cast(outIO, outT, AscendC::RoundMode::CAST_ROUND, rows * D);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                outOutQ.EnQue(outIO);
                outIO = outOutQ.DeQue<T>();
                AscendC::DataCopyPad(oG[grpBase + base * D], outIO, copyOut);
                outOutQ.FreeTensor(outIO);
            }
        }
        // 释放本组 cosF/sinF（fp32 为队列张量；fp16/bf16 为 TBuf 无需 Free）。
        if (!isCast) {
            cosQue.FreeTensor(cosF);
            sinQue.FreeTensor(sinF);
        }
    }
}

struct RopeTiling {
    int64_t numBlocks;
    int64_t groupsPerCore;
    int64_t subRows;
};

// UB: 两队列(depth2=4) + xF+outF = 6·subRows·dBytes + fixed。
```

### after

来源状态：commit `ef76ca0` 对应修复后快照；路径 `apply_rotary_pos_emb.asc`；符号 `apply_rotary_pos_emb_kernel<T>`；行 24–248。

```cpp
template <typename T>  // T = IO dtype (float / half / bfloat16_t)
__global__ __aicore__ void apply_rotary_pos_emb_kernel(
    GM_ADDR qGm, GM_ADDR kGm, GM_ADDR cosGm, GM_ADDR sinGm,
    GM_ADDR qOutGm, GM_ADDR kOutGm,
    int64_t totalGroups, int64_t S, int64_t N, int64_t D, int64_t rotaryMode,
    int64_t groupsPerCore, int64_t subRows)
{
    using CT = float;
    const int64_t half = D / 2;
    const int64_t dBytes = D * (int64_t)sizeof(CT);
    const int64_t hBytes = half * (int64_t)sizeof(CT);
    const bool isHalf = (rotaryMode == 0);
    const bool isCast = !std::is_same<T, CT>::value;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> xInQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outOutQ;
    pipe.InitBuffer(xInQ, PIPELINE_DEPTH, subRows * dBytes);
    pipe.InitBuffer(outOutQ, PIPELINE_DEPTH, subRows * dBytes);

    AscendC::TBuf xFBuf;
    pipe.InitBuffer(xFBuf, subRows * dBytes);
    AscendC::TBuf outFBuf;
    pipe.InitBuffer(outFBuf, subRows * dBytes);

    AscendC::TBuf cosFBuf, sinFBuf;
    pipe.InitBuffer(cosFBuf, dBytes);
    pipe.InitBuffer(sinFBuf, dBytes);
    AscendC::LocalTensor<CT> cosF;
    AscendC::LocalTensor<CT> sinF;

    // TQue 承接 cos/sin 的 MTE2 载入：DeQue 自动同步 MTE2→V（§2.9 TBuf MTE2→V 骨架，
    // 避免 Vector 读到 DMA 未完成/被覆盖的 TBuf）。fp32 直接作 cosF；fp16/bf16 作 cosIO 供 Cast。
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> cosQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> sinQue;
    pipe.InitBuffer(cosQue, 1, dBytes);
    pipe.InitBuffer(sinQue, 1, dBytes);

    AscendC::TBuf t1Buf, t2Buf, t3Buf, t4Buf;
    pipe.InitBuffer(t1Buf, hBytes);
    pipe.InitBuffer(t2Buf, hBytes);
    pipe.InitBuffer(t3Buf, hBytes);
    pipe.InitBuffer(t4Buf, hBytes);
    auto t1 = t1Buf.Get<CT>();
    auto t2 = t2Buf.Get<CT>();
    auto t3 = t3Buf.Get<CT>();
    auto t4 = t4Buf.Get<CT>();

    AscendC::TBuf catBuf, xeBuf, xoBuf, offEvenBuf, offOddBuf, offInterBuf;
    AscendC::LocalTensor<uint32_t> offEven;
    AscendC::LocalTensor<uint32_t> offOdd;
    AscendC::LocalTensor<uint32_t> offInter;
    AscendC::LocalTensor<CT> xe;
    AscendC::LocalTensor<CT> xo;
    AscendC::LocalTensor<CT> cat;
    if (!isHalf) {
        pipe.InitBuffer(xeBuf, hBytes);
        pipe.InitBuffer(xoBuf, hBytes);
        pipe.InitBuffer(catBuf, dBytes);
        xe = xeBuf.Get<CT>();
        xo = xoBuf.Get<CT>();
        cat = catBuf.Get<CT>();
        pipe.InitBuffer(offEvenBuf, half * (int64_t)sizeof(uint32_t));
        pipe.InitBuffer(offOddBuf, half * (int64_t)sizeof(uint32_t));
        pipe.InitBuffer(offInterBuf, D * (int64_t)sizeof(uint32_t));
        offEven = offEvenBuf.Get<uint32_t>();
        offOdd = offOddBuf.Get<uint32_t>();
        offInter = offInterBuf.Get<uint32_t>();
        const uint32_t sz = (uint32_t)sizeof(CT);
        for (int64_t k = 0; k < half; ++k) {
            offEven.SetValue(k, (uint32_t)(2 * k) * sz);
            offOdd.SetValue(k, (uint32_t)(2 * k + 1) * sz);
            offInter.SetValue(2 * k, (uint32_t)k * sz);
            offInter.SetValue(2 * k + 1, (uint32_t)(half + k) * sz);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::GlobalTensor<T> qG, kG, qOutG, kOutG, cosG, sinG;
    qG.SetGlobalBuffer((__gm__ T *)qGm);
    kG.SetGlobalBuffer((__gm__ T *)kGm);
    qOutG.SetGlobalBuffer((__gm__ T *)qOutGm);
    kOutG.SetGlobalBuffer((__gm__ T *)kOutGm);
    cosG.SetGlobalBuffer((__gm__ T *)cosGm);
    sinG.SetGlobalBuffer((__gm__ T *)sinGm);

    AscendC::DataCopyExtParams copyCSio{1, static_cast<uint32_t>(half * (int64_t)sizeof(T)), 0, 0, 0};
    AscendC::DataCopyExtParams copyCSf{1, static_cast<uint32_t>(hBytes), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padCS{false, 0, 0, 0};

    const int64_t coreId = AscendC::GetBlockIdx();
    const int64_t gStart = coreId * groupsPerCore;
    const int64_t gEnd = (gStart + groupsPerCore < totalGroups) ? (gStart + groupsPerCore) : totalGroups;

    for (int64_t g = gStart; g < gEnd; ++g) {
        const int64_t s = g % S;
        const int64_t grpBase = g * N * D;

        // 构建 cosFull=[c,c]、sinFull=[-s,+s]（fp32）。cos/sin 经 TQue 载入，DeQue 同步 MTE2→V。
        if constexpr (isCast) {
            auto cosIO = cosQue.AllocTensor<T>();
            auto sinIO = sinQue.AllocTensor<T>();
            AscendC::DataCopyPad(cosIO, cosG[s * half], copyCSio, padCS);
            AscendC::DataCopyPad(sinIO, sinG[s * half], copyCSio, padCS);
            cosQue.EnQue(cosIO);
            sinQue.EnQue(sinIO);
            cosIO = cosQue.DeQue<T>();
            sinIO = sinQue.DeQue<T>();
            cosF = cosFBuf.Get<CT>();
            sinF = sinFBuf.Get<CT>();
            AscendC::Cast(cosF, cosIO, AscendC::RoundMode::CAST_NONE, half);
            AscendC::Cast(cosF[half], cosIO, AscendC::RoundMode::CAST_NONE, half);
            AscendC::Cast(sinF, sinIO, AscendC::RoundMode::CAST_NONE, half);
            AscendC::Cast(sinF[half], sinIO, AscendC::RoundMode::CAST_NONE, half);
            AscendC::PipeBarrier<PIPE_V>();
            cosQue.FreeTensor(cosIO);
            sinQue.FreeTensor(sinIO);
        } else {
            cosF = cosQue.AllocTensor<CT>();
            sinF = sinQue.AllocTensor<CT>();
            AscendC::DataCopyPad(cosF, cosG[s * half], copyCSf, padCS);
            AscendC::DataCopyPad(cosF[half], cosG[s * half], copyCSf, padCS);
            AscendC::DataCopyPad(sinF, sinG[s * half], copyCSf, padCS);
            AscendC::DataCopyPad(sinF[half], sinG[s * half], copyCSf, padCS);
            cosQue.EnQue(cosF);
            sinQue.EnQue(sinF);
            cosF = cosQue.DeQue<CT>();
            sinF = sinQue.DeQue<CT>();
        }
        if (isHalf) {
            AscendC::Muls(sinF, sinF, CT(-1.0), half);  // 首半取负 → [-s, +s]
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (int64_t which = 0; which < 2; ++which) {
            AscendC::GlobalTensor<T> xG = (which == 0) ? qG : kG;
            AscendC::GlobalTensor<T> oG = (which == 0) ? qOutG : kOutG;
            for (int64_t base = 0; base < N; base += subRows) {
                int64_t rows = (base + subRows < N) ? subRows : (N - base);
                AscendC::DataCopyExtParams copyIn{1, static_cast<uint32_t>(rows * D * (int64_t)sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, 0};

                auto xIO = xInQ.AllocTensor<T>();
                AscendC::DataCopyPad(xIO, xG[grpBase + base * D], copyIn, padIn);
                xInQ.EnQue(xIO);
                xIO = xInQ.DeQue<T>();

                AscendC::LocalTensor<CT> xT;
                if constexpr (isCast) {
                    auto xF = xFBuf.Get<CT>();
                    AscendC::Cast(xF, xIO, AscendC::RoundMode::CAST_NONE, rows * D);
                    AscendC::PipeBarrier<PIPE_V>();
                    xT = xF;
                } else {
                    xT = xIO.template ReinterpretCast<CT>();
                }
                // 注意：xIO（fp32 即 xT）必须在 compute 读完后再 Free，否则下一迭代 DMA 复用覆盖。

                auto outIO = outOutQ.AllocTensor<T>();
                AscendC::LocalTensor<CT> outT;
                if constexpr (isCast) {
                    outT = outFBuf.Get<CT>();
                } else {
                    outT = outIO.template ReinterpretCast<CT>();
                }

                if (isHalf) {
                    for (int64_t r = 0; r < rows; ++r) {
                        const int64_t rb = r * D;
                        // 4 独立 scratch（写一次读一次），避免 t1/t2 复用致 V 管线 RAW/WAW 竞争。
                        AscendC::Mul(t1, xT[rb], cosF, half);            // x[:h]*c
                        AscendC::Mul(t2, xT[rb + half], sinF, half);      // x[h:]*(-s)
                        AscendC::Mul(t3, xT[rb + half], cosF[half], half); // x[h:]*c
                        AscendC::Mul(t4, xT[rb], sinF[half], half);        // x[:h]*(+s)
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(outT[rb], t1, t2, half);            // out[:h]=x[:h]*c - x[h:]*s
                        AscendC::Add(outT[rb + half], t3, t4, half);     // out[h:]=x[h:]*c + x[:h]*s
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    for (int64_t r = 0; r < rows; ++r) {
                        const int64_t rb = r * D;
                        AscendC::Gather(xe, xT[rb], offEven, (uint32_t)0, half);
                        AscendC::Gather(xo, xT[rb], offOdd, (uint32_t)0, half);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(t1, xe, cosF, half);     // c*xe
                        AscendC::Mul(t2, xo, sinF, half);     // s*xo
                        AscendC::Mul(t3, xo, cosF, half);     // c*xo
                        AscendC::Mul(t4, xe, sinF, half);     // s*xe
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Sub(cat, t1, t2, half);      // cat[:h]=ye=c*xe - s*xo
                        AscendC::Add(cat[half], t3, t4, half); // cat[h:]=yo=c*xo + s*xe
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Gather(outT[rb], cat, offInter, (uint32_t)0, D); // 重交错
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
                xInQ.FreeTensor(xIO);  // compute 已读完 xT（=xIO for fp32），可释放

                AscendC::DataCopyExtParams copyOut{1, static_cast<uint32_t>(rows * D * (int64_t)sizeof(T)), 0, 0, 0};
                if (isCast) {
                    AscendC::Cast(outIO, outT, AscendC::RoundMode::CAST_ROUND, rows * D);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                outOutQ.EnQue(outIO);
                outIO = outOutQ.DeQue<T>();
                AscendC::DataCopyPad(oG[grpBase + base * D], outIO, copyOut);
                outOutQ.FreeTensor(outIO);
            }
        }
        // 释放本组 cosF/sinF（fp32 为队列张量；fp16/bf16 为 TBuf 无需 Free）。
        if (!isCast) {
            cosQue.FreeTensor(cosF);
            sinQue.FreeTensor(sinF);
        }
    }
}

struct RopeTiling {
    int64_t numBlocks;
    int64_t groupsPerCore;
    int64_t subRows;
};

// UB: 两队列(depth2=4) + xF+outF = 6·subRows·dBytes + fixed。
```

## 系统信息

- SoC：Ascend910_9362，NpuArch dav-2201（平台注册值 a3）。
- CANN：9.2.0；Bisheng；经典 AscendC Vector 路径。
- 相关文件：`apply_rotary_pos_emb.asc`、`test_apply_rotary_pos_emb.py`。
- 验证工具：CMake、PyTorch NPU 直调确定性探针、cann-bench 正式精度/性能评测。

## 设计决策

1. 直接修复队列张量的所有权窗口，而不是继续添加 `PIPE_V` 屏障。屏障可能改变复现概率，但不能撤销已经发生的释放/复用。
2. `FreeTensor` 放在 half/interleaved 两条 compute 分支共同结束后，确保两种模式的最后一个输入消费者都已提交。
3. 四 scratch 改动被保留以减少复用复杂度，但重复探针证明它不是消除非确定性的必要根因，知识结论只归因于生命周期修复。
4. 当前方案属于 `recommended_fastpath`：不增加额外 GM 搬运，也不把队列深度降为 1 来掩盖竞争。

## 预防

- 对 `LocalTensor` 的别名、`ReinterpretCast`、切片建立“底层 owner”清单；owner 释放前检查所有异步消费者。
- `FreeTensor` 的安全点必须是最后一次硬件消费之后，而不是最后一次 host/device C++ 语句引用之前。
- 精度错误若在相同输入重复运行间变化，优先检查 UB 生命周期、跨流水同步与多核写冲突。
- 修复后重复运行同一 NaN/Inf 输入，比较 NaN mask/计数，而不仅是一次 aggregate error。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：同输入重复运行的 NaN 计数分别为 79104/79102 与 79111/79112，明确标记 NON-DET-RACE。（来源平台：a3；证据等级: tool_observed）。
- 修复后：移后 FreeTensor 后三轮 NaN 计数均为 79104/79104 deterministic。（来源平台：a3；证据等级: tool_observed）。
- 修复后：本地 NPU 测试 18 passed。（来源平台：a3；证据等级: tool_observed）。
- 修复后：正式评测失败用例数 0、通过率 100%、EVAL_EXIT=0。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- 只在 a3/CANN 9.2.0 验证；其他平台注册平台与其他 CANN 版本未验证，推断可迁移。
- 不应机械地把所有 `FreeTensor` 推到函数末尾；长期占用队列会降低流水并发。正确边界是该底层 UB 的最后一个消费者之后。
- 对独立 Cast 目标 TBuf，原 IO 队列张量可在 Cast 完成且建立相应 V 依赖后释放；需逐路径判断是否仍有别名。
- 本卡不覆盖多核 GM 写重叠；若确定性探针仍变化，应继续检查 tiling 与输出区间。

## 坏实践

- 看到非确定性就固定随机种子或只重跑到“碰巧通过”；固定输入正应被用来暴露时序差异。
- 通过增加 scratch、屏障或减小队列深度来改变复现概率，却不检查底层张量是否已被释放。
- 把 `ReinterpretCast` 当作复制；它通常只是视图，生命周期仍由原张量所有者控制。
- 修改测试 oracle、放宽 NaN 比较或跳过失败用例；这些做法不能修复真实 UB 覆盖。

## 误诊

最初假设是复用 `t1/t2` 造成 V 流水 RAW/WAW。轨迹把它改成四个独立 scratch 并重建，但非确定性仍稳定复现。这个失败尝试非常有判别力：当“计算临时量不复用”仍无效，应转向输入 owner 的释放点。