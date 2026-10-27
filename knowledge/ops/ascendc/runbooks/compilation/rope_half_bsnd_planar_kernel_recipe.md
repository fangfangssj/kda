---
type: "Runbook"
title: RoPE half BSND kernel 平面分解实现
description: dav-3510 上 RoPE(half,BSND,cos2d) kernel 的完整可复用实现配方：利用 cos/sin tile 性质把 rotate_half 代数分解为两个半区的平面连续运算，paddedH=alignUp(H,16) 对齐布局，逐行 1D NDDMA 加载，flat Muls/Mul + asc_vf_call<MulAddVF> 融合链，fp16/bf16 升 fp32 计算；规避 VEC 32B 对齐崩溃、MTE3 对齐崩溃、2D NDDMA 小内维错位三个坑。复现增量：MLA 融合小 Dr 场景的更简形态——设计核查确认 AscendC 无 RoPE 专用高阶 API；rotate_half 用 Muls(dst,src[half],-1)+Adds(dst[half],src,0) 两条带偏移元素级指令免 Concat（省 tmp buffer、避 Concat 约束），编译级 + CPU golden 位级验证。
tags: [task.implementation, topic.attention, topic.alignment, topic.compute.elementwise.arithmetic]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-17T01:10:00Z'
platforms: ["950"]
---

# RoPE half BSND kernel 平面分解实现

## 现象（任务配方入口）

在 Ascend950PR（CANN 9.2.0，dav-3510）上从零实现 RoPE 类算子（BenchHarnessApplyRotaryPosEmb，`queryOut = q·cos + rotate_half(q)·sin`，half 模式、BSND 布局、cos.ndim=2）的 Branch-0 kernel：要求支持 fp32/fp16/bf16 三 dtype、小 D（D=2/4/6）与 D=128 大 shape、B 维广播 cos（cos 形状 (S, Dh) 逻辑 / (1,S,1,Dh) 4D 对齐形）、空 S=0 结构用例，且 dav-3510 连续向量计算链须用 `__simd_vf__ + asc_vf_call`。直接实现会连环踩三个已知的 dav-3510 平台坑：VEC 源地址非 32B 对齐触发 errcode 340、MTE3 非对齐写触发 errcode 80、2D NDDMA 小内维错位。本卡给出跨过全部三个坑的终版完整实现，ST 12/12 全绿。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel（aclnn 工程，单源码经框架按 dtype 编译 3 二进制），ST 用独立 torch oracle 验收。
- API 路径：`AscendC::TPipe/TBuf/GlobalTensor/LocalTensor`、`DataCopy<T,1,cfg>`（NDDMA 1D）、`DataCopyPad`（DataCopyExtParams）、`Muls/Mul/Cast`（Level 2）、`asc_vf_call<MulAddVF>`（reg_compute LoadAlign/StoreAlign/MulAddDst）、`SetFlag/WaitFlag<HardEvent>`。
- 验证过的平台与 CANN 版本：Ascend950PR（dav-3510/arch3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"行内半区交换/取负 + 双输入逐元素线性组合"结构（RoPE half、gated 半区变换等），cos/sin 需具有前后半区 tile 性质。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：interleaved 模式（rotate_half 语义为相邻元素对交换，半区不连续，平面分解不成立）；cos 前后半区取值不同的变体；其他架构对齐粒度与 NDDMA 语义未验证（未验证，推断可迁移但须复核常量）。

## 根因/方案要点（为什么这样设计）

1. **平面代数分解**：`cos[:,0:H]==cos[:,H:D]=c`（tile 性质）使 `out = q·cos + rotate_half(q)·sin` 可分解为 `out1 = q1·c − q2·s`、`out2 = q2·c + q1·s`（q1/q2 为行前/后半区）。rotate_half 的"行内半区交换+取负"被消解为两个平面缓冲各自的连续线性组合——从根上消除行内非对齐向量访问（VEC errcode 340 根因）。
2. **paddedH=alignUp(H,16) 行距**：fp32 8 元素/32B、fp16/bf16 16 元素/32B，取 16 统一覆盖；每行 CopyOut 源偏移 `row×paddedH×sizeof(T)` 恒 32B 对齐（MTE3 errcode 80 根因）。padding 区为垃圾但不被 CopyOut、不污染有效位。
3. **逐行 1D NDDMA 加载**：q1/q2 行分片与 cos/sin 按 `s=(absRow/N)%S` 的行级广播都用 `NdDmaParams<T,1>` 每行显式定址（2D NDDMA 小内维错位根因）。
4. **`asc_vf_call<MulAddVF>`**：dav-3510 连续向量计算链须用 `__simd_vf__ + asc_vf_call`，Mul→Add 融合为 reg_compute 的 `MulAddDst`（`CreateAddrReg/UpdateMask/LoadAlign/StoreAlign`）。
5. **fp16/bf16 升 fp32**：输入/输出按原 dtype 搬运，计算链前 `Cast(CAST_NONE)` 升 fp32、后 `Cast(CAST_RINT)` 回原 dtype（round-to-nearest-even，与 numpy astype 一致）。
6. **dtype 分发**：框架把单源码按 dtype 编 3 个二进制（编译期 `-DDTYPE_QUERY=float/half/bfloat16_t`），kernel 内 `Branch0Kernel<DTYPE_QUERY>` 一份代码三 dtype。

## 设计决策

- 备选被放弃：行 pad 到 32B（行内半区起点仍非对齐，且 7 缓冲×pad 超 UB 预算）；`Reg::LoadUnAlign/StoreUnAlign` 做行内交换（跨半区 Gather/Interleave 组合复杂、H 运行时变化）；UB→UB `DataCopy(DataCopyParams)`（blockLen 以 32B 块为单位，小 H 无法表达）；`DataCopyPad(DataCopyExtParams)` 跨步块加载 q1/q2（`dstStride310=AlignUp(dstStride*32+burstLen,32)` 的粒度换算使 paddedH 行距难精确表达）。
- 流水线事件：单 chunk 内 MTE2(加载)→V(计算 q)→V_MTE3→MTE3(CopyOut q)→MTE2(加载 k，复用 cos/sin)→V(计算 k)→V_MTE3→MTE3(CopyOut k)；跨 chunk 用 MTE3_MTE2 保证上轮 CopyOut 完成后覆写 UB。事件 ID 经 `GetTPipePtr()->FetchEventID` 取得（全局函数，勿加 `AscendC::` 前缀）。
- 多核切分：`GetBlockIdx/GetBlockNum` 按行区间静态切分，末核吃余量行；行数按 UB 预算 `usableUB(196608B)/bytesPerRow` 反推 chunk 大小。
- 当前方案路线档位：recommended_fastpath（正确性优先；逐行 1D NDDMA 与 2D 的性能差未测量）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0（dav-3510）。
- 目录布局：`operators/BenchHarnessApplyRotaryPosEmb_package/op_kernel/bench_harness_apply_rotary_pos_emb_kernel.cpp`（kernel 主体）、`op_kernel/bench_harness_apply_rotary_pos_emb_tiling.h`（TilingData：B/S/N/D/Dh/totalRows/perCoreRows 等）、`op_host/*_tiling_host.cpp`（TilingFunc）、`tests/aclnn/`（ST：run.sh + test_aclnn.cpp，oracle 为独立 torch golden）。
- 构建与测试：`cd operators && ./build.sh BenchHarnessApplyRotaryPosEmb_package`；`cd tests/aclnn && ./run.sh`；UT `build/tests/ut/bench_harness_apply_rotary_pos_emb_tiling_ut`。

## 完整代码

终版 kernel（verbatim 取自会话终版写入与随后两次函数级编辑后的状态；`LoadPlanarHalf`/`LoadCosSinPlanar` 为逐行 1D NDDMA 终版）：

```cpp
#include "kernel_operator.h"
#include "bench_harness_apply_rotary_pos_emb_tiling.h"

template <typename T>
__simd_vf__ inline void MulAddVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::LoadAlign(dstReg, dstAddr, aReg);
        AscendC::Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ inline uint32_t AlignUpU32(uint32_t v, uint32_t f) { return CeilDivU32(v, f) * f; }

template <typename T>
class Branch0Kernel {
    using ComputeT = float;
    static constexpr bool kIsFp32 = std::is_same<T, ComputeT>::value;
    static constexpr uint32_t kVecLen = AscendC::GetVecLen() / sizeof(ComputeT);
    // 32B 对齐因子 (元素): fp32→8, fp16/bf16→16。取 16 覆盖两种情况。
    static constexpr uint32_t kAlignElems = 16;

    AscendC::TPipe pipe_;
    const BenchHarnessApplyRotaryPosEmbTilingData* td_;
    AscendC::GlobalTensor<T> gmQ_, gmK_, gmCos_, gmSin_, gmQout_, gmKout_;
    uint32_t paddedH_;     // alignUp(H, kAlignElems)
    uint32_t myUbRows_;    // 实际 chunk 行数 (按平面布局 UB 预算反推)

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQ1_, bufQ2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCosH_, bufSinH_, bufNegSin_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOut1_, bufOut2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQ1Fp_, bufQ2Fp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCosFp_, bufSinFp_, bufNegSinFp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOut1Fp_, bufOut2Fp_;

public:
    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR queryOut, GM_ADDR keyOut,
        const BenchHarnessApplyRotaryPosEmbTilingData* td)
    {
        td_ = td;
        gmQ_.SetGlobalBuffer((__gm__ T*)query);
        gmK_.SetGlobalBuffer((__gm__ T*)key);
        gmCos_.SetGlobalBuffer((__gm__ T*)cos);
        gmSin_.SetGlobalBuffer((__gm__ T*)sin);
        gmQout_.SetGlobalBuffer((__gm__ T*)queryOut);
        gmKout_.SetGlobalBuffer((__gm__ T*)keyOut);

        const uint32_t H = td_->Dh;
        paddedH_ = AlignUpU32(H, kAlignElems);
        if (paddedH_ < 1) paddedH_ = 1;

        // UB 预算反推 myUbRows (usableUB ≈ 192KB)
        const uint64_t usableUB = 196608ULL;
        uint64_t bytesPerRow;
        if constexpr (kIsFp32) {
            // 7 fp32 planar buffers
            bytesPerRow = 7ULL * paddedH_ * sizeof(ComputeT);
        } else {
            // 6 input-dtype + 7 fp32 planar buffers
            bytesPerRow = 6ULL * paddedH_ * sizeof(T) + 7ULL * paddedH_ * sizeof(ComputeT);
        }
        myUbRows_ = static_cast<uint32_t>(usableUB / bytesPerRow);
        if (myUbRows_ < 1) myUbRows_ = 1;

        const uint32_t planarBytesIn = myUbRows_ * paddedH_ * sizeof(T);
        const uint32_t planarBytesFp = myUbRows_ * paddedH_ * sizeof(ComputeT);

        if constexpr (kIsFp32) {
            pipe_.InitBuffer(bufQ1_, planarBytesFp);
            pipe_.InitBuffer(bufQ2_, planarBytesFp);
            pipe_.InitBuffer(bufCosH_, planarBytesFp);
            pipe_.InitBuffer(bufSinH_, planarBytesFp);
            pipe_.InitBuffer(bufNegSin_, planarBytesFp);
            pipe_.InitBuffer(bufOut1_, planarBytesFp);
            pipe_.InitBuffer(bufOut2_, planarBytesFp);
        } else {
            pipe_.InitBuffer(bufQ1_, planarBytesIn);
            pipe_.InitBuffer(bufQ2_, planarBytesIn);
            pipe_.InitBuffer(bufCosH_, planarBytesIn);
            pipe_.InitBuffer(bufSinH_, planarBytesIn);
            pipe_.InitBuffer(bufOut1_, planarBytesIn);
            pipe_.InitBuffer(bufOut2_, planarBytesIn);
            pipe_.InitBuffer(bufQ1Fp_, planarBytesFp);
            pipe_.InitBuffer(bufQ2Fp_, planarBytesFp);
            pipe_.InitBuffer(bufCosFp_, planarBytesFp);
            pipe_.InitBuffer(bufSinFp_, planarBytesFp);
            pipe_.InitBuffer(bufNegSinFp_, planarBytesFp);
            pipe_.InitBuffer(bufOut1Fp_, planarBytesFp);
            pipe_.InitBuffer(bufOut2Fp_, planarBytesFp);
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockDim = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t perCoreRows = td_->perCoreRows;
        const uint32_t totalRows = td_->totalRows;
        if (totalRows == 0) return;

        uint32_t start, end;
        if (coreId + 1 < blockDim) { start = coreId * perCoreRows; end = start + perCoreRows; }
        else if (coreId + 1 == blockDim) { start = (blockDim - 1) * perCoreRows; end = totalRows; }
        else return;
        if (start >= end) return;

        int32_t evMTE2_V    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evV_MTE2    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        int32_t evV_MTE3    = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        bool isFirst = true;
        for (uint32_t r = start; r < end; r += myUbRows_) {
            uint32_t cnt = myUbRows_;
            if (r + cnt > end) cnt = end - r;
            bool isLast = (r + cnt >= end);
            ProcessChunk(r, cnt, evMTE2_V, evV_MTE2, evV_MTE3, evMTE3_MTE2, isFirst, isLast);
            isFirst = false;
        }
    }

private:
    __aicore__ inline void ProcessChunk(
        uint32_t rowStart, uint32_t cnt,
        int32_t evMTE2_V, int32_t evV_MTE2, int32_t evV_MTE3, int32_t evMTE3_MTE2,
        bool isFirstChunk, bool isLastChunk)
    {
        if (!isFirstChunk) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2);

        // -- MTE2: NDDMA load q1, q2, cosH, sinH --
        LoadPlanarHalf(bufQ1_, gmQ_, rowStart, cnt, 0);
        LoadPlanarHalf(bufQ2_, gmQ_, rowStart, cnt, td_->Dh);
        LoadCosSinPlanar(rowStart, cnt);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2_V);

        // -- V: compute query out1, out2 --
        ComputeHalfOuts(cnt);

        // V 读完 q1/q2 → MTE2 可覆写为 k
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evV_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evV_MTE2);

        // -- V_MTE3 → CopyOut queryOut --
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV_MTE3);
        CopyOutInterleaved(gmQout_, rowStart, cnt);

        // -- MTE2: load k1, k2 (cosH/sinH reused) --
        LoadPlanarHalf(bufQ1_, gmK_, rowStart, cnt, 0);
        LoadPlanarHalf(bufQ2_, gmK_, rowStart, cnt, td_->Dh);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2_V);

        // -- V: compute key out1, out2 --
        ComputeHalfOuts(cnt);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV_MTE3);
        CopyOutInterleaved(gmKout_, rowStart, cnt);

        if (!isLastChunk) AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2);
    }

    __aicore__ inline void LoadPlanarHalf(
        AscendC::TBuf<AscendC::TPosition::VECCALC>& buf,
        AscendC::GlobalTensor<T>& gm, uint32_t rowStart, uint32_t cnt, uint32_t offsetInRow)
    {
        const uint32_t D = td_->D;
        const uint32_t H = td_->Dh;
        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        // 1D NDDMA per row (avoids 2D stride ambiguity for small H)
        AscendC::NdDmaParams<T, 1> params;
        params.loopInfo.loopSrcStride[0] = 1;
        params.loopInfo.loopDstStride[0] = 1;
        params.loopInfo.loopSize[0] = H;
        params.loopInfo.loopLpSize[0] = 0;
        params.loopInfo.loopRpSize[0] = 0;
        params.constantValue = T(0);
        for (uint32_t r = 0; r < cnt; ++r) {
            AscendC::DataCopy<T, 1, cfg>(
                buf.Get<T>()[static_cast<int64_t>(r) * paddedH_],
                gm[static_cast<int64_t>(rowStart + r) * D + offsetInRow], params);
        }
    }

    __aicore__ inline void LoadCosSinPlanar(uint32_t rowStart, uint32_t cnt)
    {
        const uint32_t H = td_->Dh;
        const uint32_t N = td_->N;
        const uint32_t S = td_->S;
        static constexpr AscendC::NdDmaConfig cfg = { false, AscendC::NdDmaConfig::unsetPad,
                                                       AscendC::NdDmaConfig::unsetPad, false };
        AscendC::NdDmaParams<T, 1> params;
        params.loopInfo.loopSrcStride[0] = 1;
        params.loopInfo.loopDstStride[0] = 1;
        params.loopInfo.loopSize[0] = H;
        params.loopInfo.loopLpSize[0] = 0;
        params.loopInfo.loopRpSize[0] = 0;
        params.constantValue = T(0);

        // Per-row: load cos[s(r)] / sin[s(r)] (H elements, broadcast across N)
        for (uint32_t r = 0; r < cnt; ++r) {
            uint32_t absRow = rowStart + r;
            uint32_t s = (absRow / N) % S;
            AscendC::DataCopy<T, 1, cfg>(
                bufCosH_.Get<T>()[static_cast<int64_t>(r) * paddedH_],
                gmCos_[static_cast<int64_t>(s) * H], params);
            AscendC::DataCopy<T, 1, cfg>(
                bufSinH_.Get<T>()[static_cast<int64_t>(r) * paddedH_],
                gmSin_[static_cast<int64_t>(s) * H], params);
        }
    }

    __aicore__ inline void ComputeHalfOuts(uint32_t cnt)
    {
        const uint32_t H = td_->Dh;
        const uint32_t totalElems = cnt * paddedH_;  // padded count (VEC on full padded buffer)

        if constexpr (kIsFp32) {
            auto q1 = bufQ1_.Get<ComputeT>();
            auto q2 = bufQ2_.Get<ComputeT>();
            auto c  = bufCosH_.Get<ComputeT>();
            auto s  = bufSinH_.Get<ComputeT>();
            auto ns = bufNegSin_.Get<ComputeT>();
            auto o1 = bufOut1_.Get<ComputeT>();
            auto o2 = bufOut2_.Get<ComputeT>();

            // padding 为垃圾值但不被 CopyOut，也不影响各行前 H 个有效位。
            AscendC::Muls(ns, s, ComputeT(-1.0), static_cast<int32_t>(totalElems));

            // out1 = q1*c - q2*s = q1*c + q2*(-s)
            AscendC::Mul(o1, q1, c, static_cast<int32_t>(totalElems));
            uint16_t rep = static_cast<uint16_t>(CeilDivU32(totalElems, kVecLen));
            asc_vf_call<MulAddVF<ComputeT>>(
                (__ubuf__ ComputeT*)o1.GetPhyAddr(), (__ubuf__ ComputeT*)q2.GetPhyAddr(),
                (__ubuf__ ComputeT*)ns.GetPhyAddr(), totalElems, kVecLen, rep);

            // out2 = q2*c + q1*s
            AscendC::Mul(o2, q2, c, static_cast<int32_t>(totalElems));
            rep = static_cast<uint16_t>(CeilDivU32(totalElems, kVecLen));
            asc_vf_call<MulAddVF<ComputeT>>(
                (__ubuf__ ComputeT*)o2.GetPhyAddr(), (__ubuf__ ComputeT*)q1.GetPhyAddr(),
                (__ubuf__ ComputeT*)s.GetPhyAddr(), totalElems, kVecLen, rep);
        } else {
            auto q1In = bufQ1_.Get<T>(); auto q2In = bufQ2_.Get<T>();
            auto cIn  = bufCosH_.Get<T>(); auto sIn = bufSinH_.Get<T>();
            auto q1Fp = bufQ1Fp_.Get<ComputeT>(); auto q2Fp = bufQ2Fp_.Get<ComputeT>();
            auto cFp  = bufCosFp_.Get<ComputeT>(); auto sFp  = bufSinFp_.Get<ComputeT>();
            auto nsFp = bufNegSinFp_.Get<ComputeT>();
            auto o1Fp = bufOut1Fp_.Get<ComputeT>(); auto o2Fp = bufOut2Fp_.Get<ComputeT>();

            AscendC::Cast(q1Fp, q1In, AscendC::RoundMode::CAST_NONE, totalElems);
            AscendC::Cast(q2Fp, q2In, AscendC::RoundMode::CAST_NONE, totalElems);
            AscendC::Cast(cFp,  cIn,  AscendC::RoundMode::CAST_NONE, totalElems);
            AscendC::Cast(sFp,  sIn,  AscendC::RoundMode::CAST_NONE, totalElems);

            AscendC::Muls(nsFp, sFp, ComputeT(-1.0), static_cast<int32_t>(totalElems));

            AscendC::Mul(o1Fp, q1Fp, cFp, static_cast<int32_t>(totalElems));
            uint16_t rep = static_cast<uint16_t>(CeilDivU32(totalElems, kVecLen));
            asc_vf_call<MulAddVF<ComputeT>>(
                (__ubuf__ ComputeT*)o1Fp.GetPhyAddr(), (__ubuf__ ComputeT*)q2Fp.GetPhyAddr(),
                (__ubuf__ ComputeT*)nsFp.GetPhyAddr(), totalElems, kVecLen, rep);

            AscendC::Mul(o2Fp, q2Fp, cFp, static_cast<int32_t>(totalElems));
            rep = static_cast<uint16_t>(CeilDivU32(totalElems, kVecLen));
            asc_vf_call<MulAddVF<ComputeT>>(
                (__ubuf__ ComputeT*)o2Fp.GetPhyAddr(), (__ubuf__ ComputeT*)q1Fp.GetPhyAddr(),
                (__ubuf__ ComputeT*)sFp.GetPhyAddr(), totalElems, kVecLen, rep);

            auto o1 = bufOut1_.Get<T>(); auto o2 = bufOut2_.Get<T>();
            AscendC::Cast(o1, o1Fp, AscendC::RoundMode::CAST_RINT, totalElems);
            AscendC::Cast(o2, o2Fp, AscendC::RoundMode::CAST_RINT, totalElems);
        }
    }

    __aicore__ inline void CopyOutInterleaved(
        AscendC::GlobalTensor<T>& gm, uint32_t rowStart, uint32_t cnt)
    {
        const uint32_t D = td_->D;
        const uint32_t H = td_->Dh;
        auto o1 = bufOut1_.Get<T>();
        auto o2 = bufOut2_.Get<T>();
        for (uint32_t row = 0; row < cnt; ++row) {
            uint64_t gmOff = static_cast<uint64_t>(rowStart + row) * D;
            int64_t ubOff = static_cast<int64_t>(row) * paddedH_;  // 32B-aligned (paddedH_ * sizeof(T) ≥ 32)
            AscendC::DataCopyExtParams ext;
            ext.blockCount = 1;
            ext.blockLen = H * sizeof(T);
            ext.srcStride = 0;
            ext.dstStride = 0;
            AscendC::DataCopyPad(gm[static_cast<int64_t>(gmOff)], o1[ubOff], ext);
            AscendC::DataCopyPad(gm[static_cast<int64_t>(gmOff + H)], o2[ubOff], ext);
        }
    }
};

extern "C" __global__ __aicore__ void bench_harness_apply_rotary_pos_emb(
    GM_ADDR query, GM_ADDR key, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR queryOut, GM_ADDR keyOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(BenchHarnessApplyRotaryPosEmbTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.tilingKey == 0u) {
        Branch0Kernel<DTYPE_QUERY> kernel;
        kernel.Init(query, key, cos, sin, queryOut, keyOut, &tilingData);
        kernel.Process();
    } else {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
```

## 验证信号

以下信号来自该算子 Branch-0 kernel 终版的构建、ST、UT 与后续 TTK 白盒全量执行（均为同一次开发会话内的实测）。

- 构建：`./build.sh BenchHarnessApplyRotaryPosEmb_package` 成功（3 dtype 二进制 + 打包完成）（来源平台：950；证据等级：tool_observed）。
- ST：`tests/aclnn` 12/12 全绿——numeric 11/11 PASSED（含 D=2/4/6、B 维广播、大 shape D=128 fp16、NaN/全 0/Inf/常数特殊值）+ 空张量结构 1/1 OK（来源平台：950；证据等级：tool_observed）。
- TilingUT：`bench_harness_apply_rotary_pos_emb_tiling_ut` 45/45 通过（公共 tiling 33 + 分支 12）（来源平台：950；证据等级：tool_observed）。
- 交付检查：`grep -cE '__simd_vf__|asc_vf_call'` 于 op_kernel 7 处命中；oracle（torch golden）独立于被测 kernel（来源平台：950；证据等级：tool_observed）。
- 后续 TTK 白盒 kernel 模式全量 236 条：fp16 72/72 全过、bf16 60/72（12 条极值溢出）、fp32 40/92（fp32 正常范围失配为另一未决问题，见失效边界）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 依赖 cos/sin 的半区 tile 性质与 `s=(absRow/N)%S` 的行→序列位置映射（BSND 行主序）；BNSD/interleaved 布局需重推映射或换分支实现。
- fp32 正常数值范围在 TTK 白盒下有 31 条 mare 超阈（fp16 全过、bf16 仅极值溢出），本配方在 fp32 高精度指标下的正确性存疑、未定论（根因未追，为遗留优先项）。
- 逐行 1D NDDMA 与 padded 布局的性能未测量；UB 预算公式按 usableUB≈192KB 给出，其他平台常量不同（未验证，推断可迁移）。

**未决项**：

- fp32 路径在 TTK 白盒下的 mare 失配根因未定论（优先项：复现并定位）。
- 逐行 1D NDDMA 相对 2D NDDMA 的性能代价未测量。
## 坏实践

- 三个坑的原始写法分别是：interleaved 半区交换用逐行 Muls（VEC 源地址非 32B 对齐 → errcode 340）；未 padding 的逐行 CopyOut（MTE3 非对齐 → errcode 80）；小内维 2D NDDMA（块内错位）——不要按这三条原始路线重写。

## 复现增量：MLA 融合小 Dr 场景的免 Concat rotate_half 元素级构造（bf16，编译级验证）

MLA 前处理融合算子 MlaProlog（Ascend950PR/DAV_3510，bf16，融合 kernel 内 2 处 RoPE，Dr=64）给出与上文平面分解互补的更简形态（适用小 Dr、UB 紧张的融合 kernel）：

**设计核查事实**：AscendC 无 RoPE 专用高阶 API——写 RoPE 前不要找"专用 API"，直接按 `x*cos + rotate_half(x)*sin` 公式用 Mul/Add/Muls/Adds 组合。rotate_half 的两段交换**不要用 Concat**：Concat 需要额外 tmp buffer 且受 shape/对齐约束；两条带偏移的元素级操作即可原位构造。

**rotate_half 免 Concat 构造**（`rotated[0:half] = -x[half:Dr]`、`rotated[half:Dr] = x[0:half]`，verbatim）：

```cpp
        // 3. rotate_half: rotated = [-x[half:], x[:half]]
        // rotated[0:half] = -x[half:Dr]
        Muls<float>(rotated, xFp32[half], -1.0f, half);
        PipeBarrier<PIPE_V>();
        // rotated[half:Dr] = x[0:half] (通过 Adds 0 实现拷贝)
        Adds<float>(rotated[half], xFp32, 0.0f, half);
        PipeBarrier<PIPE_V>();
```

要点：`Adds(dst, src, 0.0f, n)` 是向量 API 无独立带偏移 copy 原语时最轻的等价拷贝写法；全 fp32 中间精度对齐 golden（torch 在 fp32 做 `x*cos + rotate_half(x)*sin` 后舍回 bf16）。

- kernel 编译通过：`[100%] Built target MlaProlog_ascend950`，二进制生成（来源平台：950；证据等级：tool_observed，编译级，未上板）。
- 同公式 CPU golden 自测：sin=0/cos=1 恒等路径 `out=[1,2,3,4]` 期望同值 PASS；sin=1/cos=0 纯 rotate_half 路径 `out=[-3,-4,1,2]` 期望同值 PASS；端到端与 Python golden（含 RoPE 步骤）bf16 位级一致 PASS（来源平台：CPU Mock（x86 host，非 NPU）；证据等级：tool_observed）。

边界：kernel 版仅编译级验证（NPU 数值随算子整体 ST 真机运行闭合）；仅偶数 Dr 验证（Dr=64，奇数维 rotate_half 语义未定义）；Dr 需 ≤ 单个 VECCALC buffer 容量（超大 Dr 需分块，未验证）。与上文平面分解的选择：小 Dr 融合场景用本增量的两条元素级指令（零额外 buffer）；大 H 的独立 RoPE kernel 用上文的平面分解+NDDMA 布局。