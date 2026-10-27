---
type: "Runbook"
title: ChunkedR ARA softmax 三趟 kernel 实现配方
description: 归约轴非末轴且 R 超 UB 的分块在线 softmax（[A0,R,A1] 合轴布局）：tiling 按 per-row 32B 对齐 padded 2D 块预算反解 rUbFactor（含 32KB headroom），kernel 三趟（全局 max / 全局 sum / 归一化）+ UB 向量 carry（VecMaxVf/VecAddVf）消除在线 rescale；DataCopyPad 逐行 -inf 填充、CopyOut 只写有效元素。
tags: [task.implementation, topic.compute.normalization, topic.tiling, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# ChunkedR ARA softmax 三趟 kernel 实现配方

## 现象（任务场景）

一个 softmax 基准算子的分块分支（R 轴超 UB、按 chunk 在线处理）最初只实现了 AR（归约轴=末轴、每行连续）路径；TTK kernel 模式白盒执行 dim=0/1/-2 的 ARA 用例（`((50000, 4),)`、`((4, 50000, 8),)` 等 fp16 大 R）时平均/最大相对误差双超阈。需要为 ARA 布局（`[A0, R, A1]` 合轴：沿 R 归约、A1 列各自独立、每个 a0 块在 GM 里是 `[R, A1]` 连续 2D 块）补完整路径：tiling 预算、搬运、归约、跨 chunk carry、分发，并全量通过白盒与 C++ ST。

## 适用场景

- 工程形态：opbuild/算子仓工程 kernel+tiling 联调（`op_kernel/*.h` + `op_host/*_tiling_host.cpp`），或 kernel 直调复现。
- API 路径：`DataCopyPad`（`DataCopyExtParams.blockCount` 逐行 2D 块搬运 + `DataCopyPadExtParams` 右填充）、`AscendC::ReduceMax/ReduceSum`（`Pattern::Reduce::RA`，`srcShape` + `srcInnerPad=false`）、`asc_vf_call` + `__simd_vf__` 向量函数（Cast/Sub+Exp/Div+Cast/Max/Add/Copy）、`TPipe InitBuffer`。
- 验证过的平台与 CANN 版本：Ascend 950 系列（SOC Ascend950PR）+ CANN 9.1.0（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`operator_family`（ARA 合轴布局 + 分块在线归约的模式对同类归约算子一致）。
- 路线优先级：`recommended_fastpath`。
- 明确不适用/未验证场景：AR（末轴）路径（另有关联实现）；A1 大到 per-row padding 开销显著时的预算需重算（本例 A1=4/8）；非 max-subtraction 语义算子不能用 -inf 填充。

## 设计决策

1. **三趟而不是在线两趟**：在线两趟（Pass1 融合 max+sum，`s *= exp(m_old - m_new)` 跨 chunk 重缩放；Pass2 归一化）读 x 仅 2 次，但每次 rescale 引入 fp32 舍入并按 chunk 数累积，是 ARA 平均误差比 AR 路径差约两个数量级的嫌疑机制（该机制为开发期分析，见"失效边界"的诚实标注）。三趟（Pass1 全局 max、Pass2 全局 sum、Pass3 归一化）消除 rescale，代价是 x 读 3 次。AR 路径同期同样改为三趟，两路径算法同构。
2. **carry 用 UB 向量 + 元素级 VF**：A1 列各自的 m/s 是 `aBundleAlignF32_` 元素的 UB 向量（carryBuf 前半 m、后半 s）；`ReduceMax/ReduceSum(RA)` 输出落 maxBuf/sumBuf 后，经 `VecMaxVf`/`VecAddVf` 做 V→V 元素级 carry；跨趟用 `VecCopyVf` 把最终 m/s 复制回 maxBuf/sumBuf 供广播使用。Reduce API 内部含 V→S 同步，链路上无需显式 PipeBarrier（实测删除无影响）。
3. **per-row 32B 对齐 padded 布局**：A1×dtype 非 32B 对齐时向量/Reduce 按未对齐行宽会读错布局（同工程更早分支已有"输出全 0"教训，另见「tiling UB 预算不留 headroom 输出全 0」卡），UB 内统一 padded 行 stride（D_T 行 `aBundleAlignD_`、fp32 行 `aBundleAlignF32_`），`DataCopyPad` 以 -inf 填充尾（max 语义下安全），CopyOut 逐行只写有效 A1 元素回 dense GM。
4. **tiling 预算含 32KB headroom**：`avail = ubAvailable - fixedCost - 32KB`，不留余量会把 UB 占满并静默输出全 0（见「tiling UB 预算不留 headroom 输出全 0」卡）。
5. **buffer 尺寸 CeilAlign 到 64（kRepF32）**：VF 函数按 64 元素一轮，尾轮 masked 写越界会污染相邻 buffer。
6. **放弃的候选**：标量 C++ 循环 carry（设备编译器不提供 std::max/std::exp/expf 等标准数学函数，需自实现标量版，见「kernel 无 libc 数学函数需自实现」卡）；在线两趟 ARA（精度嫌疑）；显式 PipeBarrier 同步（对照证明非必需）。

## 系统信息

- 工具链：CANN 9.1.0（bisheng clang 15.0.5 设备编译 + ld.lld），opbuild 工程模板（`operators/<pkg>/{CMakeLists.txt, build.sh, op_host/, op_kernel/, tests/}`）。
- 目标平台：Ascend 950 系列（SOC Ascend950PR）。
- 目录布局与相关文件：`op_kernel/bench_harness_softmax.h`（ARA 数据结构/Init/搬运/三趟主流程/VF 函数）、`op_kernel/bench_harness_softmax_kernel.cpp`（入口按 TilingData 运行时分发）、`op_kernel/bench_harness_softmax_tiling.h`（TilingData 结构）、`op_host/bench_harness_softmax_tiling_host.{h,cpp}`（`ChunkedRInputs`（含 aBundle）与 `ComputeChunkedRTiling` ARA 分支）、`tests/aclnn/`（C++ ST 复现用例）、`tests/whitebox/`（TTK 白盒 CSV 与 result.csv）。

## 完整代码

以下代码 verbatim 取自本次实现的最终版本（tiling 预算行为含 headroom 修复后的形态）。

**1. tiling host：`ChunkedRInputs` 增加 aBundle，`ComputeChunkedRTiling` ARA 分支：**

```cpp
struct ChunkedRInputs {
    int64_t rTotal;        // R 轴总长（valid）
    int64_t ubAvailable;   // UB 可用预算（字节）
    int32_t elemBytes;     // sizeof(D_T)
    int64_t blockSize;     // UB 搬运对齐粒度
    int64_t aBundle;       // A1（ARA 内层 A 轴 size）；AR = 1
};

bool ComputeChunkedRTiling(const ChunkedRInputs& in, BenchHarnessSoftmaxTilingData& td)
{
    int64_t alignFactor = in.blockSize / in.elemBytes;  // 32B 对齐粒度（元素数）：fp16/bf16=16, fp32=8
    int64_t rUbFactor = 0;

    if (in.aBundle <= 1) {
        // AR pattern：预算反解（3 份 D_T + 3 份 fp32 + 16KB），与 ComputeRFullloadThresh 同公式
        int64_t rImax = ComputeRFullloadThresh(in.ubAvailable, in.elemBytes, in.blockSize);
        int64_t cap = std::min(rImax, in.rTotal);
        rUbFactor = FloorAlign(cap, alignFactor);
    } else {
        // ARA pattern：per-chunk [rChunkLen, A1] 2D 块，per-row padded 到 32B。
        //   预算：2×rF×aAlignD×d (xBuf DB=2) + rF×aAlignF32×4 (xF32Buf) + rF×aAlignF32×4 (eBuf)
        //         + rF×aAlignD×d (yBuf) + 2×CeilAlign(aAlignF32×4, bs) (max+sum) + 16KB ≤ ubAvailable
        int64_t aAlignD = CeilAlign(in.aBundle, alignFactor);           // D_T 行 stride（padded）
        int64_t aAlignF32 = CeilAlign(in.aBundle, in.blockSize / FP32_BYTES); // fp32 行 stride（padded）
        int64_t perElemCost = 2 * aAlignD * in.elemBytes                 // xBuf DB=2 per elem
                            + 2 * aAlignF32 * FP32_BYTES;               // xF32Buf + eBuf per elem
        perElemCost += aAlignD * in.elemBytes;                           // yBuf per elem
        int64_t fixedCost = 2 * CeilAlign(aAlignF32 * FP32_BYTES, in.blockSize)  // max + sum
                          + CACHE_BUF_16KB;
        int64_t avail = in.ubAvailable - fixedCost - 32 * 1024; // 32KB stack/overhead headroom
        if (avail <= 0) return false;
        int64_t rImax = avail / perElemCost;  // 反解最大 rUbFactor
        int64_t cap = std::min(rImax, in.rTotal);
        rUbFactor = FloorAlign(cap, alignFactor);
    }

    if (rUbFactor == 0) {
        return false; // TILING_FAIL：连一个 block 都装不下
    }

    td.rUbFactor = rUbFactor;
    td.rUbFactorAlign = CeilAlign(rUbFactor, alignFactor);
    td.rChunkCnt = CeilDiv(in.rTotal, rUbFactor);
    return true;
}
```

**2. host 胶水：axisNum→aBundle 传递：**

```cpp
    } else {
        // ChunkedR：据 axisNum 判定 AR(2)/ARA(3)，ARA 时 aBundle=A1=axisShape[2]
        int64_t aBundle = (fp.axisNum == 3) ? fp.axisShape[2] : 1;
        ChunkedRInputs in{fp.rTotal, ubAvailable, elemBytes, BLOCK_SIZE, aBundle};
        if (!ComputeChunkedRTiling(in, *td)) {
            return ge::GRAPH_FAILED; // rUbFactor==0 FAIL 守卫（连一个 block 都装不下）
        }
    }
```

**3. kernel `Init`：pattern 判定 + ARA buffer 尺寸（64 元素对齐防尾轮越界）：**

```cpp
    // pattern 判定：axisNum==2 → AR，axisNum==3 → ARA
    axisNum_ = td->axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        axisShape_[i] = td->axisShape[i];
        axisStride_[i] = td->axisStride[i];
    }
    isAr_ = (axisNum_ == 2);
    if (isAr_) {
        aBundle_ = 1;
        aBundleAlignD_ = 1;
        aBundleAlignF32_ = 1;
        aTotalOuter_ = aTotal_;
    } else {
        aBundle_ = axisShape_[2]; // A1
        aBundleAlignD_ = CeilAlignKernel(aBundle_, bsElemD_);
        if (aBundleAlignD_ < bsElemD_) aBundleAlignD_ = bsElemD_;
        aBundleAlignF32_ = CeilAlignKernel(aBundle_, static_cast<int64_t>(kBlockBytes) / 4);
        if (aBundleAlignF32_ < static_cast<int64_t>(kBlockBytes) / 4)
            aBundleAlignF32_ = static_cast<int64_t>(kBlockBytes) / 4;
        aTotalOuter_ = aTotal_ / aBundle_; // A0
    }

    // GM
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(y));

    // UB buffer 尺寸
    // VF 函数以 kRepF32(64) 元素为一轮处理，buffer 须 CeilAlign 到 64 防止尾轮越界写入
    // 污染相邻 buffer（尤其 eBuf → carryBuf）。
    if (isAr_) {
        const int64_t rAlign64 = CeilAlignKernel(rUbFactorAlign_, static_cast<int64_t>(kRepF32));
        const int64_t xBufBytes = rAlign64 * elemBytes_;
        const int64_t f32BufBytes = rAlign64 * 4;
        const int64_t reduceBufBytes = static_cast<int64_t>(kBlockBytes);

        pipe_.InitBuffer(xQue_, /*bufNum=*/2, xBufBytes);
        pipe_.InitBuffer(yQue_, /*bufNum=*/1, xBufBytes);
        pipe_.InitBuffer(xF32Buf_, f32BufBytes);
        pipe_.InitBuffer(eBuf_, f32BufBytes);
        pipe_.InitBuffer(maxBuf_, reduceBufBytes);
        pipe_.InitBuffer(sumBuf_, reduceBufBytes);
    } else {
        // ARA：per-chunk [rChunkLen, A1] 2D 块，per-row padded 到 32B
        const int64_t rAlign64 = CeilAlignKernel(rUbFactorAlign_, static_cast<int64_t>(kRepF32));
        // D_T buffer: rChunkLen rows × aBundleAlignD_ per row
        const int64_t xRowBytes = aBundleAlignD_ * elemBytes_;
        const int64_t xBufBytes = rAlign64 * xRowBytes;
        // fp32 buffer: rChunkLen rows × aBundleAlignF32_ per row
        const int64_t f32RowBytes = aBundleAlignF32_ * 4;
        const int64_t f32BufBytes = rAlign64 * f32RowBytes;
        // reduce output: A1_align fp32 (per-column)
        const int64_t reduceBufBytes = aBundleAlignF32_ * 4;

        pipe_.InitBuffer(xQue_, /*bufNum=*/2, xBufBytes);
        pipe_.InitBuffer(yQue_, /*bufNum=*/1, xBufBytes);
        pipe_.InitBuffer(xF32Buf_, f32BufBytes);
        pipe_.InitBuffer(eBuf_, f32BufBytes);
        pipe_.InitBuffer(maxBuf_, reduceBufBytes);
        pipe_.InitBuffer(sumBuf_, reduceBufBytes);
    }
    pipe_.InitBuffer(carryBuf_, td->cacheBufUbSize); // s carry（16KB）
```

**4. 2D 块搬运（CopyIn 逐行 -inf 填充、CopyOut 只写有效元素）：**

```cpp
// ════════════════════════════════════════════════════════════════════════════
// CopyInChunkAra：GM [rChunkLen, A1] 连续 2D 块 → UB（per-row pad 到 32B）
//   GM 布局：[R, A1] row-major，每行 A1 个 D_T 元素连续。
//   UB 布局：每行 aBundleAlignD_ 元素（32B padded），-inf pad 覆盖尾。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxChunkedKernel<DType>::CopyInChunkAra(
    int64_t gmOff, int64_t rChunkLen, LocalTensor<D_T>& dst)
{
    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(aBundle_ * elemBytes_);
    ext.blockCount = static_cast<uint16_t>(rChunkLen);
    ext.srcStride = 0;  // GM dense（行间无 gap）
    ext.dstStride = 0;  // UB padded block = 行 stride（pad 后无额外 gap）
    ext.rsv = 0;
    uint32_t misalign = ext.blockLen & (kBlockBytes - 1u);
    uint8_t rPad = (misalign == 0) ? 0
        : static_cast<uint8_t>((kBlockBytes - misalign) / elemBytes_);
    DataCopyPadExtParams<D_T> padParams{/*isPad=*/true, /*leftPadding=*/0, rPad, NegInfD<D_T>()};
    DataCopyPad(dst, xGm_[gmOff], ext, padParams);
}

// ════════════════════════════════════════════════════════════════════════════
// CopyOutChunkAra：UB [rChunkLen, A1] padded → GM dense（仅写有效 A1 元素/行）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxChunkedKernel<DType>::CopyOutChunkAra(
    int64_t gmOff, int64_t rChunkLen, LocalTensor<D_T>& src)
{
    for (int64_t r = 0; r < rChunkLen; ++r) {
        int64_t gmRowOff = gmOff + r * aBundle_;
        int64_t ubRowOff = r * aBundleAlignD_;
        DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(aBundle_ * elemBytes_);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        DataCopyPad(yGm_[gmRowOff], src[ubRowOff], ext);
    }
}
```

**5. carry 用 VF 函数（元素级 max / add / copy）：**

```cpp
// VF12：向量元素级 max（m = max(m, mChunk)），用于 ARA 三趟 Pass1 carry m。
//   不做 rescale（区别于 VecMaxRescaleVf），消除在线 s*=exp(delta) 累积误差。
__simd_vf__ inline void VecMaxVf(
    __ubuf__ float* mAddr, __ubuf__ float* mChunkAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> mReg, mChunkReg, mNewReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(mReg, mAddr + off);
        Reg::LoadAlign(mChunkReg, mChunkAddr + off);
        Reg::Max(mNewReg, mReg, mChunkReg, mask);
        Reg::StoreAlign(mAddr + off, mNewReg, mask);
    }
}

// VF10：ARA 向量累加 s = s + addend（element-wise，addend 为 UB 向量）。
__simd_vf__ inline void VecAddVf(
    __ubuf__ float* sAddr, __ubuf__ float* addendAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> sReg, addReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(sReg, sAddr + off);
        Reg::LoadAlign(addReg, addendAddr + off);
        Reg::Add(sReg, sReg, addReg, mask);
        Reg::StoreAlign(sAddr + off, sReg, mask);
    }
}

// VF11：向量复制（dst = src，用 Max(x,x)=x 实现）。
__simd_vf__ inline void VecCopyVf(
    __ubuf__ float* dstAddr, __ubuf__ float* srcAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> srcReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(srcReg, srcAddr + off);
        Reg::StoreAlign(dstAddr + off, srcReg, mask);
    }
}
```

**6. 三趟主流程 `ProcessOneBlockAra`（最终版，完整）：**

```cpp
// ════════════════════════════════════════════════════════════════════════════
// ProcessOneBlockAra：单 a0 块 ARA ChunkedR softmax（[A0, R, A1] pattern）
//   每个 a0 块 = [R, A1] 连续 GM 块，沿 R 归约，A1 列各自独立。
//   三趟算法（与 ProcessOneRow 同构，消除在线 rescale 累积误差）：
//     Pass1（读 x 1 次）：跨 chunk carry 全局 max m[A1]（ReduceMax + VecMaxVf）
//     Pass2（读 x 1 次）：exp(x-m)，ReduceSum，VecAddVf 累加 s[A1]
//     Pass3（读 x 1 次）：exp(x-m)/s，Cast D_T，CopyOut
//   x 共读 3 次。m/s 为 UB 向量（carryBuf），VecMaxVf/VecAddVf 经 asc_vf_call
//   完成 V→V 元素级 carry，Reduce API 内部 V→S 同步后 GetValue 安全。
//   carry 布局：carryBuf[0..A1F-1]=m，carryBuf[A1F..2A1F-1]=s（A1F=aBundleAlignF32_）。
//   特殊值传播：NaN 经 exp(NaN-m)=NaN → s=NaN → y=NaN 全切片传播（同 AR 路径）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxChunkedKernel<DType>::ProcessOneBlockAra(int64_t a0Idx)
{
    const int64_t blockGmOff = a0Idx * rTotal_ * aBundle_;

    // carry buffer 布局：m 在 [0, A1F)，s 在 [A1F, 2*A1F)
    auto carry = carryBuf_.Get<float>();
    __ubuf__ float* carryPtr = reinterpret_cast<__ubuf__ float*>(carry.GetPhyAddr());
    __ubuf__ float* mPtr = carryPtr;
    __ubuf__ float* sPtr = carryPtr + aBundleAlignF32_;
    const uint32_t carryCount = static_cast<uint32_t>(aBundleAlignF32_);
    const uint16_t carryReps = static_cast<uint16_t>(CeilDivKernel(carryCount, kRepF32));
    const uint32_t cnt = carryCount;
    const uint16_t reps = carryReps;

    // ════════ Pass 1：全局 max（跨 chunk carry m[A1] 向量，读 x 1 次）════════
    asc_vf_call<DuplicateVf>(mPtr, -std::numeric_limits<float>::infinity(),
                             carryCount, kRepF32, carryReps);

    for (int64_t c = 0; c < rChunkCnt_; ++c) {
        int64_t chunkOff = c * rUbFactor_;
        int64_t rChunkLen = (c == rChunkCnt_ - 1) ? (rTotal_ - chunkOff) : rUbFactor_;
        int64_t gmChunkOff = blockGmOff + chunkOff * aBundle_;

        // ── CopyIn（MTE2）──
        auto xLocal = xQue_.template AllocTensor<D_T>();
        CopyInChunkAra(gmChunkOff, rChunkLen, xLocal);
        xQue_.EnQue(xLocal);
        auto xDeq = xQue_.template DeQue<D_T>();

        // ── Cast D_T → fp32（per-row padded）──
        auto xF32 = xF32Buf_.Get<float>();
        __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xDeq.GetPhyAddr());
        __ubuf__ float* xF32Ptr = reinterpret_cast<__ubuf__ float*>(xF32.GetPhyAddr());
        for (int64_t r = 0; r < rChunkLen; ++r) {
            int64_t srcOff = r * aBundleAlignD_;
            int64_t dstOff = r * aBundleAlignF32_;
            asc_vf_call<CastInVf<D_T>>(xF32Ptr + dstOff, xPtr + srcOff, cnt, kRepF32, reps);
        }
        xQue_.FreeTensor(xDeq);

        // ── ReduceMax（Pattern::RA → A1 个 max）──
        auto maxDst = maxBuf_.Get<float>();
        __ubuf__ float* maxPtr = reinterpret_cast<__ubuf__ float*>(maxDst.GetPhyAddr());
        {
            uint32_t srcShape[2] = {static_cast<uint32_t>(rChunkLen),
                                    static_cast<uint32_t>(aBundleAlignF32_)};
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(
                maxDst, xF32, srcShape, /*srcInnerPad=*/false);
        }

        // ── m = max(m, mChunk)（元素级，无 rescale）──
        asc_vf_call<VecMaxVf>(mPtr, maxPtr, carryCount, kRepF32, carryReps);
    }

    // 复制 carry m → maxBuf 供 Pass 2/3 的 SubExpVfRow 使用
    auto maxDst2 = maxBuf_.Get<float>();
    __ubuf__ float* maxBufPtr = reinterpret_cast<__ubuf__ float*>(maxDst2.GetPhyAddr());
    asc_vf_call<VecCopyVf>(maxBufPtr, mPtr, carryCount, kRepF32, carryReps);

    // ════════ Pass 2：全局 sum（exp(x-m)，ReduceSum，VecAddVf 累加 s[A1]，读 x 1 次）════════
    asc_vf_call<DuplicateVf>(sPtr, 0.0f, carryCount, kRepF32, carryReps);

    for (int64_t c = 0; c < rChunkCnt_; ++c) {
        int64_t chunkOff = c * rUbFactor_;
        int64_t rChunkLen = (c == rChunkCnt_ - 1) ? (rTotal_ - chunkOff) : rUbFactor_;
        int64_t gmChunkOff = blockGmOff + chunkOff * aBundle_;

        // ── CopyIn ──
        auto xLocal = xQue_.template AllocTensor<D_T>();
        CopyInChunkAra(gmChunkOff, rChunkLen, xLocal);
        xQue_.EnQue(xLocal);
        auto xDeq = xQue_.template DeQue<D_T>();

        // ── Cast ──
        auto xF32 = xF32Buf_.Get<float>();
        __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xDeq.GetPhyAddr());
        __ubuf__ float* xF32Ptr = reinterpret_cast<__ubuf__ float*>(xF32.GetPhyAddr());
        for (int64_t r = 0; r < rChunkLen; ++r) {
            int64_t srcOff = r * aBundleAlignD_;
            int64_t dstOff = r * aBundleAlignF32_;
            asc_vf_call<CastInVf<D_T>>(xF32Ptr + dstOff, xPtr + srcOff, cnt, kRepF32, reps);
        }
        xQue_.FreeTensor(xDeq);

        // ── Sub + Exp（per-row，最终 m 向量广播）──
        auto eDst = eBuf_.Get<float>();
        __ubuf__ float* ePtr = reinterpret_cast<__ubuf__ float*>(eDst.GetPhyAddr());
        for (int64_t r = 0; r < rChunkLen; ++r) {
            int64_t off = r * aBundleAlignF32_;
            asc_vf_call<SubExpVfRow<D_T>>(ePtr + off, xF32Ptr + off, maxBufPtr,
                                          cnt, kRepF32, reps);
        }

        // ── ReduceSum（Pattern::RA → A1 个 sum）──
        auto sumDst = sumBuf_.Get<float>();
        __ubuf__ float* sumPtr = reinterpret_cast<__ubuf__ float*>(sumDst.GetPhyAddr());
        {
            uint32_t srcShape[2] = {static_cast<uint32_t>(rChunkLen),
                                    static_cast<uint32_t>(aBundleAlignF32_)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(
                sumDst, eDst, srcShape, /*srcInnerPad=*/false);
        }

        // ── s += sumChunk（ReduceSum 已同步，VecAddVf 读 sumBuf 安全）──
        asc_vf_call<VecAddVf>(sPtr, sumPtr, carryCount, kRepF32, carryReps);
    }

    // 复制 carry s → sumBuf 供 Pass 3 的 DivCastVfRow 使用
    auto sumDst2 = sumBuf_.Get<float>();
    __ubuf__ float* sumBufPtr = reinterpret_cast<__ubuf__ float*>(sumDst2.GetPhyAddr());
    asc_vf_call<VecCopyVf>(sumBufPtr, sPtr, carryCount, kRepF32, carryReps);

    // ════════ Pass 3：归一化（exp(x-m)/s，Cast，CopyOut，读 x 1 次）════════
    for (int64_t c = 0; c < rChunkCnt_; ++c) {
        int64_t chunkOff = c * rUbFactor_;
        int64_t rChunkLen = (c == rChunkCnt_ - 1) ? (rTotal_ - chunkOff) : rUbFactor_;
        int64_t gmChunkOff = blockGmOff + chunkOff * aBundle_;

        // ── CopyIn ──
        auto xLocal = xQue_.template AllocTensor<D_T>();
        CopyInChunkAra(gmChunkOff, rChunkLen, xLocal);
        xQue_.EnQue(xLocal);
        auto xDeq = xQue_.template DeQue<D_T>();

        // ── Cast ──
        auto xF32 = xF32Buf_.Get<float>();
        __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xDeq.GetPhyAddr());
        __ubuf__ float* xF32Ptr = reinterpret_cast<__ubuf__ float*>(xF32.GetPhyAddr());
        for (int64_t r = 0; r < rChunkLen; ++r) {
            int64_t srcOff = r * aBundleAlignD_;
            int64_t dstOff = r * aBundleAlignF32_;
            asc_vf_call<CastInVf<D_T>>(xF32Ptr + dstOff, xPtr + srcOff, cnt, kRepF32, reps);
        }
        xQue_.FreeTensor(xDeq);

        // ── Sub + Exp（per-row，最终 m 向量广播）──
        auto eDst = eBuf_.Get<float>();
        __ubuf__ float* ePtr = reinterpret_cast<__ubuf__ float*>(eDst.GetPhyAddr());
        for (int64_t r = 0; r < rChunkLen; ++r) {
            int64_t off = r * aBundleAlignF32_;
            asc_vf_call<SubExpVfRow<D_T>>(ePtr + off, xF32Ptr + off, maxBufPtr,
                                          cnt, kRepF32, reps);
        }

        // ── Div + Cast（per-row，向量 s 广播）──
        auto yLocal = yQue_.template AllocTensor<D_T>();
        __ubuf__ D_T* yPtr = reinterpret_cast<__ubuf__ D_T*>(yLocal.GetPhyAddr());
        for (int64_t r = 0; r < rChunkLen; ++r) {
            int64_t eOff = r * aBundleAlignF32_;
            int64_t yOff = r * aBundleAlignD_;
            asc_vf_call<DivCastVfRow<D_T>>(yPtr + yOff, ePtr + eOff, sumBufPtr,
                                           cnt, kRepF32, reps);
        }

        // ── CopyOut ──
        yQue_.EnQue(yLocal);
        auto yDeq = yQue_.template DeQue<D_T>();
        CopyOutChunkAra(gmChunkOff, rChunkLen, yDeq);
        yQue_.FreeTensor(yDeq);
    }
}
```

**7. `Process` 按 pattern 分发（ARA 按 a0 块、AR 按行）：**

```cpp
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxChunkedKernel<DType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
        return; // idle core
    }
    if (aTotal_ <= 0) {
        return;
    }
    // ARA 按 a0 块（aTotalOuter_ = A0）分发；AR 按行（aTotal_）分发
    const int64_t totalBlocks = isAr_ ? aTotal_ : aTotalOuter_;
    if (totalBlocks <= 0) {
        return;
    }
    const int64_t coreNum = static_cast<int64_t>(usedCoreNum_);
    int64_t small = totalBlocks / coreNum;            // floor
    int64_t bigCnt = totalBlocks % coreNum;
    int64_t big = small + 1;
    int64_t start, end;
    if (blockIdx < bigCnt) {
        start = blockIdx * big;
        end = start + big;
    } else {
        start = bigCnt * big + (blockIdx - bigCnt) * small;
        end = start + small;
    }
    for (int64_t a = start; a < end; ++a) {
        if (isAr_) {
            ProcessOneRow(a);
        } else {
            ProcessOneBlockAra(a);
        }
    }
}
```

## 预防

- 新增合轴路径时先落 tiling 预算（含 headroom）再写 kernel 主体，避免"贴满 UB 静默全 0"。
- C++ ST 同时挂语义不变量（sum_to_one）与口径一致的 mare/mere 复现用例。

## 验证信号

以下信号来自同一次开发会话的构建/执行/独立验收记录：

- ARA 复现用例从 `sum_to_one=0.000000`（实现初期 UB 预算未留 headroom）→ 修复预算后输出非 0，mare/mere 进入 1e-6/1e-3 量级（WB_125 mare=4.5011e-06 mere=9.1743e-03）（来源平台：950；证据等级：tool_observed）。
- 构建通过后 C++ ST：既有分支 32/32 + 本分支 17/17（含 8 条白盒失败复现用例）（来源平台：950；证据等级：tool_observed）。
- TTK kernel 模式白盒（`--compare close` + spec 逐 dtype atol）：8 条原失败用例全 PASS，全量 130 条 `total=130 pass=130 fail=0 skip=0`（来源平台：950；证据等级：tool_observed）。
- 独立验收复跑全量 130/130，与开发结果逐用例一致（REPRODUCIBLE）；C++ ST 经 run.sh 全量重建复跑 exit=0 无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- A1 维度大（per-row padding 开销显著）时预算公式需重新推导，本例仅验证 A1=4/8。
- 三趟读 x 3 次的带宽代价未单独量化（无对照 profiling 数据）；带宽受限场景在线两趟可能更优，但需另行验证精度代价。
- **诚实标注**：三趟对 ARA 平均误差的独立改善量未隔离测量——本会话中三趟重写后 C++ ST 的 mare/mere 数值与在线两趟完全相同（WB_125 mare=4.5011e-06 mere=9.1743e-03 不变，C++ f32 golden 口径对该算法差不敏感），最终白盒全绿由"kernel 修复 + 比对口径对齐 spec（stat_rel_err 分母口径）"共同达成；"在线两趟 rescale 误差按 chunk 数累积"是开发期机制分析（agent 推断），无独立 before/after 信号。
- fp16 之外的 dtype（fp32/bf16）在 ARA 路径上经全量白盒通过（38+44 条），但未做逐 dtype 误差分解。

**未决项**：

- 三趟相对在线两趟的精度增益需补一次隔离测量（同口径 stat_rel_err 对照跑），当前只有算法层分析。
- 三趟的带宽代价（x 读 3 次）无 profiling 对照，性能敏感算子采用前应实测。
## 坏实践

- 在线两趟的 `s *= exp(m_old - m_new)` 标量 rescale：设备侧无标量数学库（编译失败），向量化后又引入跨 chunk 累积舍入嫌疑——三趟直接消除该算子。
- ARA 布局不做 per-row 32B 对齐：向量/Reduce 按未对齐行宽读错布局，输出全 0（同工程更早分支已踩）。
