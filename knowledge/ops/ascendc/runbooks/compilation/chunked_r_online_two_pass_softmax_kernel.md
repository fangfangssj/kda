---
type: "Runbook"
title: "ChunkedR 在线两趟 softmax kernel 实现"
description: "大 R 轴分 chunk 的在线两趟 softmax 完整 regbase kernel 配方（AR 末轴布局）：Pass1 跨 chunk 在线 carry m/s（s*=exp(m_old-m_new) 重缩放，carryBuf + RescaleSumVf/AddScalarVf），Pass2 重读 x 归一化，读 x 仅 2 次；s 提取用 ReduceSum 聚合镜像（V→S 同步），分支分发用 TilingData 字段而非 tilingKey。950 实测 ST 9/9 + 回归 32/32、3 次运行确定。"
tags: [task.implementation, topic.compute.normalization, topic.compute.reduction, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# ChunkedR 在线两趟 softmax kernel 实现

## 现象（任务场景）

在 Ascend 950（regbase 范式，CANN 9.1.0）开发 softmax 类基准算子时遇到的任务形态：归约切片（一行）的 R 轴长度超出单个 chunk 的 UB 预算（`rTotal > rFullloadThresh`，如 R=50272 或 2097152），整行装不进 UB，必须沿 R 分 `rChunkCnt > 1` 个 chunk 串行处理，且设计要求**在线两趟**（online softmax）：Pass1 在线融合 max+sum 跨 chunk carry 标量 `m`/`s`，Pass2 重读 x 归一化，共读 x 2 次、e 不落 GM。首次实现走了两次弯路（朴素三趟数值对但读 x 3 次被验收打回；在线两趟 carry 取值缺同步致多行 sum>1，修复见 [VF 归约落 UB 后 GetValue 读取陈旧标量卡](../precision/vector_scalar_ub_handoff_requires_explicit_completion.md)），最终落地本卡配方，ST 9/9（含 8192 行多核、226 chunk 单行、NaN/±inf 大 R 特殊值）+ 全载分支回归 32/32，3 次运行确定。

## 适用场景

- 工程形态：Ascend C regbase kernel（`__simd_vf__` + `asc_vf_call` + `Reg::` 寄存器 API + `AscendC::Reduce*`），单输入单输出、沿一维归约的 normalization 类算子（softmax/相关 max-shift 归约变体）。
- API 路径：`TQue<VECIN/VECOUT>` DB 流水、`TBuf<VECCALC>` 暂存、`DataCopyPad` 尾块 -inf padding、`AscendC::ReduceMax/ReduceSum<Pattern::Reduce::AR>`、`asc_vf_call` 调 `__simd_vf__` 函数。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`（大 R 分 chunk 在线归约的通用骨架；UB 预算不等式按 dtype 代入）。
- 路线优先级：`recommended_fastpath`（满足"读 x 2 次 + 在线 carry"设计口径的最短路径；朴素三趟是 safety_fallback——多约 50% x 读，来源平台 950）。
- 明确不适用/未验证场景：ARA pattern（R 在中间轴）未实现（本配方只覆盖 AR，axisNum=2——ARA 布局的三趟实现与取舍另见 [ChunkedR ARA softmax 三趟 kernel 实现配方卡](chunkedr_ara_softmax_three_pass_recipe.md)）；跨核切 R（本配方 R chunk 单核串行，满足确定性约束）；`ASCENDC_TPL_SEL` 编译期分支工程未用。

## 完整代码

元素级 VF 函数（carry 三件套 + 共用的 Sub+Exp / Div+Cast）：

```cpp
constexpr uint32_t kVlBytes = 256;                     // VL = 256B
constexpr uint32_t kRepF32 = kVlBytes / sizeof(float); // = 64（fp32 单次循环元素数）
constexpr uint32_t kBlockBytes = 32;                   // UB 32B 对齐粒度

// Duplicate（标量广播写）。carry buffer 初始化（s=0）。
__simd_vf__ inline void DuplicateVf(
    __ubuf__ float* dstAddr, float val,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> valReg;
    Reg::MaskReg mask;
    Reg::Duplicate(valReg, val);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::StoreAlign(dstAddr + off, valReg, mask);
    }
}

// 在线重缩放 s = s * exp(delta)（delta = m_old - m_new ∈ (-∞,0]，标量）。
__simd_vf__ inline void RescaleSumVf(
    __ubuf__ float* sAddr, float deltaScalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> sReg, deltaReg, scaleReg;
    Reg::MaskReg mask;
    Reg::Duplicate(deltaReg, deltaScalar);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(sReg, sAddr + off);
        Reg::Exp(scaleReg, deltaReg, mask);
        Reg::Mul(sReg, sReg, scaleReg, mask);
        Reg::StoreAlign(sAddr + off, sReg, mask);
    }
}

// 标量累加 s = s + addend（每 chunk 的 ReduceSum 结果折入 carry）。
__simd_vf__ inline void AddScalarVf(
    __ubuf__ float* sAddr, float addend,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> sReg, addReg;
    Reg::MaskReg mask;
    Reg::Duplicate(addReg, addend);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(sReg, sAddr + off);
        Reg::Add(sReg, sReg, addReg, mask);
        Reg::StoreAlign(sAddr + off, sReg, mask);
    }
}

// Sub + Exp（AR 扁平，标量 max 广播）：e = exp(x - max)。
template <typename D_T>
__simd_vf__ inline void SubExpVfAr(
    __ubuf__ float* eAddr, __ubuf__ float* xAddr, float maxScalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> xReg, eReg, maxReg;
    Reg::MaskReg mask;
    Reg::Duplicate(maxReg, maxScalar);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(xReg, xAddr + off);
        Reg::Sub(xReg, xReg, maxReg, mask);
        Reg::Exp(eReg, xReg, mask);
        Reg::StoreAlign(eAddr + off, eReg, mask);
    }
}

// Div + Cast（AR 扁平，标量 s 广播）：y = (e / s) cast D_T。
template <typename D_T>
__simd_vf__ inline void DivCastVfAr(
    __ubuf__ D_T* yAddr, __ubuf__ float* eAddr, float sScalar,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> eReg, sReg;
    Reg::MaskReg mask;
    Reg::Duplicate(sReg, sScalar);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(oneRepeatSize);
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(eReg, eAddr + off);
        Reg::Div(eReg, eReg, sReg, mask);
        if constexpr (kIsFp32<D_T>) {
            Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(yAddr) + off, eReg, mask);
        } else {
            Reg::RegTensor<D_T> bReg;
            Reg::Cast<D_T, float, kCastF32ToB16<D_T>>(bReg, eReg, mask);
            Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(yAddr + off, bReg, mask);
        }
    }
}
```

核心：单行在线两趟（`ProcessOneRow`，AR pattern）：

```cpp
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxChunkedKernel<DType>::ProcessOneRow(int64_t aIdx)
{
    const int64_t rowGmOff = aIdx * rTotal_; // AR：行 GM 偏移 = aIdx * R

    // ════════ Pass 1：在线融合 max+sum（跨 chunk carry m/s，读 x 1 次）════════
    float m = -std::numeric_limits<float>::infinity();

    // 初始化 carry s = 0（carryBuf_ 即 cacheBuf，全寄存器置 0；[0] 为有效 s，余为同值镜像）
    auto carry = carryBuf_.Get<float>();
    __ubuf__ float* carryPtr = reinterpret_cast<__ubuf__ float*>(carry.GetPhyAddr());
    constexpr uint32_t kCarryCount = kRepF32; // 64 fp32 = 1 full VL 寄存器
    constexpr uint16_t kCarryReps = 1;
    asc_vf_call<DuplicateVf>(carryPtr, 0.0f, kCarryCount, kRepF32, kCarryReps);

    for (int64_t c = 0; c < rChunkCnt_; ++c) {
        int64_t chunkOff = c * rUbFactor_;
        int64_t rChunkLen = (c == rChunkCnt_ - 1) ? (rTotal_ - chunkOff) : rUbFactor_;
        int64_t rChunkAlign = CeilAlignKernel(rChunkLen, bsElemD_); // 32B 对齐 padded

        // ── CopyIn（MTE2）──
        auto xLocal = xQue_.template AllocTensor<D_T>();
        CopyInChunk(rowGmOff + chunkOff, rChunkLen, xLocal);
        xQue_.EnQue(xLocal);
        auto xDeq = xQue_.template DeQue<D_T>();

        // ── Cast D_T → fp32（V，asc_vf_call）──
        auto xF32 = xF32Buf_.Get<float>();
        __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xDeq.GetPhyAddr());
        __ubuf__ float* xF32Ptr = reinterpret_cast<__ubuf__ float*>(xF32.GetPhyAddr());
        const uint32_t castCount = static_cast<uint32_t>(rChunkAlign);
        const uint16_t castReps = static_cast<uint16_t>(CeilDivKernel(castCount, kRepF32));
        asc_vf_call<CastInVf<D_T>>(xF32Ptr, xPtr, castCount, kRepF32, castReps);
        xQue_.FreeTensor(xDeq);

        // ── ReduceMax（V）→ mChunk（标量）──
        auto maxDst = maxBuf_.Get<float>();
        uint32_t maxShape[2] = {1, static_cast<uint32_t>(rChunkAlign)};
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR>(maxDst, xF32, maxShape, true);
        float mChunk = maxDst.GetValue(0);

        // ── 在线重缩放：mNew = max(m, mChunk); s *= exp(m - mNew) ──
        //   IEEE754：NaN 与任何比较为 false → mNew 取 mChunk（NaN 传播）。
        //   delta = m - mNew ∈ (-∞,0]；m=+inf 且 mNew=+inf 时 delta=NaN → exp(NaN)=NaN。
        float mNew = (m > mChunk) ? m : mChunk;
        float delta = m - mNew;
        asc_vf_call<RescaleSumVf>(carryPtr, delta, kCarryCount, kRepF32, kCarryReps);

        // ── Sub + Exp（V）：eChunk = exp(x - mNew)（以新 m 为基准）──
        auto eDst = eBuf_.Get<float>();
        __ubuf__ float* ePtr = reinterpret_cast<__ubuf__ float*>(eDst.GetPhyAddr());
        asc_vf_call<SubExpVfAr<D_T>>(ePtr, xF32Ptr, mNew, castCount, kRepF32, castReps);

        // ── ReduceSum（V，fp32 累加）→ sumChunk（标量）──
        auto sumDst = sumBuf_.Get<float>();
        uint32_t sumShape[2] = {1, static_cast<uint32_t>(rChunkAlign)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(sumDst, eDst, sumShape, true);
        float sumChunk = sumDst.GetValue(0);

        // ── s += sumChunk（fp32 累加，AddScalarVf）──
        asc_vf_call<AddScalarVf>(carryPtr, sumChunk, kCarryCount, kRepF32, kCarryReps);

        // ── 更新 carry m ──
        m = mNew;
    }

    // 读取最终 s（Pass2 归一化分母）。carryBuf[0..kCarryCount-1] 经 DuplicateVf(0)
    // 初始化且 RescaleSumVf/AddScalarVf 均匀作用，故全部元素 == s（镜像）。
    // 用 ReduceSum 聚合取出 s：Reduce API 内部完成 V→S 同步，使后续 GetValue 安全
    // （GetValue 跨 pipe 读 UB，仅 Reduce API 后有保证；raw asc_vf_call 后无同步）。
    auto sDst = sumBuf_.Get<float>();
    uint32_t sShape[2] = {1, kCarryCount};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(sDst, carry, sShape, true);
    float s = sDst.GetValue(0) / static_cast<float>(kCarryCount);

    // ════════ Pass 2：归一化（重读 x，复用最终 m/s，读 x 1 次）════════
    for (int64_t c = 0; c < rChunkCnt_; ++c) {
        int64_t chunkOff = c * rUbFactor_;
        int64_t rChunkLen = (c == rChunkCnt_ - 1) ? (rTotal_ - chunkOff) : rUbFactor_;
        int64_t rChunkAlign = CeilAlignKernel(rChunkLen, bsElemD_);

        // ── CopyIn（MTE2），重读 x chunk ──
        auto xLocal = xQue_.template AllocTensor<D_T>();
        CopyInChunk(rowGmOff + chunkOff, rChunkLen, xLocal);
        xQue_.EnQue(xLocal);
        auto xDeq = xQue_.template DeQue<D_T>();

        // ── Cast D_T → fp32（V）──
        auto xF32 = xF32Buf_.Get<float>();
        __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xDeq.GetPhyAddr());
        __ubuf__ float* xF32Ptr = reinterpret_cast<__ubuf__ float*>(xF32.GetPhyAddr());
        const uint32_t castCount = static_cast<uint32_t>(rChunkAlign);
        const uint16_t castReps = static_cast<uint16_t>(CeilDivKernel(castCount, kRepF32));
        asc_vf_call<CastInVf<D_T>>(xF32Ptr, xPtr, castCount, kRepF32, castReps);
        xQue_.FreeTensor(xDeq);

        // ── Sub + Exp（V）：e = exp(x - m)（最终 m）──
        auto eDst = eBuf_.Get<float>();
        __ubuf__ float* ePtr = reinterpret_cast<__ubuf__ float*>(eDst.GetPhyAddr());
        asc_vf_call<SubExpVfAr<D_T>>(ePtr, xF32Ptr, m, castCount, kRepF32, castReps);

        // ── Div + Cast（V）：y = (e / s) cast D_T ──
        auto yLocal = yQue_.template AllocTensor<D_T>();
        __ubuf__ D_T* yPtr = reinterpret_cast<__ubuf__ D_T*>(yLocal.GetPhyAddr());
        asc_vf_call<DivCastVfAr<D_T>>(yPtr, ePtr, s, castCount, kRepF32, castReps);

        // ── CopyOut（MTE3）──
        yQue_.EnQue(yLocal);
        auto yDeq = yQue_.template DeQue<D_T>();
        CopyOutChunk(rowGmOff + chunkOff, rChunkLen, yDeq);
        yQue_.FreeTensor(yDeq);
    }
}
```

chunk 搬运（尾块非对齐用 `-inf` pad，使 ReduceMax 忽略 padding、`exp(-inf-m)=0` 使 ReduceSum 忽略 padding）：

```cpp
template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxChunkedKernel<DType>::CopyInChunk(
    int64_t gmOff, int64_t elems, LocalTensor<D_T>& dst)
{
    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(elems * elemBytes_);
    ext.blockCount = 1;
    ext.srcStride = 0;
    ext.dstStride = 0;
    ext.rsv = 0;
    uint32_t misalign = ext.blockLen & (kBlockBytes - 1u);
    uint8_t rPad = (misalign == 0) ? 0 : static_cast<uint8_t>((kBlockBytes - misalign) / elemBytes_);
    DataCopyPadExtParams<D_T> padParams{/*isPad=*/true, /*leftPadding=*/0, rPad, NegInfD<D_T>()};
    DataCopyPad(dst, xGm_[gmOff], ext, padParams);
}

template <typename DType>
__aicore__ inline void BenchHarnessSoftmaxChunkedKernel<DType>::CopyOutChunk(
    int64_t gmOff, int64_t elems, LocalTensor<D_T>& src)
{
    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(elems * elemBytes_);
    ext.blockCount = 1;
    ext.srcStride = 0;
    ext.dstStride = 0;
    ext.rsv = 0;
    DataCopyPad(yGm_[gmOff], src, ext);
}
```

多核分发行（大小核均衡、R chunk 单核串行满足确定性"禁止跨核 reduce 同一归约切片"）：

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
    const int64_t coreNum = static_cast<int64_t>(usedCoreNum_);
    int64_t small = aTotal_ / coreNum;            // floor
    int64_t bigCnt = aTotal_ % coreNum;
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
        ProcessOneRow(a);
    }
}
```

UB 初始化（按 chunk 尺寸分配，DB=2 的 xQue；`cacheBufUbSize` 为 tiling 给的固定 16KB）：

```cpp
    const int64_t xBufBytes = rUbFactorAlign_ * elemBytes_;          // D_T chunk
    const int64_t f32BufBytes = rUbFactorAlign_ * 4;                 // fp32 chunk
    const int64_t reduceBufBytes = static_cast<int64_t>(kBlockBytes); // 32B（标量 + pad）

    pipe_.InitBuffer(xQue_, /*bufNum=*/2, xBufBytes);   // xBuf DB=2
    pipe_.InitBuffer(yQue_, /*bufNum=*/1, xBufBytes);   // yBuf（同 D_T 大小）
    pipe_.InitBuffer(xF32Buf_, f32BufBytes);             // Cast 升 fp32
    pipe_.InitBuffer(eBuf_, f32BufBytes);                 // exp 中间结果
    pipe_.InitBuffer(maxBuf_, reduceBufBytes);            // ReduceMax 输出
    pipe_.InitBuffer(sumBuf_, reduceBufBytes);            // ReduceSum 输出
    pipe_.InitBuffer(carryBuf_, td->cacheBufUbSize);      // s carry（16KB）
```

## 设计决策

- 为何 Pass1 在线融合而非三趟：设计访存模型要求读 x 恰 2 次（2×N×sizeof(D_T) 级有效访存）；朴素三趟（m 趟 + s 趟 + 归一化趟）多 50% x 读，数值全对仍被验收按结构打回——**结构指标（读 x 次数）与数值指标分离验收**是本次的关键教训。ARA 布局的同类算子因 rescale 舍入累积的精度嫌疑选择了三趟（见 ChunkedR ARA 三趟卡），两卡取舍口径不同（AR 结构口径 / ARA 精度口径），按验收标准选型。
- s carry 放 carryBuf（UB）而非纯标量变量：在线重缩放需要 `exp(m_old-m_new)`，kernel 侧标量 `expf` 可用性未验证，而 `Reg::Exp` 只在 `__simd_vf__` 里可用；carryBuf + 均匀 VF 运算是把标量运算折叠进 VF 域的最小改动。
- 最终 s 用 ReduceSum 镜像聚合而非 `carry.GetValue(0)`：raw `asc_vf_call` 写后 GetValue 无 V→S 同步保证（详见 VF 归约落 UB 后 GetValue 卡），Reduce API 是已验证的跨 pipe 安全通道。
- 分支分发用 TilingData 字段（`rChunkCnt`）而非框架 tilingKey：单一 binary 下 `SetTilingKey(1)` 会选不到 binary（同算子 361001 分发问题，另见库内 tilingKey/funcEntry 映射卡）。
- m 用纯标量 `float` carry：m 的每次更新只依赖 `GetValue`（Reduce API 后安全）与比较运算，无跨 pipe 写回，标量即可。
- 特殊值语义对齐 PyTorch：`mNew = (m > mChunk) ? m : mChunk`（NaN 比较为 false → NaN 传播为 mChunk）；全 -inf 时 `delta=-inf-(-inf)=NaN → exp=NaN → s=NaN`，输出整切片 NaN，与全载路径等价；不做 clamp/饱和。

## 系统信息

- 硬件/软件：Ascend 950（regbase 范式），CANN 9.1.0。
- 工程布局：算子仓 ST 工程形态——`./build.sh <pkg>` 构建打包，run.sh 驱动 ST 与全载分支回归（Real 模式上板）。
- 相关文件：ChunkedR 分支 kernel 实现（`BenchHarnessSoftmaxChunkedKernel` 的 `ProcessOneRow`/`CopyInChunk`/`CopyOutChunk`/`Process` 与 UB `InitBuffer` 段，verbatim 见「完整代码」）；tiling 侧 `rUbFactor`/`cacheBufUbSize` 等 TilingData 字段的 UB 预算反解（`FloorAlign(min(r_i_max, rTotal), blockSize/elemBytes)`，含 `==0` FAIL 守卫）；关联卡：[VF 归约落 UB 后 GetValue 读取陈旧标量](../precision/vector_scalar_ub_handoff_requires_explicit_completion.md)、[ChunkedR ARA softmax 三趟 kernel 实现配方](chunkedr_ara_softmax_three_pass_recipe.md)。

## 预防

- 大 R 分 chunk 归约先固定"读 x 次数"与"carry 通道"两个结构决策，再写计算链；结构不符会在数值全对的情况下被验收打回。
- 跨 pipe 标量交换只走 Reduce API 输出（或已验证的显式同步）；raw VF 写 + GetValue 的组合在 review 阶段直接拦截。

## 验证信号

实现与验证来源：该 softmax 算子 ChunkedR 分支最终版（在线两趟）在 950 真机的 ST 回归与验收：

- ChunkedR 分支 ST 9/9 PASSED：多行多核用例 `[4,2048,50272]` fp16/bf16/fp32（8192 行多核 + 末 chunk 尾块 3792 非 16 对齐）、单行长 R `[1,2097152]` fp16（单核退化 + 226 chunk）、大 R 特殊值 `[1,16384]` fp32（NaN/+inf/全 -inf/-inf+有限/全相同值五类）（来源平台：950；证据等级：tool_observed）。
- 全载分支回归 32/32 PASSED，run.sh 退出码 0（来源平台：950；证据等级：tool_observed）。
- 结构核对：`ProcessOneRow` 内 `CopyInChunk` 恰 2 次调用（读 x 2 次）、`asc_vf_call` 22 处与 `__simd_vf__` 11 处命中真实实现代码（来源平台：950；证据等级：tool_observed）。
- 确定性：连续 3 次独立运行全过（修复前该路径存在逐次漂移）（来源平台：950；证据等级：tool_observed）。
- build `./build.sh <pkg>` 退出码 0、无 tiling 未注册告警（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只覆盖 AR pattern（axisNum=2，dim 为最内连续轴）；ARA（R 在中间轴、每块 `[R, A1]`）需按行 max/s 向量 carry 重构，未实现未验证。
- `rUbFactor` 必须由 tiling 的 UB 预算反解给出（`FloorAlign(min(r_i_max, rTotal), blockSize/elemBytes)`，含 `==0` FAIL 守卫）；kernel 侧假定 `rUbFactor` 已对齐且 ≥1 block。
- `Reg::UpdateMask(count)` 在 count > VL 元素数时末 repeat 的超读语义依赖"padding 元素无害"（-inf pad 或不落 GM 的尾块），不可把 count 直接设成未对齐有效长而依赖超读区数据。
- 950/regbase 专用指令序列（`Reg::*`、`asc_vf_call`、`Pattern::Reduce::AR`）；其它平台未验证，推断可迁移（未验证，推断可迁移）。
- kernel 内 `expf`（标量 libm）路线未验证可用性，不作为 carry 备选依据。

**未决项**：

- ARA pattern 的大 R 在线路径未实现（设计要求后续补）。
- 在线两趟 vs 朴素三趟的真机性能差未测量（无 profiling 信号）；如需量化，下个测量点是同 shape 两版本端到端耗时对比。
- `xBuf` DB=2 的流水收益未单独测量（当前串行 Alloc/EnQue/DeQue/Free 用法下 DB 只提供缓冲复用）。
## 坏实践

- 先写朴素三趟"先把数值做对"再改两趟——两次返工 + 一次结构打回；结构（读 x 次数、carry 通道、分发方式）应一次定对。
- 在线 carry 的取值点放 raw `GetValue`（详见 VF 归约落 UB 后 GetValue 读取陈旧标量卡的坏实践清单）。

## 误诊

- 验收第一轮误判"数值 9/9 通过即达成"——结构标准（在线两趟、读 x 2 次）独立于数值标准，数值通过不豁免结构核对。
- 曾怀疑多核分核/GM 偏移/UB 预算为 sum>1 根因，全部排除（详见 VF 归约落 UB 后 GetValue 卡误诊节）。
