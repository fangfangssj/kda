---
type: "Runbook"
title: tile-loop 统一 kernel：full-load 退化为单 tile
description: 多分支 tiling 算子新增 tile-loop 分支的完整配方：tiling 公式（ubLoop/ubTail）+ kernel 统一退化循环（full-load 即 loopNum=1 特例，不按 tilingKey 分派）+ 分支标记 ST 红绿闭环；含 TDD 红基线的 shape 尺寸陷阱。
tags: [task.implementation, topic.tiling, topic.implementation]
status: stable
sources: []
created_at: '2026-09-16T11:11:04Z'
updated_at: '2026-09-16T11:11:04Z'
platforms: ["950"]
---

# tile-loop 统一 kernel：full-load 退化为单 tile

## 现象

在 Ascend950PR（CANN 9.1.0）上为逐元素算子 SoftplusV2Grad 补齐第二个 tiling 分支：已有 Branch-1（full-load，tilingKey=1）只覆盖"单核工作集装得进 UB"的 shape（blockFormer ≤ ubFormer）；大 shape（如 `[4096,4096]` fp32，16.7M 元素）单核工作集超 UB，需要 Branch-2（tile-loop，tilingKey=2）在每个 block 内做 UB 循环切 tile。走测试先行流程：先写 Branch-2 的 TilingUT（独立 oracle 手算公式、stub 实现保 0 通过基线），再实现 tiling 公式；再写 Branch-2 的 Kernel ST（kernel 未实现时全红），再实现 kernel。

实现 kernel 时的关键场景状态：已有的 Branch-1 kernel 是单 tile 路径（每 block 一次 CopyIn/Compute/CopyOut）；若按 tilingKey 在 kernel 里 if/else 分派两条代码路径，两个分支的搬运/计算代码几乎全重复。

## 适用场景

- 工程形态：aclnn 自定义算子开发——多分支 tiling（full-load / tile-loop）的 AscendC kernel + host TilingFunc + 测试先行 UT/ST。
- API 路径：`REGISTER_TILING_DEFAULT` / `GET_TILING_DATA` / `TQue` / `DataCopyPad`（`DataCopyExtParams`+`DataCopyPadExtParams`）/ `asc_vf_call<...>` 调 `__simd_vf__` 寄存器链 / `GetVecLen()`。
- 验证过的平台与 CANN 版本：Ascend950PR（AICore 28 / AIV 56），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"按 blockFormer 与 ubFormer 的大小关系分 full-load / tile-loop 两分支"的逐元素算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：非逐元素（含跨 tile 归约/依赖）的 kernel——退化循环不变量不成立；本卡不含双缓冲（深度调整的联动见失效边界）。

## 解决配方

核心思路：**kernel 不按 tilingKey 分派，把 full-load 建成 tile-loop 的退化特例**。tiling 侧保证 full-load 时 `ubLoopOfFormerBlock=1、ubTailOfFormerBlock=blockFormer`（尾 block 同理），kernel 统一执行"loopNum-1 个满 ubFormer 主 tile + 1 个尾 tile"，loopNum=1 时自然退化为单 tile。

1. Branch-2 tiling 公式（`softplus_v2_grad_tiling_host.cpp`，verbatim；替换测试先行阶段的 stub）：

```cpp
void ComputeBranch2Tiling(const Branch2Inputs& in, SoftplusV2GradTilingData& out)
{
    out.totalLength = in.dim0;
    out.beta = in.beta;
    out.threshold = in.threshold;
    // 本分支前提 dim0>0 且 blockFormer>ubFormer（不做空短路；空 tensor 不进本分支）
    // —— 多核切分（elemBits = minDtypeBits）——
    int64_t coreNum = CeilDiv(in.dim0 * static_cast<int64_t>(in.minDtypeBits), kMinTilingBits);
    coreNum = std::min(coreNum, static_cast<int64_t>(in.availableCoreNum));
    if (coreNum < 1) {
        coreNum = 1;
    }
    int64_t blockFormer = AlignUp(CeilDiv(in.dim0, coreNum), kElemAlignFactor);
    int64_t blockNum = CeilDiv(in.dim0, blockFormer);
    int64_t blockTail = in.dim0 - (blockNum - 1) * blockFormer;
    // —— UB 切分（extraSize = 0；bufferDivisor 字节口径）——
    // fp32 直算 K=0：bufferDivisor = bufferNum * elemBytes
    // fp16 升精度 K=3：bufferDivisor = bufferNum * elemBytes + K * 4
    int64_t bufferDivisor = static_cast<int64_t>(in.bufferNum) * in.elemBytes
                            + static_cast<int64_t>(in.kBuf) * 4;
    int64_t maxElemNum = in.ubSize / bufferDivisor;
    int64_t alignFactor = kAlign256 / in.elemBytes;
    int64_t ubFormer = AlignDown(maxElemNum, alignFactor);
    if (ubFormer < 1) {
        ubFormer = alignFactor;
    }
    // —— tile-loop 切分（本分支前提 blockFormer > ubFormer，每 block 需 UB 内循环）——
    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);   // ≥2
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t ubLoopOfTailBlock   = CeilDiv(blockTail,   ubFormer);
    int64_t ubTailOfTailBlock   = blockTail   - (ubLoopOfTailBlock   - 1) * ubFormer;
    int32_t tilingKey = 2;
    out.coreNum = static_cast<int32_t>(coreNum);
    out.blockDim = static_cast<int32_t>(blockNum);
    out.blockFormer = blockFormer;
    out.blockTail = blockTail;
    out.ubFormer = ubFormer;
    out.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    out.ubTailOfFormerBlock = ubTailOfFormerBlock;
    out.ubLoopOfTailBlock = ubLoopOfTailBlock;
    out.ubTailOfTailBlock = ubTailOfTailBlock;
    out.tilingKey = tilingKey;
}
```

（后续双缓冲修复把 bufferDivisor 乘上 depthFactor；见失效边界的联动提示。）

2. 统一 tile-loop kernel（`softplus_v2_grad_kernel.cpp` 终版核心，verbatim；UB 缓冲按单 tile 容量 ubFormer 分配，逐 tile Alloc/Free 使循环中 UB 占用恒定）：

```cpp
template <typename DT>
class SoftplusV2GradKernel {
public:
    __aicore__ inline SoftplusV2GradKernel() {}

    __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR self, GM_ADDR grad_input,
                                const SoftplusV2GradTilingData* td)
    {
        totalLength_ = td->totalLength;
        blockFormer_ = td->blockFormer;
        blockTail_ = td->blockTail;
        blockDim_ = td->blockDim;
        ubFormer_ = td->ubFormer;
        ubLoopOfFormerBlock_ = td->ubLoopOfFormerBlock;
        ubTailOfFormerBlock_ = td->ubTailOfFormerBlock;
        ubLoopOfTailBlock_ = td->ubLoopOfTailBlock;
        ubTailOfTailBlock_ = td->ubTailOfTailBlock;
        beta_ = td->beta;
        threshold_ = td->threshold;
        if (totalLength_ == 0) {  // empty-tensor guard: return, no compute
            return;
        }
        gmGrad_.SetGlobalBuffer((__gm__ DT*)grad_output);
        gmSelf_.SetGlobalBuffer((__gm__ DT*)self);
        gmDx_.SetGlobalBuffer((__gm__ DT*)grad_input);
        // UB buffers sized to ubFormer (single-tile capacity). For full-load
        // (Branch-1) ubFormer >= blockFormer so the whole per-block set fits;
        // for tile-loop (Branch-2) ubFormer < blockFormer and we loop per tile.
        // Tiling guarantees 3*ubFormer*elemBytes (+K*4 for fp16) <= ubSize.
        uint32_t bufBytes = static_cast<uint32_t>(ubFormer_ * sizeof(DT));
        pipe_.InitBuffer(inQueueX_, 1, bufBytes);
        pipe_.InitBuffer(inQueueDY_, 1, bufBytes);
        pipe_.InitBuffer(outQueueDX_, 1, bufBytes);
    }

    __aicore__ inline void Process()
    {
        if (totalLength_ == 0) {
            return;
        }
        int32_t blockIdx = AscendC::GetBlockIdx();
        bool isLastBlock = (blockIdx == blockDim_ - 1);
        // per-block loop/tail (Branch-1: loopNum=1→single tile;
        //                     Branch-2: loopNum>=2→main tiles + tail tile)
        int64_t loopNum = isLastBlock ? ubLoopOfTailBlock_ : ubLoopOfFormerBlock_;
        int64_t tailNum = isLastBlock ? ubTailOfTailBlock_ : ubTailOfFormerBlock_;
        int64_t gmOffset = static_cast<int64_t>(blockIdx) * blockFormer_;
        // —— UB inner loop: main tiles (loopNum-1 × full ubFormer) ——
        for (int64_t i = 0; i < loopNum - 1; i++) {
            ProcessTile(gmOffset + i * ubFormer_, ubFormer_);
        }
        // —— tail tile (tailNum effective elems, padding not written to GM) ——
        ProcessTile(gmOffset + (loopNum - 1) * ubFormer_, tailNum);
    }

private:
    __aicore__ inline void ProcessTile(int64_t offset, int64_t tileLen)
    {
        CopyIn(offset, tileLen);
        Compute(tileLen);
        CopyOut(offset, tileLen);
    }
    __aicore__ inline void CopyIn(int64_t offset, int64_t tileLen)
    {
        AscendC::LocalTensor<DT> xLocal = inQueueX_.template AllocTensor<DT>();
        AscendC::LocalTensor<DT> dyLocal = inQueueDY_.template AllocTensor<DT>();
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = static_cast<uint32_t>(tileLen * sizeof(DT));
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DT> padParams;  // isPad=false, no padding
        AscendC::DataCopyPad(xLocal, gmSelf_[offset], extParams, padParams);
        AscendC::DataCopyPad(dyLocal, gmGrad_[offset], extParams, padParams);
        inQueueX_.EnQue(xLocal);
        inQueueDY_.EnQue(dyLocal);
    }

    __aicore__ inline void Compute(int64_t tileLen)
    {
        AscendC::LocalTensor<DT> xLocal = inQueueX_.template DeQue<DT>();
        AscendC::LocalTensor<DT> dyLocal = inQueueDY_.template DeQue<DT>();
        AscendC::LocalTensor<DT> dxLocal = outQueueDX_.template AllocTensor<DT>();

        // oneRepeatSize = VL / sizeof(float) = 64 fp32 elements per Vector repeat.
        // The fp16 path unpacks 64 half values into 64 fp32 slots, so 64 applies
        // to both dtype variants.
        uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
        uint32_t count = static_cast<uint32_t>(tileLen);
        uint16_t repeatTimes = static_cast<uint16_t>((count + oneRepeatSize - 1) / oneRepeatSize);
        asc_vf_call<SoftplusV2GradVF<DT>>(
            (__ubuf__ DT*)dxLocal.GetPhyAddr(),
            (__ubuf__ DT*)xLocal.GetPhyAddr(),
            (__ubuf__ DT*)dyLocal.GetPhyAddr(),
            count, oneRepeatSize, repeatTimes, beta_, threshold_);

        outQueueDX_.EnQue(dxLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueDY_.FreeTensor(dyLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t tileLen)
    {
        AscendC::LocalTensor<DT> dxLocal = outQueueDX_.template DeQue<DT>();
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = static_cast<uint32_t>(tileLen * sizeof(DT));
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmDx_[offset], dxLocal, extParams);
        outQueueDX_.FreeTensor(dxLocal);
    }
```

3. 计算链：整条数值稳定 sigmoid 链保持为**单个** `__simd_vf__` 函数（dav-3510 连续 Vector 链不拆成多个 UB 层调用），fp16 在装载时 `LoadAlign<DIST_UNPACK_B16>` + `Cast` 升 fp32、K=3 份中间量驻留寄存器不占 UB，两分支共用同一条链，Branch-2 只是在外面包 tile 循环（函数全文见下方完整代码）。

4. ST 分支标记与红绿闭环：golden 生成器为每个分支提供 `_branchX_cases()`，manifest 每条用例加 `"branch"` 字段，测试端按分支汇总，红/绿以分支为单位判定（kernel 未实现 Branch-2 时 Branch-2 应 0 通过、Branch-1 15/15 不回归）。

## 完整代码

VF 计算链（`SoftplusV2GradVF`，verbatim；数值稳定 sigmoid：`sig_pos=1/(1+exp(Mins(-z,0)))`、`sig_neg=exp(Mins(z,0))/(1+exp(Mins(z,0)))`，`dx = where(z>threshold, dy, dy*sig)`，严格 `>` 使 z==threshold 走光滑分支）：

```cpp
// CastTraits mirroring CANN Internal::castTraitB16ToB32 / castTraitB32ToB16
// (half<->float on DAV_3510 / ascend950). Defined locally to avoid depending on
// internal headers.
constexpr AscendC::Reg::CastTrait CAST_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::Reg::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait CAST_B32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::Reg::RoundMode::CAST_RINT};

template <typename DT>
__simd_vf__ inline void SoftplusV2GradVF(
    __ubuf__ DT* dxAddr, __ubuf__ DT* xAddr, __ubuf__ DT* dyAddr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes,
    float beta, float threshold)
{
    constexpr bool isHalf = (sizeof(DT) == sizeof(half));
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg mGT;  // z > threshold
    AscendC::Reg::MaskReg mGE;  // z >= 0
    AscendC::Reg::RegTensor<DT> xRaw;
    AscendC::Reg::RegTensor<DT> dyRaw;
    AscendC::Reg::RegTensor<DT> dxRaw;
    AscendC::Reg::RegTensor<float> xF;
    AscendC::Reg::RegTensor<float> dyF;
    AscendC::Reg::RegTensor<float> z;
    AscendC::Reg::RegTensor<float> t1;   // sig_pos work register
    AscendC::Reg::RegTensor<float> t2;   // sig_neg work register
    AscendC::Reg::RegTensor<float> t3;   // scratch (1 + e_pos)
    AscendC::Reg::RegTensor<float> sig;  // sigma(z)
    AscendC::Reg::RegTensor<float> smooth;
    AscendC::Reg::RegTensor<float> dxF;

    uint32_t sreg = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        // Mask for oneRepeatSize (64) fp32 slots; POST_UPDATE decrements sreg
        // so the tail repeat enables only the remaining valid elements.
        mask = AscendC::Reg::UpdateMask<float>(sreg);

        // ---- load + cast-in (fp16 path upcasts to fp32) ----
        if constexpr (isHalf) {
            AscendC::Reg::LoadAlign<DT, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xRaw, xAddr + i * oneRepeatSize);
            AscendC::Reg::Cast<float, DT, CAST_B16_TO_B32>(xF, xRaw, mask);
            AscendC::Reg::LoadAlign<DT, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(dyRaw, dyAddr + i * oneRepeatSize);
            AscendC::Reg::Cast<float, DT, CAST_B16_TO_B32>(dyF, dyRaw, mask);
        } else {
            AscendC::Reg::LoadAlign(xF, xAddr + i * oneRepeatSize);
            AscendC::Reg::LoadAlign(dyF, dyAddr + i * oneRepeatSize);
        }

        // z = beta * x
        AscendC::Reg::Muls(z, xF, beta, mask);
        // maskGT = (z > threshold)  — strict >, z==threshold -> smooth branch
        AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(mGT, z, threshold, mask);

        // ---- stable sigmoid, z >= 0 branch: sig_pos = 1/(1+exp(Mins(-z,0))) ----
        AscendC::Reg::Muls(t1, z, -1.0f, mask);   // -z
        AscendC::Reg::Mins(t1, t1, 0.0f, mask);   // Mins(-z, 0) -> exp arg <= 0
        AscendC::Reg::Exp(t1, t1, mask);          // exp(-z) in (0,1]
        AscendC::Reg::Adds(t1, t1, 1.0f, mask);   // 1 + exp(-z)
        AscendC::Reg::Duplicate(sig, 1.0f, mask);
        AscendC::Reg::Div(sig, sig, t1, mask);    // sig_pos = 1/(1+exp(-z))

        // ---- stable sigmoid, z < 0 branch: sig_neg = exp(Mins(z,0))/(1+exp(Mins(z,0))) ----
        AscendC::Reg::Mins(t2, z, 0.0f, mask);    // Mins(z, 0) -> exp arg <= 0
        AscendC::Reg::Exp(t2, t2, mask);          // exp(z) in (0,1)
        AscendC::Reg::Adds(t3, t2, 1.0f, mask);   // 1 + exp(z)
        AscendC::Reg::Div(t2, t2, t3, mask);      // sig_neg = exp(z)/(1+exp(z))

        // sig = (z >= 0) ? sig_pos : sig_neg
        AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(mGE, z, 0.0f, mask);
        AscendC::Reg::Select(sig, sig, t2, mGE);  // dst=src0 (sig_pos) in-place

        // smooth = dy * sig ; dx = where(z > threshold, dy, smooth)
        AscendC::Reg::Mul(smooth, dyF, sig, mask);
        AscendC::Reg::Select(dxF, dyF, smooth, mGT);

        // ---- cast-out + store (fp16 path rounds fp32 back to half) ----
        if constexpr (isHalf) {
            AscendC::Reg::Cast<DT, float, CAST_B32_TO_B16>(dxRaw, dxF, mask);
            AscendC::Reg::StoreAlign<DT, AscendC::Reg::StoreDist::DIST_PACK_B32>(dxAddr + i * oneRepeatSize, dxRaw, mask);
        } else {
            AscendC::Reg::StoreAlign(dxAddr + i * oneRepeatSize, dxF, mask);
        }
    }
}
```

kernel 入口（`extern "C" __global__ __aicore__`，verbatim）：

```cpp
extern "C" __global__ __aicore__ void softplus_v2_grad(
    GM_ADDR grad_output, GM_ADDR self, GM_ADDR grad_input,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(SoftplusV2GradTilingData);
    GET_TILING_DATA(tilingData, tiling);

    SoftplusV2GradKernel<T> op;
    op.Init(grad_output, self, grad_input, &tilingData);
    op.Process();

    AscendC::PipeBarrier<PIPE_ALL>();
}
```

Branch-2 ST 用例生成（`golden.py` `_branch2_cases()` 骨架与特殊值注入，verbatim 节选；`spec_shape` 的取值见坏实践——首版 `[4096,512]` 曾意外通过）：

```python
# ===== Branch-2 所辖组合（tile-loop tilingKey=2）=====
# 大 shape 单核工作集超 UB（blockFormer > ubFormer），需 UB 内循环切 tile。
def _branch2_cases():
    """返回 [(case_id, dtype_str, shape, beta, threshold, grad_output_np, self_np, remark)]。"""
    cases = []
    rng = np.random.RandomState(20260716)

    # B2_L1: 大连续 shape [4096,4096] fp32 (16,777,216 元素) — 本分支首要带宽用例
    dim0_l1 = 4096 * 4096
    slf = rng.uniform(-25.0, 25.0, size=dim0_l1).astype(np.float32).reshape(4096, 4096)
    go = rng.uniform(0.1, 1.0, size=dim0_l1).astype(np.float32).reshape(4096, 4096)
    cases.append(("B2_L1", "float32", [4096, 4096], 1.0, 20.0, go, slf,
                  "L1 [4096,4096] fp32 large shape, multi-core + UB tile-loop"))

    # ===== 大 shape 中注入特殊值各一例 =====
    # 使用 [4096,1024]=4,194,304 元素 fp32/fp16，展平后 blockFormer > ubFormer 命中本分支。
    # shape 足够大使 Branch-1 kernel 的 InitBuffer(blockFormer) 超 UB 预算 →
    # Branch-2 kernel 未实现时必然 ERROR/FAIL（TDD 红，期望通过数 = 0）。
    # 注入位置跨越不同 block/tile，验证 tile-loop 下逐 tile 逐元素特殊值路径不退化。
    spec_dim = 4096 * 1024
    spec_shape = [4096, 1024]
    inject_pos = [0, 21120, 44032, 100000, 500000, 1000000, 2000000, spec_dim - 1]

    # B2_S1_NAN: self 含 NaN -> nan_propagates
    slf = rng.uniform(-25.0, 25.0, size=spec_dim).astype(np.float32).reshape(spec_shape)
    go = rng.uniform(0.1, 1.0, size=spec_dim).astype(np.float32).reshape(spec_shape)
    for p in inject_pos:
        slf.flat[p] = np.nan
    cases.append(("B2_S1_NAN", "float32", spec_shape, 1.0, 20.0, go, slf,
                  "large shape self inject NaN -> nan_propagates"))
    # ... S2 +Inf / S3 -Inf / S4 零点 / S5 z==threshold / S6 全零梯度 / S7 fp16 上溢 同构
    return cases
```

manifest 分支标记与按分支汇总（`generate_manifest` 核心改动，verbatim 节选）：

```python
    # Branch-1 + Branch-2 所辖组合，标注 branch
    all_branch_cases = [
        ("Branch-1", _branch1_cases()),
        ("Branch-2", _branch2_cases()),
    ]
    for branch_name, branch_cases in all_branch_cases:
        for cid, dtstr, shape, beta, thr, go_np, slf_np, remark in branch_cases:
            # 独立 golden：CPU + PyTorch，fp16 先 upcast fp32 计算
            ...
            cases_json.append({
                "id": cid,
                "branch": branch_name,
                "dtype": dtstr,
                "shape": shape,
                ...
            })
```

bitwise 一致性探针（独立小程序核心，verbatim；确定性逐元素算子同输入两次运行必须字节一致）：

```cpp
    if (runOnce(devOut1) != 0) { fprintf(stderr, "run1 failed\n"); return 1; }
    if (runOnce(devOut2) != 0) { fprintf(stderr, "run2 failed\n"); return 1; }

    std::vector<uint8_t> out1(bytes), out2(bytes);
    CHECK_ACL(aclrtMemcpy(out1.data(), bytes, devOut1, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(out2.data(), bytes, devOut2, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    int mism = memcmp(out1.data(), out2.data(), bytes);
    if (mism == 0) {
        fprintf(stdout, "[PASS] bitwise consistent: 2 runs on [4096,4096] fp32 produce identical output (%zu bytes)\n", bytes);
    } else {
        fprintf(stderr, "[FAIL] bitwise mismatch between 2 runs\n");
        return 1;
    }
```

## 设计决策

- 为什么 kernel 统一而不按 tilingKey if/else 分派：full-load 是 tile-loop 的退化（loopNum=1），统一路径后搬运/计算/同步代码只有一份，两分支回归同一实现，消除"改一条路径漏另一条"的维护风险；tilingKey 只用于 host 侧决定产出哪种切分字段组合。
- 为什么 UB 缓冲按 ubFormer（单 tile 容量）分配而不是 blockFormer：tile-loop 分支的 blockFormer 本来就大于 UB 容量；full-load 分支 ubFormer ≥ blockFormer，按 ubFormer 分配同样覆盖。逐 tile Alloc/Free 使循环全程 UB 占用恒定。
- 为什么尾 tile 单独传 tailNum：DataCopyPad 按字节数搬运，尾 tile 只搬有效元素，padding 不写回 GM，天然支持非对齐尾块（`ubTailOfXxxBlock ∈ (0, ubFormer]`）。
- 为什么 dtype 用编译期模板参数（构建注入 `-DDTYPE_GRAD_OUTPUT=float/half`）而不是 TilingData 字段：dtype 是编译期已知量，进 TilingData 会把编译期分发变运行期分支（本算子设计约定）。
- 为什么 `oneRepeatSize = GetVecLen()/sizeof(float)` 对 fp16 也成立：fp16 路径用 `DIST_UNPACK_B16` 把 64 个 half 解包进 64 个 fp32 槽位，每 repeat 处理的元素数与 fp32 相同。
- 首版实现把 TQue 深度定为 1（当时的判定"fp32 UB 余量不足"），后被带宽实测推翻并改为 fp32 深度 2——该演进另有专卡；本卡的验证信号对应深度 1 版本的正确性验收。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0，`cd operators && ./build.sh softplus_v2_grad_package`（kernel 二进制按 dtype 组合生成两份）。
- 目录布局：
  - `op_kernel/softplus_v2_grad_kernel.cpp`——统一 kernel + `__simd_vf__` VF 链（本卡完整代码）。
  - `op_kernel/softplus_v2_grad_tiling.h`——TilingData 结构（blockFormer/blockTail/ubFormer/ubLoop*/ubTail*/tilingKey 等）。
  - `op_host/softplus_v2_grad_tiling_host.{h,cpp}`——公共切分 + Branch-1/Branch-2 TilingFunc。
  - `docs/softplus_v2_grad/develop/golden.py`——独立 oracle（CPU + PyTorch，fp16 先升 fp32），`_branch1_cases()` + `_branch2_cases()` 生成带 branch 字段的 manifest 与 .bin。
  - `tests/aclnn/test_aclnn.cpp` + `run.sh`——ST 执行与分支汇总。
  - `tests/ut/test_softplus_v2_grad_tiling.cpp` / `..._branch1_tiling.cpp` / `..._branch2_tiling.cpp`——独立 oracle TilingUT（oracle 自带 `CeilDiv/AlignUp/AlignDown` 手算，不调被测函数）。
- Branch-2 ST 用例矩阵（10 例）：B2_L1 `[4096,4096]` fp32、B2_L3 rank=8 `[8]*8` fp32（各 16.7M 元素）、B2_L1_FP16 `[4096,4096]` fp16、特殊值注入 B2_S1-S7（NaN/+Inf/-Inf/零点/z==threshold/全零梯度/fp16 上溢，`[4096,1024]` 4.2M 元素，注入位置跨不同 block/tile：`[0, 21120, 44032, 100000, 500000, 1000000, 2000000, dim-1]`）。

## 预防

- 新增 tiling 分支先写 UT（独立 oracle + stub 0 通过基线）再写实现；kernel 侧优先考虑"新分支能否建为旧分支的退化特例"而不是 if/else 复制路径。
- 多分支 ST 的用例 manifest 必须带分支标记并在测试端按分支汇总，否则分支级红/绿不可判定。
- 设计 tile-loop 测试 shape 时先算一遍 `blockFormer × elemBytes × 队列数` 与 UB 的关系（见坏实践）。

## 验证信号

以下信号来自该算子 Branch-2 的测试先行与实现会话（Ascend950PR + CANN 9.1.0；TilingUT → tiling 实现 → ST 红 → kernel 实现 → 全绿）：

- TDD 红基线（kernel 未实现 Branch-2）：ST 总计 25 用例，Branch-1 15/15 PASSED 不回归，Branch-2 0/10 PASSED、10 ERROR（特殊值用例放大到 4.2M 元素后；首版 2.1M 元素时 1 例意外通过，见坏实践）（来源平台：950；证据等级：tool_observed）。
- tiling 公式实现后：Branch-2 TilingUT 9/9、公共 TilingUT 22/22、Branch-1 TilingUT 11/11 全过，build exit 0，无 tiling struct 未注册告警（来源平台：950；证据等级：tool_observed）。
- kernel 实现后 ST 全绿：25/25 PASSED（Branch-1 15/15 + Branch-2 10/10）；B2_L3 rank=8 16,777,216 元素 fp32 rtol=1e-05 通过、B2_L1_FP16 16,777,216 元素 rtol=0.001 通过、特殊值用例 4,194,304 元素逐点对齐（来源平台：950；证据等级：tool_observed）。
- bitwise 确定性：同输入 2 次运行 `[4096,4096]` fp32 输出 67,108,864 字节 memcmp=0，退出码 0（来源平台：950；证据等级：tool_observed）。
- kernel 内 `__simd_vf__` 计数 2、`asc_vf_call` 计数 2（验收 grep 项）（来源平台：950；证据等级：tool_observed）。
注：以上正确性验收对应 TQue 深度 1 版本；后续带宽修复（深度 2）后的复测见同主题带宽专卡。

## 失效边界

- 退化不变量依赖 tiling 保证 full-load 时 `ubLoopOfFormerBlock=1 且 ubTailOfFormerBlock=blockFormer`；若 tiling 公式改了（如双缓冲 depthFactor 重算），kernel 与 UT oracle 必须同步。
- 非对齐尾块靠 DataCopyPad 的字节级 blockLen 处理；若换成要求数据对齐的搬运原语，尾 tile 需另做对齐处理（未验证）。
- fp16 的 K=3 中间量驻留寄存器只对"中间量数量固定且小"的计算链成立；中间量多的链会挤占寄存器或需要回 UB（未验证）。
- 仅 Ascend950PR + CANN 9.1.0 实测（未验证，推断可迁移至其他有 TQue/DataCopyPad/asc_vf_call 的 AscendC 平台）。

**未决项**：

- 双缓冲修复后的 ST/bandwidth 复测未在本切片内完成（同主题带宽专卡的遗留项）。
- rank=8 以上的高维 shape、int 类型、`beta≠1`/`threshold≠20` 之外的属性组合未在 Branch-2 覆盖（golden 矩阵按设计文档取样，未穷举）。
## 坏实践

- **TDD 红基线的 shape 尺寸陷阱**：首版 Branch-2 特殊值用例用 `[4096,512]`=2,097,152 元素 fp16，Branch-1 kernel 按 blockFormer 分配 UB（3×44032×2B≈258KB）恰好装进实际 UB，导致该用例在 kernel 未实现 Branch-2 时**意外通过**（红基线破口 1/10）。判别算式：`blockFormer × elemBytes × 3 > ubSize` 必须对所有红基线用例成立——放大到 `[4096,1024]`=4,194,304 元素（blockFormer≈87552，fp16 3×87552×2B≈513KB、fp32 ≈1MB，均超 ~256KB UB）后 0/10 ERROR，红基线恢复。中等规模的 fp16 用例最容易踩这个边界。
- 按 tilingKey 在 kernel 里复制两条搬运/计算路径——重复代码 + 双份回归负担。
- ST 汇总只看总数不看分支——Branch-2 全 ERROR 时总数 15/25 仍可能被误读为"部分通过"。

## 误诊

- kernel 开发者自评时把带宽验收一并判为通过（基于错误的峰值假设），未察觉性能项未达标——带宽判定另有专卡（该自评为 agent 转述，证据等级：agent_reported，不进验证信号）。
- 测试先行阶段 oracle 独立性自检曾两次把 grep 窗口扩到测试驱动调用行，误报"oracle 调用了被测函数"；改为只扫 oracle 函数体 span 后确认独立（通用工具方法问题，非垂域结论）。
