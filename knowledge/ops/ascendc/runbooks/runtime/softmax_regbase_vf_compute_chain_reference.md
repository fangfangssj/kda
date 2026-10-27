---
type: "Runbook"
title: "softmax regbase VF计算链参考结构"
description: "dav-3510 上 softmax 类 kernel 的已验证 VF 计算链结构：m/s 按块跨度存 cacheBuf、kernel 侧 GetValue 取标量、逐行 asc_vf_call + Reg::Duplicate 标量广播、StoreIfNeedCast(DIST_PACK_B32) 收窄写回、PIPE_ALL 全管道屏障。"
tags: [task.implementation, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# softmax regbase VF计算链参考结构

## 现象

（本卡是完整实现配方，从成功实现中提取。）在 ascend950（arch35/dav-3510，CANN 9.2.0）上为一个 softmax 类自定义算子实现分支 kernel（AR pattern：沿最后一轴 R 归约，每 A 行输出一组 softmax 值）时，计算链必须满足三条硬约束：(1) 连续 Vector 计算链（Sub+Exp、Div+Cast、Sub+Exp+Div+Cast）要用 `__simd_vf__` + `asc_vf_call` 实现；(2) 每行的 m（行最大值）与 s（行指数和）需要广播到整行参与逐元素运算；(3) fp16/bf16 输入要先升 fp32 计算、最终收窄回原 dtype 写回。开发过程中试错过三种 m/s 广播与存储结构（VF 内单元素加载+寄存器间广播 → 崩溃；kernel 侧预广播整 tile 缓冲 → 可用但多一块 UB；kernel 侧标量传参 + VF 内标量广播 → 最终采用），并从 SDK `SimpleSoftMaxGenericNDImpl` 吸收了块跨度 m/s 存储与 `StoreIfNeedCast` 收窄模式，最终结构在 46 例 ST 上取得 fp32 全过、特殊值切片（NaN/±inf/全零/标量）全过、0 运行时异常。

## 适用场景

- 工程形态：算子仓式自定义算子 kernel 开发（`op_kernel/arch35/*.h`），softmax/归约-归一化类算子（行归约 + 逐元素指数运算 + 收窄写回）。
- API 路径：`asc_vf_call`；`AscendC::Reg::{LoadAlign, StoreAlign, Sub, Exp, Div, Duplicate, Cast, CreateMask, UpdateMask}`；`AscendC::ReduceMax/ReduceSum<T, Pattern::Reduce::AR, false>`（5 参显式 sharedTmpBuffer）；`LocalTensor::GetValue`；`AscendC::Duplicate`（UB 级）；`AscendC::PipeBarrier<PIPE_ALL>`。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.2.0。与 frontmatter `platforms` 一致。
- 适用粒度：operator_family（行归约 + 逐元素变换 + dtype 收窄的算子族）。
- 路线优先级：recommended_fastpath（本结构已避开该平台全部已验证陷阱：非对齐 VF 访存、隐式 sharedTmpBuffer、PIPE_V 跨管道、缺 DIST_PACK_B32）。
- 明确不适用/未验证场景：需要跨核归约（行数据分布在多核）的场景需另加核间规约结构；int8 等非 b16 窄类型未验证；三遍 R-loop 路径的超长归约精度问题在本结构下仍存在（见遗留）。

## 根因/解决配方

（配方卡无故障根因；结构要点如下，全部代码见 ## 完整代码。）

结构五要素：

1. **m/s 块跨度存储**：每行的 m/s 标量在 cacheBuf 中按 `MS_STRIDE = 8`（fp32 元素，即 32 B 数据块）间隔存放——`dst[cacheOffElems + i * MS_STRIDE]`。块跨度保证任何单标量的加载地址都 32 B 对齐，且一个 `Duplicate` 恰好填满一个数据块。
2. **kernel 侧取标量**：`cache.GetValue(i * MS_STRIDE)` 把 m/s 读成 `float` 标量（Scalar 管道），随后直接作为 VF 函数的**值参数**传入——VF 内不再有任何标量缓冲加载。
3. **逐行 VF + 标量广播**：VF 函数内 `AscendC::Reg::Duplicate<float>(reg, scalar, maskFull)` 把标量广播到整寄存器（这是"标量值"重载，不是寄存器间广播重载），再对行内按 `VL_F32`（64 元素 = 256 B）分块 `LoadAlign/Sub/Exp/Div/StoreAlign`。
4. **收窄写回**：`StoreIfNeedCast<D_T>`——fp32 寄存器经 `Reg::Cast<D_T, float, kCastTraitF32ToB16>`（NO_SAT + CAST_RINT）后用 `StoreAlign<D_T, StoreDist::DIST_PACK_B32>` 打包写入 b16 目标；fp32 直接 `DIST_NORM`。
5. **全管道屏障**：所有阶段间（含 cacheBuf 的 V 管道写 → GetValue 的 S 管道读）用 `PipeBarrier<PIPE_ALL>()`。

## 设计决策

- **三种 m/s 广播结构的取舍**（本轨迹全部实测）：
  - VF 内 `LoadAlign(mReg, mAddr + i)` 单元素加载 + 寄存器间 `Duplicate(mReg, mReg, mask)`：**崩溃**（507035）——逐元素偏移破坏 32 B 对齐（详见同批专题卡 `asc_vf_call_unaligned_access_507035`）；
  - kernel 侧 `GetValue` + UB 级 `Duplicate` 预广播整 tile 缓冲 + 扁平 VF：可用，但额外占一整块 tile UB 且多一趟填充；
  - kernel 侧 `GetValue` 标量传参 + VF 内 `Reg::Duplicate<float>(reg, scalar, mask)` 标量广播（**采用**）：无额外缓冲、无 VF 内标量加载、对齐责任最小。文件头注释明确记录选择理由："avoids register-to-register Duplicate ambiguity"。
- **为何逐行 VF 而非整 tile 扁平 VF**：整 tile 扁平 VF 需要预广播缓冲（结构 2 的代价）；逐行 VF 把广播简化成标量传参，且行粒度正好匹配 `VL_F32` 分块循环。
- **MS_STRIDE=8 的依据**：`GetDataBlockSizeInBytes() / sizeof(float) = 32/4`，与 SDK `SimpleSoftMax` 的 blockStride 口径一致。
- **Reduce 用 5 参显式 sharedTmpBuffer**：4 参隐式分配在大 tile 崩溃（见库内 `reduce_work_buf_ub_507035`），显式 tmpBuf1 尺寸与 tile fp32 大小对齐。
- **CastTrait 提为全局 constexpr**：`Reg::Cast` 的 trait 模板参数需要全局 constexpr 引用，函数内构造不可用。
- 当前方案属 recommended_fastpath。

## 系统信息

- 工具链：CANN 9.2.0；bisheng compiler (clang 15.0.5)；算子仓 `operators/build.sh <Pkg>_package` 编译。
- 目标平台：ascend950 / dav-3510 / arch35；48 AIV 核；UB 253952 B。
- 参考实现：SDK `$ASCEND_HOME_PATH/asc/impl/adv_api/detail/activation/softmax/regbase/3510/simple_softmax_impl.h`（块跨度 + 寄存器广播 + StoreIfNeedCast 的出处）、同仓 `scale_custom` 算子（asc_vf_call 扁平循环的出处）。
- UB 布局：`preInBuf_`（D_T I/O）+ `computeBuf_`（fp32 计算，b16 时启用）+ `tmpBuf0_`（reduce 输出）+ `tmpBuf1_`（reduce 显式 sharedTmpBuffer）+ `cacheBuf_`（m/s，16384 B）。
- 相关文件：`op_kernel/arch35/bench_harness_softmax_kernel.h`（本卡代码出处）、`op_kernel/bench_harness_softmax_apt.cpp`（tilingKey 分发：key=0/3 共用、key=2 空 shape 早退）、`op_host/arch35/*_tiling_arch35.cpp`（tile 参数：aUbFactorAlign/rUbFactorAlign）、`tests/aclnn/run.sh`（ST 入口）。

## 完整代码

以下为结构核心（verbatim 取自终版写入，屏障为 PIPE_ALL 全管道版）：

```cpp
// ── 常量与 CastTrait（文件级 constexpr）──────────────────────────────
// Stride (in fp32 elements) between per-row m/s values in cacheBuf.
constexpr uint32_t MS_STRIDE = 8;

constexpr AscendC::Reg::CastTrait kCastTraitB16ToF32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::Reg::RoundMode::UNKNOWN
};
constexpr AscendC::Reg::CastTrait kCastTraitF32ToB16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::Reg::RoundMode::CAST_RINT
};

// ── VF helper：fp32 寄存器 → b16/fp32 UB（mirror SDK StoreIfNeedCast）──
template <typename D_T>
__simd_callee__ inline void StoreIfNeedCast(__ubuf__ D_T* dstUb,
                                             AscendC::Reg::RegTensor<float>& srcReg,
                                             AscendC::Reg::MaskReg& preg)
{
    if constexpr (sizeof(D_T) == 2) {
        AscendC::Reg::RegTensor<D_T> tmpReg;
        AscendC::Reg::Cast<D_T, float, kCastTraitF32ToB16>(tmpReg, srcReg, preg);
        AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstUb, tmpReg, preg);
    } else {
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(
            reinterpret_cast<__ubuf__ float*>(dstUb), srcReg, preg);
    }
}

// ── VF：单行 Sub + Exp（in-place fp32），m 标量经 Reg::Duplicate 广播 ──
template <uint16_t VL_F32>
__simd_vf__ inline void BHSoftmaxSubExpRowVf(
    __ubuf__ float* dataUb, float mScalar, uint16_t srcK)
{
    AscendC::Reg::MaskReg maskFull =
        AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<float> srcVreg;
    AscendC::Reg::RegTensor<float> maxVreg;
    AscendC::Reg::RegTensor<float> tmpVreg;

    AscendC::Reg::Duplicate<float>(maxVreg, mScalar, maskFull);

    uint32_t remaining = srcK;
    uint16_t repTimes = (srcK + VL_F32 - 1) / VL_F32;
    for (uint16_t j = 0; j < repTimes; ++j) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(srcVreg, dataUb + j * VL_F32);
        AscendC::Reg::Sub(tmpVreg, srcVreg, maxVreg, mask);
        AscendC::Reg::Exp(srcVreg, tmpVreg, mask);
        AscendC::Reg::StoreAlign(dataUb + j * VL_F32, srcVreg, mask);
    }
}

// ── VF：单行 Div + Cast(fp32→D_T)（fast path 用）─────────────────────
template <typename D_T, uint16_t VL_F32>
__simd_vf__ inline void BHSoftmaxDivCastRowVf(
    __ubuf__ D_T* dstUb, __ubuf__ float* srcUb, float sScalar, uint16_t srcK)
{
    AscendC::Reg::MaskReg maskFull =
        AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<float> srcVreg;
    AscendC::Reg::RegTensor<float> sumVreg;
    AscendC::Reg::RegTensor<float> tmpVreg;

    AscendC::Reg::Duplicate<float>(sumVreg, sScalar, maskFull);

    uint32_t remaining = srcK;
    uint16_t repTimes = (srcK + VL_F32 - 1) / VL_F32;
    for (uint16_t j = 0; j < repTimes; ++j) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(srcVreg, srcUb + j * VL_F32);
        AscendC::Reg::Div(tmpVreg, srcVreg, sumVreg, mask);
        StoreIfNeedCast<D_T>(dstUb + j * VL_F32, tmpVreg, mask);
    }
}

// ── VF：单行 Sub + Exp + Div + Cast（三遍路径 Pass 3 用）──────────────
template <typename D_T, uint16_t VL_F32>
__simd_vf__ inline void BHSoftmaxSubExpDivCastRowVf(
    __ubuf__ D_T* dstUb, __ubuf__ float* srcUb,
    float mScalar, float sScalar, uint16_t srcK)
{
    AscendC::Reg::MaskReg maskFull =
        AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<float> srcVreg;
    AscendC::Reg::RegTensor<float> maxVreg;
    AscendC::Reg::RegTensor<float> sumVreg;
    AscendC::Reg::RegTensor<float> tmpVreg;

    AscendC::Reg::Duplicate<float>(maxVreg, mScalar, maskFull);
    AscendC::Reg::Duplicate<float>(sumVreg, sScalar, maskFull);

    uint32_t remaining = srcK;
    uint16_t repTimes = (srcK + VL_F32 - 1) / VL_F32;
    for (uint16_t j = 0; j < repTimes; ++j) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(srcVreg, srcUb + j * VL_F32);
        AscendC::Reg::Sub(tmpVreg, srcVreg, maxVreg, mask);
        AscendC::Reg::Exp(srcVreg, tmpVreg, mask);
        AscendC::Reg::Div(tmpVreg, srcVreg, sumVreg, mask);
        StoreIfNeedCast<D_T>(dstUb + j * VL_F32, tmpVreg, mask);
    }
}
```

kernel 侧调用点（`BenchHarnessSoftmaxKernel<T, isTailR>` 成员；`VL_F32_U16 = 64`）：

```cpp
// Per-row VF calls: read m scalar from cacheBuf, call SubExpRowVf per row.
__aicore__ inline void DoSubExp(int64_t validARows)
{
    __ubuf__ float* dataPtr = GetComputePtr();
    auto cache = cacheBuf_.Get<float>();
    uint16_t srcK = static_cast<uint16_t>(td_->rUbFactorAlign);
    for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
        float mVal = cache.GetValue(static_cast<uint32_t>(i * MS_STRIDE));
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<BHSoftmaxSubExpRowVf<VL_F32_U16>>(
            dataPtr + i * td_->rUbFactorAlign, mVal, srcK);
    }
}

// Fast path: Div+Cast only (data is already exp(x-m)).
__aicore__ inline void DoDivCast(int64_t validARows)
{
    __ubuf__ float* srcPtr = GetComputePtr();
    __ubuf__ T* dstPtr = (__ubuf__ T*)preInBuf_.Get<T>().GetPhyAddr();
    auto cache = cacheBuf_.Get<float>();
    int64_t sOffset = td_->aUbFactorAlign * MS_STRIDE;
    uint16_t srcK = static_cast<uint16_t>(td_->rUbFactorAlign);
    for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
        float sVal = cache.GetValue(static_cast<uint32_t>(sOffset + i * MS_STRIDE));
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<BHSoftmaxDivCastRowVf<T, VL_F32_U16>>(
            dstPtr + i * td_->rUbFactorAlign,
            srcPtr + i * td_->rUbFactorAlign, sVal, srcK);
    }
}

// Three-pass Pass 3: Sub+Exp+Div+Cast per row.
__aicore__ inline void DoSubExpDivCast(int64_t validARows)
{
    __ubuf__ float* srcPtr = GetComputePtr();
    __ubuf__ T* dstPtr = (__ubuf__ T*)preInBuf_.Get<T>().GetPhyAddr();
    auto cache = cacheBuf_.Get<float>();
    int64_t sOffset = td_->aUbFactorAlign * MS_STRIDE;
    uint16_t srcK = static_cast<uint16_t>(td_->rUbFactorAlign);
    for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
        float mVal = cache.GetValue(static_cast<uint32_t>(i * MS_STRIDE));
        float sVal = cache.GetValue(static_cast<uint32_t>(sOffset + i * MS_STRIDE));
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<BHSoftmaxSubExpDivCastRowVf<T, VL_F32_U16>>(
            dstPtr + i * td_->rUbFactorAlign,
            srcPtr + i * td_->rUbFactorAlign, mVal, sVal, srcK);
    }
}
```

m/s 块跨度写入与 fast-path 编排：

```cpp
// m/s 写入 cacheBuf：每行一个数据块（MS_STRIDE 个 fp32），首元素为值
__aicore__ inline void StoreReduceToCache(int64_t validARows, int64_t cacheOffElems)
{
    auto src = tmpBuf0_.Get<float>();
    auto dst = cacheBuf_.Get<float>();
    for (int64_t i = 0; i < validARows; ++i) {
        float val = src.GetValue(static_cast<uint32_t>(i));
        AscendC::Duplicate(dst[cacheOffElems + i * MS_STRIDE], val,
                           static_cast<int32_t>(MS_STRIDE));
    }
    for (int64_t i = validARows; i < td_->aUbFactorAlign; ++i) {
        AscendC::Duplicate(dst[cacheOffElems + i * MS_STRIDE], 0.0f,
                           static_cast<int32_t>(MS_STRIDE));
    }
}

// fast path 编排（阶段间全管道屏障）
__aicore__ inline void ProcessFastPath(int64_t aStart, int64_t aLoops)
{
    for (int64_t li = 0; li < aLoops; ++li) {
        int64_t aLoopIdx = aStart + li;
        int64_t validARows = ValidARows(aLoopIdx);

        CopyIn(aLoopIdx, 0, validARows);
        AscendC::PipeBarrier<PIPE_ALL>();
        CastToFp32(validARows);
        AscendC::PipeBarrier<PIPE_ALL>();
        DoReduceMax();
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreReduceToCache(validARows, 0);
        AscendC::PipeBarrier<PIPE_ALL>();
        DoSubExp(validARows);
        AscendC::PipeBarrier<PIPE_ALL>();
        DoReduceSum();
        AscendC::PipeBarrier<PIPE_ALL>();
        StoreReduceToCache(validARows, td_->aUbFactorAlign * MS_STRIDE);
        AscendC::PipeBarrier<PIPE_ALL>();
        DoDivCast(validARows);
        AscendC::PipeBarrier<PIPE_ALL>();
        CopyOut(aLoopIdx, 0, validARows);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
```

（`DoReduceMax/DoReduceSum` 用 5 参显式 sharedTmpBuffer 形式、`CopyIn` 用 DataCopyPad 且 padValue=-inf、`InitMaxCache/InitSumCache/AccumulateMax/AccumulateSum` 支撑三遍路径，结构同上不再展开。）

## 性能影响

未做性能专项测量；结构取舍只按正确性与 UB 占用决策（见设计决策）。逐行 VF 相对整 tile 扁平 VF 省一块预广播缓冲、多 `aUbFactorAlign` 次 `GetValue` 标量读——两者相对代价未测量（来源平台：950；证据等级：无实测，结构论证）。

## 验证信号

结构与验证来自同一次 softmax 分支 kernel 的两轮实现（终版全量重跑 run.sh）：

- 终版结构上板：`29/46 precision PASSED, 46 aclnn-OK, 0 ACLNN_FAIL`——fp32 除两个超长归约用例外全过；特殊值切片（contains_nan/pos_inf/all_neg_inf/all_zero/mixed_inf/scalar/reduce_axis_size_1，三 dtype）全部逐位通过（来源平台：950；证据等级：tool_observed）。
- 结构反证（同轨迹）：破坏任一要素都会回到对应故障——VF 内单元素加载+寄存器广播 → 507035；隐式 sharedTmpBuffer → 507035；PIPE_V 屏障 → elem 0 全错；StoreAlign 缺 DIST_PACK_B32 → fp16/bf16 elem-1 损坏（各故障详见同批专题卡：`asc_vf_call_unaligned_access_507035`、库内 `reduce_work_buf_ub_507035` 与 `b16_duplicate_storealign_plain_vs_cast_dist_pack_b32`、`arch35_aicore_localmembar_507035`）（来源平台：950；证据等级：tool_observed）。
- UT 回归：tiling UT 51/51 不退化（来源平台：950；证据等级：tool_observed）。

## 失效边界

- fp16/bf16 的行内中段精度缺口（elem 3/4 级，阈值 2^-10/2^-7 边界）在本结构下未解决（轨迹边界处仍在迭代）。
- 超长归约（R≈131072/2097152）三遍路径的 fp32/bf16 精度失败未解决。
- 跨核行归约、非 b16 窄类型、双输入逐元素算子未验证。

**未决项**：

- fp16/bf16 残余精度（elem 3/4）与三遍路径超长归约精度——接续方向：对照 oracle 的舍入路径（CAST_RINT vs 逐位）与三遍路径的 m/s 累计误差（`AccumulateMax/AccumulateSum` 的标量链）。
- `GetValue` 逐行标量读在 `aUbFactorAlign` 大时的性能代价未测量；如成瓶颈可回到预广播结构（正确性等价，多一块 UB）。
- `SetScheduleMode/SyncAll` 对三遍路径的影响做过一次移除实验（无改善），未深究。
## 坏实践

- 把 SDK `SimpleSoftMax` 的寄存器间广播照搬进 `asc_vf_call` 的用户 VF 而不带块跨度对齐（SDK 那套在直接 `__simd_vf__` 调用语境才安全）。
- m/s 紧凑存储（stride 1）后用 `mAddr + i` 加载——非对齐且与 `Duplicate` 块填充语义不符。
- 阶段间用 `PIPE_V` 屏障省钱——`GetValue` 在 Scalar 管道，跨管道可见性不成立。

## 误诊

- 开发中曾按"in-place Cast 覆盖源数据"方向排查 fp16 损坏（推演后排除，实际是 DIST_PACK_B32 缺失）；曾按"SyncAll 核间同步"方向排查大 shape 崩溃（实际是 tiling 超 UB 与隐式 tmpBuffer）。
