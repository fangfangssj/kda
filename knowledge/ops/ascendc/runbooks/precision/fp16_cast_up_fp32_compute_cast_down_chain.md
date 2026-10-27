---
type: "Runbook"
title: "FP16 算子 Cast 提升-计算-回退链设计"
description: "FP16/BF16 输入算子用 Cast(FP16→FP32, CAST_NONE) 提升到 FP32 做 VF 计算、Cast(FP32→FP16, CAST_RINT) 回退输出；AscendC::Cast 在 CANN 9.0.0/Ascend950 须带 RoundMode:: 前缀。ApplyAdamV2 穿刺与 ST 全量验证；增量（GroupNorm，9.1.0）：bf16 不支持 ReduceSum/Sub/Mul 等向量 API、Rsqrt 半精度不足，都靠 fp32 中间类型解决——DtypeTrait + if constexpr 编译期分发、tiling 按实际 dtype 算 dtypeKey、Cast 目标放 TBuf（fp32 视图）不复用 T 队列 tensor。增量（Mish 激活链，9.1.0）：Exp/Log/Tanh/Abs/Muls/Adds/Maxs/Mul 均不支持 bfloat16_t，BF16 elementwise 整块 Cast 升 fp32 计算再 Cast 回——队列 buffer 保持 bf16（按 2 字节搬运）、计算区 float、UB 预算按 24 字节/元素记账；RegBase 路线因 Tanh 不在 VF-safe 白名单被否。增量（grid sampler，9.1.0）：标量隐式转换（(float)GetValue / (T)result+Duplicate）按设计替换为 Level-2 Cast 向量 API——升精度 CAST_NONE 经 VECCALC TQue 完成 V→S 同步、降精度 CAST_ROUND 与 Duplicate 同 V pipe 按序；Level-0 Cast/Duplicate 带 __ASC_USE_RESERVED_UBUF__ 注解而 Level-2 无。"
tags: ["task.implementation", "topic.precision", "topic.compute.elementwise.type_conversion", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-17T02:27:00Z"
updated_at: "2026-09-17T04:15:00Z"
platforms: ["950"]
---

# FP16 算子 Cast 提升-计算-回退链设计

## 现象

在 Ascend950 上为 ApplyAdamV2 实现 FP16 dtype 支持。FP16 直接做 VF（向量函数）计算会因动态范围小、累加溢出而失精度；需要把 FP16 输入提升到 FP32 计算、输出再回退 FP16。该模式在穿刺 `probe_adam_fp16` / `probe_mbart_adam_fp16`（NPU Kernel 直调）全通过，ST NPU 2895/2895（含 FP16 全量用例）通过。期间发现 `AscendC::Cast` 在 CANN 9.0.0/Ascend950 上要求 `AscendC::RoundMode::` 前缀枚举值（`CAST_NONE` / `CAST_RINT`），裸用 `AscendC::CAST_NONE` 会编译失败（vendor 缓存陈旧版本时触发）。

## 适用场景

- 工程形态：AscendC 算子 FP16 dtype 实现（kernel 直调与图模式）。
- API 路径：`AscendC::Cast`、`AscendC::Reg::*`（Muls/Add/Sub）、`asc_vf_call`、`AscendC::GetVecLen`。
- 验证平台与版本：Ascend950PR（arch35/DAV_3510），CANN 9.0.0。
- 适用粒度：`operator_family`（适用于所有 FP16 输入、内部用 VF 做浮点运算的算子）。
- 路线优先级：`recommended_fastpath`（FP16 提升到 FP32 计算再回退，是该类算子精度与动态范围的标准做法）。
- 不适用或未验证场景：FP16 直接用 FP16 VF 指令（若硬件有原生 FP16 高精度指令且动态范围足够）不适用；a2/a3 平台未验证 RoundMode:: 前缀要求。

## 完整实现

FP16 路径以 `CopyIn(FP16) → Cast(FP16→FP32, CAST_NONE) → VF(FP32) → Cast(FP32→FP16, CAST_RINT) → CopyOut(FP16)` 为完整计算单元。Cast 提升用 `CAST_NONE`（无损、直接位扩展），Cast 回退用 `CAST_RINT`（就近舍入，避免截断引入偏置）。VF 函数模板用 `ComputeT = std::conditional<IS_FP16, float, T>` 在 FP16 路径下实例化为 float，复用同一份 FP32 VF 实现。同步事件 `MTE2_V`（CopyIn→VF）与 `MTE3_MTE2`（上一轮 CopyOut→本轮 CopyIn）保证 buffer 依赖。该实现经穿刺 + ST NPU 全量验证通过。

### 增量（GroupNorm 三 dtype 全通路，CANN 9.1.0）

fp16/bf16 扩展的完整通路：`DataCopyPad(T) → Cast<float,T>(CAST_NONE) → fp32 全程计算 → Cast<T,float>(CAST_RINT) → DataCopyPad(T)`，tiling 从实际 dtype 计算 dtypeKey 选模板。为什么中间计算统一 fp32：bf16（bfloat16_t）不支持 ReduceSum/Sub/Mul/Adds/Muls/Rsqrt 等向量计算 API；fp16 虽有部分支持，但归约/方差对精度敏感且 Rsqrt 的 half 精度不足。统一 fp32 后三 dtype 共享同一套计算代码（模板只分支"搬运与 Cast"两端），维护面最小。

**dtype → 模板参数映射与编译选择**（`op_kernel/arch35/*_tiling_key.h`）：

```cpp
ASCENDC_TPL_ARGS_DECL(BenchNoHarnessGroupNorm,
    ASCENDC_TPL_UINT_DECL(DTYPE, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2),
    ASCENDC_TPL_UINT_DECL(MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);

// 全 dtype (fp32/fp16/bf16) + fullload/colsplit
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, 0, 1, 2),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);
```

**tiling 从实际 dtype 计算 dtypeKey**（模板选择权收敛在 tiling 一处，UT 可枚举验证）：

```cpp
    // 根据 dtype 计算 dtypeKey (0=fp32, 1=fp16, 2=bf16)
    uint32_t dtypeKey = (dtype == ge::DT_FLOAT) ? 0 : (dtype == ge::DT_FLOAT16) ? 1 : 2;
```

**kernel 侧 DtypeTrait + 搬入即 Cast**（`op_kernel/arch35/*_tiling_key.h` 同目录实现头）：

```cpp
// DTYPE → C++ 类型映射
template <uint32_t DTYPE>
struct DtypeTrait;
template <>
struct DtypeTrait<0> { using Type = float; };
template <>
struct DtypeTrait<1> { using Type = half; };
template <>
struct DtypeTrait<2> { using Type = bfloat16_t; };

template <uint32_t DTYPE, uint32_t MODE>
class KernelFullLoad {
public:
    using T = typename DtypeTrait<DTYPE>::Type;
    using CalcT = float; // 中间计算统一用 fp32
    ...
    // fp16/bf16 专用: Cast 结果 buffer (T→fp32 输入, fp32→T 输出时复用)
    TBuf<QuePosition::VECCALC> xCalcBuf;
    TBuf<QuePosition::VECCALC> gammaCalcBuf;
    TBuf<QuePosition::VECCALC> betaCalcBuf;
```

**ProcessOneGroup 的 Cast 通路**（核心三段，verbatim）：

```cpp
    // ===== Step 0: 搬入 x 数据 (GM → UB) 并转为 fp32 =====
    LocalTensor<float> xLocal;
    if constexpr (DTYPE == 0) {
        // fp32: 直接搬入 float
        xLocal = xQueue.template AllocTensor<float>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(groupSize_ * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
        xQueue.EnQue(xLocal);
        xLocal = xQueue.template DeQue<float>();
    } else {
        // fp16/bf16: 搬入 T 类型, 再 Cast 到 fp32
        LocalTensor<T> xInput = xQueue.template AllocTensor<T>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(groupSize_ * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xInput, xGm[gmOffset], copyParams, padParams);
        xQueue.EnQue(xInput);
        xInput = xQueue.template DeQue<T>();

        xLocal = xCalcBuf.Get<float>();
        Cast<float, T>(xLocal, xInput, RoundMode::CAST_NONE, static_cast<int32_t>(groupSize_));
        xQueue.FreeTensor(xInput);
    }

    // Step 1-6: fp32 全程计算 (ReduceSum 均值 → Adds 中心化 → Mul 平方
    //           → ReduceSum 方差 → 标量 rstd → Muls 归一化)
    ReduceSum<float>(meanVarLocal, xLocal, tmpLocal, static_cast<int32_t>(groupSize_));
    float mean = meanVarLocal.GetValue(0) * invGroupSize_;
    Adds<float>(workLocal, xLocal, static_cast<float>(-mean), static_cast<int32_t>(groupSize_));
    Mul<float>(workLocal, workLocal, workLocal, static_cast<int32_t>(groupSize_));
    ReduceSum<float>(meanVarLocal, workLocal, tmpLocal, static_cast<int32_t>(groupSize_));
    float var = meanVarLocal.GetValue(0) * invGroupSize_;
    // Rsqrt 在 fp32 标量上计算 (half 精度不足)
    float rstd = 1.0f / sqrt(var + epsilon_);
    Adds<float>(workLocal, xLocal, static_cast<float>(-mean), static_cast<int32_t>(groupSize_));
    Muls<float>(workLocal, workLocal, rstd, static_cast<int32_t>(groupSize_));

    // ===== Step 9: 搬出 y 数据 (UB → GM) =====
    if constexpr (DTYPE == 0) {
        // fp32: 直接搬出 float
        yQueue.EnQue(yLocal);
        yLocal = yQueue.template DeQue<float>();
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(groupSize_ * sizeof(float));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(yGm[gmOffset], yLocal, outParams);
        yQueue.FreeTensor(yLocal);
    } else {
        // fp16/bf16: Cast fp32→T, 再搬出
        LocalTensor<T> yOutput = yQueue.template AllocTensor<T>();
        Cast<T, float>(yOutput, yLocal, RoundMode::CAST_RINT, static_cast<int32_t>(groupSize_));
        yQueue.EnQue(yOutput);
        yOutput = yQueue.template DeQue<T>();
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(groupSize_ * sizeof(T));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(yGm[gmOffset], yOutput, outParams);
        yQueue.FreeTensor(yOutput);
    }
```

（分载模式 KernelColSplit 在三个遍历的 chunk 搬入点同样插 `Cast<float, T>(CAST_NONE)`、在遍历 3 写回前 `Cast<T, float>(CAST_RINT)`，结构与全载一致。）

增量设计要点：Cast 结果放 TBuf（xCalcBuf 等）而不是复用队列 tensor——队列 tensor 类型是 T（2B），Cast 目标是 fp32（4B），尺寸/类型都不匹配；TBuf 直接 `Get<float>()` 且可跨步骤复用。`Cast` 必须显式写双模板参数 `Cast<dst, src>`（参考 CANN 内置 cast 算子写法）。

### 增量（Mish 激活链，独立 trace，CANN 9.1.0）

arch35 向量数学 API 的 bf16 缺位补充口径（官方 API 文档"参数说明-模板参数 T"逐个确认）：Exp/Tanh/Log/Abs 为 half、float；Mul/Adds/Muls/Maxs 为 half、int16_t、int32_t、float——**均不含 bfloat16_t**；Cast 是唯一被明确声明支持 bfloat16_t↔float 的通路（bfloat16_t→float 用 CAST_NONE 无精度损失；float→bfloat16_t 用 CAST_RINT，round-to-nearest-even）。BF16 elementwise 超越函数算子（Mish = x·tanh(softplus(x))）据此整块升精度：

**模板类编译期分流**（队列 buffer 保持 bfloat16_t——GM 侧数据按 2 字节原生 dtype 搬运最小带宽，升精度只发生在 UB 内；计算 buffer 全部 float）：

```cpp
template <typename D_T_X>
class BenchNoHarnessMish {
    // BF16 输入需 Cast 到 FP32 计算后再 Cast 回 BF16
    static constexpr bool IS_CAST = std::is_same_v<D_T_X, bfloat16_t>;
    // 实际计算类型：BF16 提升至 FP32，其余保持原类型
    using ComputeT = typename std::conditional_t<IS_CAST, float, D_T_X>;
    static constexpr int32_t BUFFER_NUM = 2;  // double buffer
```

**Init 里 BF16 分支单独记账 buffer**（队列按 bfloat16_t、计算区按 float）：

```cpp
    // Buffer 初始化（按是否 Cast 分支）
    if constexpr (IS_CAST) {
        // BF16 路径：inQueue/outQueue 为 BF16，计算 buffer 为 FP32
        pipe.InitBuffer(inQueue, BUFFER_NUM, ubFormer_ * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubFormer_ * sizeof(bfloat16_t));
        pipe.InitBuffer(bufXF32, ubFormer_ * sizeof(float));
        pipe.InitBuffer(tmpBuf1, ubFormer_ * sizeof(float));
        pipe.InitBuffer(tmpBuf2, ubFormer_ * sizeof(float));
        pipe.InitBuffer(tmpBuf3, ubFormer_ * sizeof(float));
    } else {
        // FP16/FP32 路径：所有 buffer 同类型
        pipe.InitBuffer(inQueue, BUFFER_NUM, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(tmpBuf1, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(tmpBuf2, ubFormer_ * sizeof(D_T_X));
        pipe.InitBuffer(tmpBuf3, ubFormer_ * sizeof(D_T_X));
    }
```

**Compute 的 BF16 分支（进 fp32 → 计算链 → 出 bf16）**：

```cpp
    if constexpr (IS_CAST) {
        // BF16 路径：Cast → FP32 计算 → Cast 回 BF16
        LocalTensor<float> xF32 = bufXF32.Get<float>();
        LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuf2.Get<float>();
        LocalTensor<float> tmp3 = tmpBuf3.Get<float>();
        // BF16 → FP32（CAST_NONE 无精度损失）
        Cast<float, bfloat16_t>(xF32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        // FP32 上执行计算链，结果写入 tmp1（dst 与 src 不重叠）
        MishCompute(xF32, tmp1, tmp1, tmp2, tmp3, sharedTmp, count);
        // FP32 → BF16（CAST_RINT 四舍六入五成双）
        Cast<bfloat16_t, float>(yLocal, tmp1, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
    } else {
        // FP16/FP32 路径：直接计算，结果写入 yLocal
        LocalTensor<ComputeT> tmp1 = tmpBuf1.template Get<ComputeT>();
        LocalTensor<ComputeT> tmp2 = tmpBuf2.template Get<ComputeT>();
        LocalTensor<ComputeT> tmp3 = tmpBuf3.template Get<ComputeT>();
        MishCompute(xLocal, yLocal, tmp1, tmp2, tmp3, sharedTmp, count);
    }
```

**host 侧 tiling 的 UB 字节预算必须按 BF16 路径单独记账**（本例每元素 24 字节：inQueue 2×2 + outQueue 2×2 + bufXF32 4 + 3×tmpBuf 3×4）：

```cpp
        case ge::DT_BF16:
            // inQueue(2*2) + outQueue(2*2) + bufXF32(4) + 3*tmpBuf(3*4) = 24; tmp=4; align=256/2=128
            return {24U, 4U, 128U};
```

路线取舍：RegBase（寄存器级）路线被否——Tanh 不在 VF-safe unary reg compute 白名单（白名单只有 Abs/Relu/Exp/Sqrt/Ln/Log/Log2/Log10/Neg/Not），寄存器内无法完成完整链路；对 memory-bound 的 elementwise 算子收益不抵复杂度。

## 完整代码

`op_kernel/arch35/apply_adam_v2_kernel.h` 主循环 FP16 路径（verbatim，含 Cast 提升、VF 链、Cast 回退）：

```cpp
        // FP16 path: use FP32 VL for VF function calls (after Cast to FP32)
        constexpr uint32_t VL_FP32 = AscendC::GetVecLen() / sizeof(float);
        constexpr bool IS_FP16 = (sizeof(T) == 2);

        int64_t coord[8] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t a_i_seg = GetUBSplitRange(flat % td_->split.a_o, td_->split.a_o,
                                              td_->split.a_i, td_->split.a_i_tail);
            int64_t count = a_i_seg * inner_count;
            FlatToEffectiveCoord(flat, td_->max_bro_shape, RANK,
                                 td_->split.axis, td_->split.a_i, td_->split.a_o, coord);

            // === Sync: MTE3_MTE2 (non-first iteration) ===
            if (flat != start) AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            // === CopyIn: 4 element tensors → UB (MTE2) ===
            CopyInBrc(coord, IN_GRAD, B0, a_i_seg);
            CopyInBrc(coord, IN_M, B1, a_i_seg);
            CopyInBrc(coord, IN_V, B2, a_i_seg);
            CopyInBrc(coord, IN_VAR, B3, a_i_seg);

            // === Sync: MTE2_V (RAW) ===
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);

            // === FP16 path: Cast FP16→FP32 (in-place, 4 buffers) ===
            if constexpr (IS_FP16) {
                AscendC::Cast(buf_[B0].Get<float>(), buf_[B0].Get<T>(),
                              AscendC::RoundMode::CAST_NONE, count);
                AscendC::Cast(buf_[B1].Get<float>(), buf_[B1].Get<T>(),
                              AscendC::RoundMode::CAST_NONE, count);
                AscendC::Cast(buf_[B2].Get<float>(), buf_[B2].Get<T>(),
                              AscendC::RoundMode::CAST_NONE, count);
                AscendC::Cast(buf_[B3].Get<float>(), buf_[B3].Get<T>(),
                              AscendC::RoundMode::CAST_NONE, count);
            }

            // === VF chain: Moments → Div → Update ===
            // Use FP32 type for VF calls in FP16 path (after Cast)
            using ComputeT = typename std::conditional<IS_FP16, float, T>::type;
            uint16_t repCount = AscendC::CeilDivision(count, VL_FP32);

            // Moments VF: adam or mbart_adam
            if (td_->adam_mode_flag == 0) {
                // adam mode: AdamMomentsVF (with gradient clipping)
                asc_vf_call<AdamMomentsVF<ComputeT>>(
                    (__ubuf__ ComputeT*)buf_[B0].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B1].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B2].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B3].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B4].Get<ComputeT>().GetPhyAddr(),
                    count, VL_FP32, repCount,
                    clipCoeff, beta1, beta2);
            } else {
                // mbart_adam mode: MbartMomentsVF (no gradient clipping)
                asc_vf_call<MbartMomentsVF<ComputeT>>(
                    (__ubuf__ ComputeT*)buf_[B0].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B1].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B2].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B3].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B4].Get<ComputeT>().GetPhyAddr(),
                    count, VL_FP32, repCount,
                    beta1, beta2);
            }

            // Div VF: shared between both modes
            asc_vf_call<AdamDivVF<ComputeT>>(
                (__ubuf__ ComputeT*)buf_[B0].Get<ComputeT>().GetPhyAddr(),
                (__ubuf__ ComputeT*)buf_[B2].Get<ComputeT>().GetPhyAddr(),
                (__ubuf__ ComputeT*)buf_[B4].Get<ComputeT>().GetPhyAddr(),
                count, VL_FP32, repCount,
                epsilon);

            // Update VF: adam or mbart_adam
            if (td_->adam_mode_flag == 0) {
                // adam mode: AdamUpdateVF (weight_decay * var, lr * u_final)
                asc_vf_call<AdamUpdateVF<ComputeT>>(
                    (__ubuf__ ComputeT*)buf_[B0].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B2].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B3].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B4].Get<ComputeT>().GetPhyAddr(),
                    count, VL_FP32, repCount,
                    weightDecay, lr);
            } else {
                // mbart_adam mode: MbartUpdateVF (step_size * u, weight_decay * lr * var)
                asc_vf_call<MbartUpdateVF<ComputeT>>(
                    (__ubuf__ ComputeT*)buf_[B0].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B2].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B3].Get<ComputeT>().GetPhyAddr(),
                    (__ubuf__ ComputeT*)buf_[B4].Get<ComputeT>().GetPhyAddr(),
                    count, VL_FP32, repCount,
                    weightDecay, stepSize, lr);
            }

            // === FP16 path: Cast FP32→FP16 (output buffers only: B0, B4, B2) ===
            if constexpr (IS_FP16) {
                AscendC::Cast(buf_[B0].Get<T>(), buf_[B0].Get<float>(),
                              AscendC::RoundMode::CAST_RINT, count);
                AscendC::Cast(buf_[B4].Get<T>(), buf_[B4].Get<float>(),
                              AscendC::RoundMode::CAST_RINT, count);
                AscendC::Cast(buf_[B2].Get<T>(), buf_[B2].Get<float>(),
                              AscendC::RoundMode::CAST_RINT, count);
            }
```

## 系统信息

- 工具链：CANN 9.0.0，bisheng clang 15.0.5，`asc_op_compile_base` 编译器。
- 目标平台：Ascend950PR（arch35 / DAV_3510）。
- 目录布局：`op_kernel/arch35/apply_adam_v2_kernel.h`（kernel 主体）、`apply_adam_v2_tiling_struct.h`、`apply_adam_v2_struct.h`、`kernel_operator.h`。
- 相关文件清单（与事件流读写一一对应）：`op_kernel/arch35/apply_adam_v2_kernel.h`、`op_host/arch35/apply_adam_v2_tiling_arch35.cpp`、`op_api/aclnn_apply_adam_v2.cpp`、`probe/probe_adam_fp16/op_kernel/apply_adam_v2_kernel.asc`、`probe/probe_mbart_adam_fp16/op_kernel/apply_adam_v2_kernel.asc`。
- RoundMode 枚举定义位置：`/usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_mode.h`（`enum class RoundMode : uint8_t`）。

## 设计决策

- 为何选"提升到 FP32 计算-回退 FP16"而非"全程 FP16 VF"：Adam 优化器有 `m_t = β1·m + (1-β1)·g`、`v_t = β2·v + (1-β2)·g²` 等累加与平方运算，FP16 动态范围小、累加易溢出且精度损失大；FP32 中间计算保证精度，仅输入/输出边界做 FP16 转换，是该类优化器算子的标准做法。
- 为何 Cast 提升用 `CAST_NONE`、回退用 `CAST_RINT`：`CAST_NONE` 在 FP16→FP32 是无损位扩展（不引入误差）；`CAST_RINT` 在 FP32→FP16 做就近舍入（round to nearest even），比截断 `CAST_TRUNC` 引入更小偏置。若回退用截断，大数会被系统性偏小。
- 为何 VF 函数用 `ComputeT = std::conditional<IS_FP16, float, T>` 模板：FP32 与 FP16 路径复用同一份 VF 实现，仅在 buffer 类型与是否 Cast 上分叉，避免维护两份算法代码导致公式不一致。
- 被放弃的设计：曾考虑 FP16 直接用 FP16 VF 指令并做 Kahan 补偿；因复杂度高、动态范围仍不足、收益不明确而放弃。
- 当前方案属哪档路线：`recommended_fastpath`（FP32 中间计算 + 边界 Cast）。

## 验证信号

穿刺 `probe_adam_fp16` / `probe_mbart_adam_fp16`（NPU Kernel 直调，`<<<>>>` 语法）全通过；ST NPU 含 FP16 全量用例 2895/2895 通过（双千双万 rtol=1e-3, atol=1e-4）。（来源平台：950；证据等级: tool_observed）

`AscendC::Cast` 在 CANN 9.0.0/Ascend950 要求 `AscendC::RoundMode::` 前缀；vendor 缓存陈旧版本（裸 `AscendC::CAST_NONE`）触发编译错误，源仓正确版本带前缀，重新编译部署后通过。（来源平台：950；证据等级: tool_observed）

增量信号（GroupNorm fp16/bf16 扩展，Ascend950PR + CANN 9.1.0，实现合入后编译、UT、Mock ST 与全量 CSV 回归）：

- 编译：`bash build.sh --soc=ascend950` 成功，产出 3 个 kernel 二进制；`binary_info_config.json` 显示每 dtype 二进制含 6 个模板实例（DTYPE=0/1/2 × MODE=0/1），inputs dtype 分别为 float32/float16/bfloat16（来源平台：950；证据等级：tool_observed）。
- UT：26/26 通过（fp32 全载/分载 + fp16/bf16 tiling 用例无回归）（来源平台：950；证据等级：tool_observed）。
- Mock ST（CPU golden + dtype 量化模拟）：fp32 回归 L0 41/41；golden 自测 15/15（含 fp16/bf16 专项）；全量 CSV 716/716（41 L0 + 25 分载 + 50 dtype/边界 + 600 L1）通过，fp16 用例 MERE≈1.76e-04 < 阈值 9.77e-04、bf16 用例阈值 7.81e-03 内通过（来源平台：950；证据等级：tool_observed）。
- Real 模式 fp32 回归：L0 41/41（NPU 实测，分载 dtype 骨架未引入回归）（来源平台：950；证据等级：tool_observed）。

增量信号（Mish 激活链 BF16 路径，独立 trace，Ascend950PR + CANN 9.1.0，设计期 API 文档核对 → 实现 → 交叉编译）：

- dtype 支持证据：Exp/Tanh 文档模板参数表只列 half、float（A3/A2/Kirin X90/9030 各行一致）（来源平台：950；证据等级：tool_observed）。
- 编译验证：`bash build.sh --soc=ascend950` exit 0；生成 3 个 kernel .o（对应 TilingKey FP16/FP32/BF16），算子包 custom_opp_ubuntu_x86_64.run 生成（来源平台：950；证据等级：tool_observed）。
- host 侧：op_host UT 12/12 通过（含 infershape_bf16_4d 用例 OK）（来源平台：950；证据等级：tool_observed）。
上板运行与 BF16 精度未验证（该会话无 NPU 执行；CPU mock ST 仅覆盖 FP16 golden）——BF16 路径上板精度（CAST_RINT 舍入 vs golden）与性能代价是后续验证优先项。

## 失效边界

- 该 Cast 链模式适用于"FP16 输入、内部浮点运算、FP16 输出"的算子；若算子本身只需整数/位运算（无浮点动态范围问题），不需要提升。
- `RoundMode::` 前缀要求基于 CANN 9.0.0/Ascend950；推断可迁移到 a2/a3（API 命名空间一致），但未验证，推断可迁移。
- `CAST_RINT` 回退在某些需要确定性截断的场景（如量化感知训练对齐）不适用，需按目标精度策略改 `CAST_TRUNC` 或 `CAST_ROUND`。
- （增量）GroupNorm 形态的 Real 模式 fp16/bf16 NPU 精度实测在该开发时段内尚未完成（算子包安装后转入后续验证阶段，该切片验证止步于 Mock + UT + 编译产物）；Mock 量化模拟（输入/输出按 dtype 往返量化）覆盖了大部分精度风险，但不能替代上板实测（未验证，待补充）。
- （增量）两次 Cast（T→fp32→T）带来额外指令与 UB 流量；未做与"原生半精度计算"的成对性能对比（无实测数据，不构成性能结论）。`CAST_RINT` 对 fp32→fp16 的次正规区间行为未做位级专项验证（golden 量化模拟按同型 round-to-nearest-even 对照，误差在阈值内）。
- （增量）int8 等整型 dtype（Cast 舍入模式不同）、输出 dtype 与输入不同的混合精度（本配方输出回原 dtype）不适用。
- （复现增量）Mish 形态的 BF16 路径仅编译级 + UT 验证，上板精度与性能代价未测（无 profiling 数据，不构成可引用结论）；BF16 路径相对 fp16 直算路径多两次整块 Cast 与更大 UB 占用（24 vs 14 字节/元素，压低 ubFormer），采用前自行测量。dtype 支持表绑定 CANN 9.1.0/arch35 文档，其他版本或架构若已支持 bf16 直算（未验证），本通路退化为多余开销。

## 预防

- 写 `AscendC::Cast` 调用时，RoundMode 参数一律写 `AscendC::RoundMode::CAST_*` 全限定形式，不依赖 `using` 把 `RoundMode` 引入当前作用域，避免跨 CANN 版本命名空间变化导致编译失败。
- 部署后若 TTK 报 Cast 相关编译错误，第一查项是 vendor 缓存 (`opp/vendors/<op>_custom/...`) 的 kernel.h 是否陈旧，而非源仓。

### 增量（grid sampler 标量转换 Cast 向量化，独立 trace，CANN 9.1.0）

grid_sampler_3d 类 fp16 kernel 的数据通路把标量隐式转换替换为 Level-2（count 模式）Cast 向量 API，两处同步结构是本增量的关键事实：

1. **grid 路径（half→float，升精度无舍入，CAST_NONE）**：Cast 是 V pipe 写、之后的 `GetValue` 是标量读（S pipe），必须保证 V pipe 完成后再读——用 `TQue<QuePosition::VECCALC, 1>` 走 Alloc→Cast→EnQue→DeQue 的队列同步（DeQue 等待 V pipe 完成）：

```cpp
    float thetaW, thetaH, thetaD;
    if constexpr (sizeof(T) == 2) {  // half
        // Cast(half→float): CAST_NONE 无精度损失
        LocalTensor<float> gridFp32Alloc = gridFp32Queue_.template AllocTensor<float>();
        Cast<float, T>(gridFp32Alloc, gridLocal, RoundMode::CAST_NONE, 3);
        gridQueue_.FreeTensor(gridLocal);
        gridFp32Queue_.EnQue(gridFp32Alloc);
        LocalTensor<float> gridFp32 = gridFp32Queue_.template DeQue<float>();
        thetaW = gridFp32.GetValue(0);  // → W 轴
        thetaH = gridFp32.GetValue(1);  // → H 轴
        thetaD = gridFp32.GetValue(2);  // → D 轴
        gridFp32Queue_.FreeTensor(gridFp32);
    } else {  // float
        thetaW = gridLocal.GetValue(0);
        thetaH = gridLocal.GetValue(1);
        thetaD = gridLocal.GetValue(2);
        gridQueue_.FreeTensor(gridLocal);
    }
```

（替换前：`float thetaW = (float)gridLocal.GetValue(0);` 等三行标量强转。）

2. **结果路径（float→half，降精度需舍入）**：标量 `Duplicate` 写 UB 与后续 `Cast` 读 UB 同在 V pipe 按序执行，无需跨 pipe 同步——`TBuf<TPosition::VECCALC>` 做 FP32 中转缓冲即可（不需要队列）：

```cpp
    if constexpr (sizeof(T) == 2) {  // half
        LocalTensor<float> yFp32 = yFp32Buf_.Get<float>();
        Duplicate<float>(yFp32, result, 1);
        Cast<T, float>(yLocal, yFp32, RoundMode::CAST_ROUND, 1);
    } else {  // float
        T resultVal = (T)result;
        Duplicate<T>(yLocal, resultVal, 1);
    }
```

（替换前：`T resultVal = (T)result; Duplicate<T>(yLocal, resultVal, 1);`。）

3. **Level-0 vs Level-2 API 事实（CANN 9.1.0/DAV_3510 头文件核实）**：`Cast` 的 Level-2 原型为 `void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, const uint32_t count)`（`asc/include/basic_api/kernel_operator_vec_vconv_intf.h`）；Level-0（mask 模式）的 `Cast`/`Duplicate` 带 `__ASC_USE_RESERVED_UBUF__` 注解（`--cce-disable-asc-reserved-ubuf` 编译选项下禁用），Level-2 无该注解——工程 FP32 路径已在用 Level-2 `Duplicate` 且正常运行，Level-2 `Cast` 同样可用（编译+上板验证支持）。
4. **RoundMode 中点语义**：`CAST_RINT`（nearest-even）才与 PyTorch 默认舍入逐位一致，`CAST_ROUND` 是 away-from-zero——两者只在精确 0.5 中点值上不同；本例遵循设计文档用 CAST_ROUND、靠 fp16 阈值（MARE < 10×2^-10）容纳，golden 对中点值逐位敏感时应改用 CAST_RINT（未在本轨迹验证）。
5. **fp16 分支用 `if constexpr (sizeof(T) == 2)`**（kernel 环境不引入 `<type_traits>`）；float 实例不 InitBuffer FP16 专用缓冲，fp32 二进制大小不变。

**复现增量验证信号**（grid_sampler_3d fp16 支持，Ascend950PR + CANN 9.1.0）：

- 编译：`bash build.sh --soc=ascend950 -j8` 成功，4 个 TilingKey 全部生成（2 个 .o 二进制）（来源平台：950；证据等级：tool_observed）。
- 二进制尺寸：fp16 kernel 16744→18496 字节（Cast 代码进入 fp16 实例），fp32 kernel 16744 字节不变（来源平台：950；证据等级：tool_observed）。
- UT 回归：op_host 32/32 通过，无回归（来源平台：950；证据等级：tool_observed）。
- ST Real 上板全量：99/99 通过（59 fp32 + 40 fp16），EXIT_CODE=0；fp16 全局最大 MARE=8.850e-04（阈值 9.766e-04）、全局最大 MERE=8.572e-07；PyTorch 接入路线复测 99/99（来源平台：950；证据等级：tool_observed）。

**复现增量边界**：

- 仅 DAV_3510 + CANN 9.1.0 实测；其他架构的 RoundMode 扩展枚举（CAST_HYBRID/CAST_EVEN/CAST_ZERO 仅 3510/5102/3003）与保留 UB 策略不同，需按架构头文件复核（未验证，推断可迁移）。
- `CAST_ROUND` 与 `CAST_RINT` 在精确中点值上不同：golden 若做逐位比对且输入含中点值，CAST_ROUND 可能失配（本轨迹随机数据 + 阈值评测下未触发）。
- grid 路径的 TQue 同步模板假设 Cast 元素很少（3 个坐标）且缓冲 ≥32B（一个 UB 块）；大段转换同理适用但缓冲需按元素数 ×4 字节规划。
- 本增量不覆盖 Level-0（mask 模式）Cast 与 bf16 路径。
- 坏实践增补：在标量 `GetValue` 前直接读 V pipe `Cast` 的目标缓冲而不经队列/屏障同步——读到旧值（跨 pipe 可见性详见 [VF 归约落 UB 后 GetValue 读取陈旧标量卡](vector_scalar_ub_handoff_requires_explicit_completion.md)）；凭 RoundMode 字面语义猜舍入（"CAST_ROUND 看起来像四舍五入"实际是 away-from-zero）；为 float 实例也初始化 FP16 专用缓冲（浪费 UB）。

## 误诊

vendor 缓存陈旧导致 Cast 编译错误时，曾怀疑源仓 kernel.h 有误；实际源仓已正确带 `RoundMode::` 前缀，错误只在 vendor 部署的旧拷贝。重新编译安装 `cann-ops-nn-custom_linux-x86_64.run` 后修复。
