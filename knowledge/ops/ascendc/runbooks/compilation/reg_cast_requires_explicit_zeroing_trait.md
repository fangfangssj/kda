---
type: "Runbook"
title: "Reg Cast 默认 UNKNOWN mask 被拒 — 显式 ZEROING"
description: "dav-3510 Reg::Cast 的默认 CastTrait 使用 MaskMergeMode::UNKNOWN，设备实现静态断言只接受 ZEROING 或 MERGING，需传显式 trait。R4（CANN 9.0.0-beta.2/dav_c310）：窄化方向 f322f16 的底层 vcvt 还要求 SatMode/RegLayout 映射合法——第 5 参只接受 RS_DISABLE/RS_ENABLE、第 6 参只接受 PART_EVEN/PART_ODD；模式枚举（CMPMODE/RoundMode/SatMode）在 AscendC 顶层而非 AscendC::Reg。"
tags: ["task.debugging", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-09-17T03:25:37Z"
platforms: ["950"]
---

# Reg Cast 默认 UNKNOWN mask 被拒 — 显式 ZEROING

## 现象

在 Ascend950 / dav-3510 的 `__simd_vf__` 中，把低精度升 fp32 和 fp32 降回低精度的 `Reg::Cast` 改成省略第三个 trait 模板参数后，远程构建进入设备实现并触发静态断言：

```text
static assertion failed due to requirement
'SupportEnum<(AscendC::Reg::MaskMergeMode)0,
MaskMergeMode::ZEROING, MaskMergeMode::MERGING>()':
current Cast api only supported Mode ZEROING/MERGING on current device!
```

安装头文件中的默认 `castTrait` 把 `mrgMode` 初始化为 `MaskMergeMode::UNKNOWN`；这个默认值能参与模板匹配，却不能通过 dav-3510 实现层的能力约束。

## 适用场景

- 工程形态：Ascend950 kernel 直调，使用 Reg API 做寄存器级类型转换；
- API 路径：`AscendC::Reg::Cast<T, U>` 默认 trait，或 `AscendC::Reg::Cast<T, U, trait>` 显式 trait；
- 验证过的平台与 CANN 版本：950（Ascend950PR_957b / dav-3510）+ CANN 9.0.1；
- 适用粒度：`platform_wide`，适用于当前设备实现要求显式 mask merge mode 的 Reg Cast；
- 路线优先级：`recommended_fastpath`，widen/narrow 均显式选择 `ZEROING`，narrow 同时保留预期舍入模式；
- 明确不适用 / 未验证场景：若算法要求未命中 lane 保留旧寄存器值，应评估 `MERGING` 而不是机械使用 `ZEROING`；未验证其他 CANN/架构的默认 trait。

## 根因

CANN 9.0.1 的默认 `castTrait` 将 `MaskMergeMode` 置为 `UNKNOWN`，而 dav-3510 的 `Reg::Cast` 实现静态约束只接受 `ZEROING` 或 `MERGING`。

## 解决配方

不要依赖默认 `castTrait`；为 widening 和 narrowing 分别构造显式 `CastTrait`，至少把 `mrgMode` 设为目标语义对应的 `ZEROING` 或 `MERGING`，并把 trait 作为第三个模板参数传入所有低精度转换调用。

### before

代码证据：revision 为本开发会话 中“移除显式 trait 参数”后的完整重建快照；path `operators/rms_norm/rms_norm.asc`；symbol `rms_norm_vf<T>`；该函数 verbatim 支撑“所有低精度 Cast 使用默认 trait”。

```cpp
template <typename T>
__simd_vf__ inline void rms_norm_vf(__ubuf__ T *xAddr, __ubuf__ T *gAddr, __ubuf__ T *yAddr,
                                    uint32_t numCol, uint32_t chunkCount, float avgFactor, float epsilon)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    // Cast uses the global default `castTrait` (all UNKNOWN): lossless for the
    // widening T->float path; for the float->T narrowing the hardware default
    // rounding is round-to-nearest, which meets the precision thresholds.
    // A custom CastTrait cannot be a local constexpr (no linkage as non-type
    // template arg) and a namespace-scope one breaks the host compile pass.
    AscendC::Reg::RegTensor<float> xFReg, gFReg;
    AscendC::Reg::RegTensor<float> accReg, sqReg, partReg, rstdReg, rstdBcast, oneReg, t1Reg, t2Reg;

    // UpdateMask takes a non-const uint32_t&; use mutable locals (cf. mul.asc).
    uint32_t fullCount = vectorLength;
    uint32_t oneCount = 1;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(fullCount);
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneCount);

    // ---- Pass 1: Σx² via per-chunk Reduce<SUM> + Add accumulator ----
    AscendC::Reg::Duplicate<float>(accReg, 0.0f, fullMask);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = numCol - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xFReg, xAddr + offset);
        } else {
            AscendC::Reg::RegTensor<T> xReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xReg, xAddr + offset);
            AscendC::Reg::Cast<float, T>(xFReg, xReg, mask);
        }
        AscendC::Reg::Mul<float>(sqReg, xFReg, xFReg, mask);                         // x^2
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float>(partReg, sqReg, mask);  // chunk sum -> lane0
        AscendC::Reg::Add<float>(accReg, accReg, partReg, fullMask);                 // accumulate lane0
    }

    // ---- scalar finish: rstd = 1 / sqrt(avgFactor * Σx² + eps) ----
    AscendC::Reg::Muls<float>(accReg, accReg, avgFactor, scalarMask);               // * (1/D)
    AscendC::Reg::Adds<float>(accReg, accReg, epsilon, scalarMask);                 // + eps
    AscendC::Reg::Sqrt<float>(sqReg, accReg, scalarMask);                            // sqrt(meanSq + eps)
    AscendC::Reg::Duplicate<float>(oneReg, 1.0f, fullMask);
    AscendC::Reg::Div<float>(rstdReg, oneReg, sqReg, scalarMask);                   // rstd = 1 / sqrt(...)
    AscendC::Reg::Duplicate<float>(rstdBcast, rstdReg, fullMask);                   // broadcast lane0

    // ---- Pass 2: y = x * rstd * gamma ----
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = numCol - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xFReg, xAddr + offset);
        } else {
            AscendC::Reg::RegTensor<T> xReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xReg, xAddr + offset);
            AscendC::Reg::Cast<float, T>(xFReg, xReg, mask);
        }
        AscendC::Reg::Mul<float>(t1Reg, xFReg, rstdBcast, mask);                    // x * rstd
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(gFReg, gAddr + offset);
        } else {
            AscendC::Reg::RegTensor<T> gReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gReg, gAddr + offset);
            AscendC::Reg::Cast<float, T>(gFReg, gReg, mask);
        }
        AscendC::Reg::Mul<float>(t2Reg, t1Reg, gFReg, mask);                        // x * rstd * gamma
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, t2Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> yReg;
            AscendC::Reg::Cast<T, float>(yReg, t2Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, yReg, mask);
        }
    }
}
```

### after

代码证据：revision 为最终成功构建前的轨迹编辑快照（随后以短提交 `5effe87` 提交；轨迹未提供 40 位 revision 与 blob URL）；path `operators/rms_norm/rms_norm.asc`；symbols `rms_cast_traits::{castWiden,castNarrow}`（）与 `rms_norm_vf<T>`（）；以下声明和完整函数 verbatim 支撑“显式 ZEROING trait 替代默认 UNKNOWN”。

```cpp
// CastTrait constants must have linkage (they are non-type template args of
// Reg::Cast) AND must only be visible in the device compile pass (the host pass
// of this single-file .asc does not expose the reg_compute headers, so a
// namespace-scope CastTrait definition would break it). Guard with the device
// compile macro. Cast requires an explicit MaskMergeMode (UNKNOWN is rejected).
#if defined(__CCE_AICORE__) || defined(__KERNEL__) || defined(__ASM__) || \
    defined(__KERNEL_MODE__) || defined(__CCE_KERNEL__) || defined(__AICORE__)
namespace rms_cast_traits {
constexpr AscendC::Reg::CastTrait castWiden = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait castNarrow = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
} // namespace rms_cast_traits
#endif
```

```cpp
template <typename T>
__simd_vf__ inline void rms_norm_vf(__ubuf__ T *xAddr, __ubuf__ T *gAddr, __ubuf__ T *yAddr,
                                    uint32_t numCol, uint32_t chunkCount, float avgFactor, float epsilon)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    // Cast traits live in the device-guarded namespace rms_cast_traits above.
    AscendC::Reg::RegTensor<float> xFReg, gFReg;
    AscendC::Reg::RegTensor<float> accReg, sqReg, partReg, rstdReg, rstdBcast, oneReg, t1Reg, t2Reg;

    // UpdateMask takes a non-const uint32_t&; use mutable locals (cf. mul.asc).
    uint32_t fullCount = vectorLength;
    uint32_t oneCount = 1;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(fullCount);
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneCount);

    // ---- Pass 1: Σx² via per-chunk Reduce<SUM> + Add accumulator ----
    AscendC::Reg::Duplicate<float>(accReg, 0.0f, fullMask);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = numCol - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xFReg, xAddr + offset);
        } else {
            AscendC::Reg::RegTensor<T> xReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xReg, xAddr + offset);
            AscendC::Reg::Cast<float, T, rms_cast_traits::castWiden>(xFReg, xReg, mask);
        }
        AscendC::Reg::Mul<float>(sqReg, xFReg, xFReg, mask);                         // x^2
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float>(partReg, sqReg, mask);  // chunk sum -> lane0
        AscendC::Reg::Add<float>(accReg, accReg, partReg, fullMask);                 // accumulate lane0
    }

    // ---- scalar finish: rstd = 1 / sqrt(avgFactor * Σx² + eps) ----
    AscendC::Reg::Muls<float>(accReg, accReg, avgFactor, scalarMask);               // * (1/D)
    AscendC::Reg::Adds<float>(accReg, accReg, epsilon, scalarMask);                 // + eps
    AscendC::Reg::Sqrt<float>(sqReg, accReg, scalarMask);                            // sqrt(meanSq + eps)
    AscendC::Reg::Duplicate<float>(oneReg, 1.0f, fullMask);
    AscendC::Reg::Div<float>(rstdReg, oneReg, sqReg, scalarMask);                   // rstd = 1 / sqrt(...)
    AscendC::Reg::Duplicate<float>(rstdBcast, rstdReg, fullMask);                   // broadcast lane0

    // ---- Pass 2: y = x * rstd * gamma ----
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = numCol - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(xFReg, xAddr + offset);
        } else {
            AscendC::Reg::RegTensor<T> xReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xReg, xAddr + offset);
            AscendC::Reg::Cast<float, T, rms_cast_traits::castWiden>(xFReg, xReg, mask);
        }
        AscendC::Reg::Mul<float>(t1Reg, xFReg, rstdBcast, mask);                    // x * rstd
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(gFReg, gAddr + offset);
        } else {
            AscendC::Reg::RegTensor<T> gReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gReg, gAddr + offset);
            AscendC::Reg::Cast<float, T, rms_cast_traits::castWiden>(gFReg, gReg, mask);
        }
        AscendC::Reg::Mul<float>(t2Reg, t1Reg, gFReg, mask);                        // x * rstd * gamma
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + offset, t2Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> yReg;
            AscendC::Reg::Cast<T, float, rms_cast_traits::castNarrow>(yReg, t2Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, yReg, mask);
        }
    }
}
```

## 设计决策

- widening 选择 `{ZERO, UNKNOWN, ZEROING, UNKNOWN}`：转换本身无损，明确 mask 未命中 lane 置零。
- narrowing 选择 `{ZERO, NO_SAT, ZEROING, CAST_RINT}`：明确非饱和、置零与舍入策略，避免把精度语义交给 `UNKNOWN` 默认值。
- 放弃省略 trait 参数：安装头文件虽提供默认值，但设备实现的静态断言明确拒绝其 `UNKNOWN` merge mode。
- 当前方案属于 `recommended_fastpath`；它显式化原本隐含且非法的模板策略。

## 系统信息

- CANN：9.0.1；编译器：`bisheng --asc-aicore-lang`；NPU arch：dav-3510；产品：Ascend950PR_957b。
- 头文件证据：安装的 `kernel_reg_compute_utils.h` 定义默认 `castTrait = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::UNKNOWN, RoundMode::UNKNOWN}`。
- 涉及文件：`operators/rms_norm/rms_norm.asc`；测试文件只用于正常精度验证，未放宽容差或修改 golden。

## 性能影响

本开发会话 未隔离测量显式 trait 与默认 trait 的性能差异；默认 trait 无法编译，因而不存在可比较的运行路线。

## 预防

- 看到 API 有默认 trait 时继续检查目标架构实现层的 `static_assert`，不要把“有默认参数”误当成“默认参数在当前设备合法”。
- 对 mask merge、饱和和舍入这类影响语义的字段全部显式赋值，并让 widening 与 narrowing 分开定义。
- 出现 `SupportEnum<UNKNOWN,...>` 时优先检查 trait 枚举，而不是改 dtype、mask 长度或测试容差。

## 验证信号

来源为一次 Ascend950 Reg kernel 开发会话：省略 trait 后远程构建稳定触发 merge-mode 静态断言，恢复显式 ZEROING trait 后完成构建与上板测试。

- 默认 trait 构建在 widening 与 narrowing 实现中都触发 `current Cast api only supported Mode ZEROING/MERGING`（来源平台：950；证据等级：tool_observed）。
- 显式传入 `rms_cast_traits::castWiden/castNarrow` 后，远程构建输出 `[100%] Built target op_rms_norm`（来源平台：950；证据等级：tool_observed）。
- 干净重建后 NPU pytest 输出 `39 passed`，完整评测集输出 `20/20` 通过，覆盖低精度转换路径（来源平台：950；证据等级：tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3：工具报告 [100%] Built target op_cummin。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `ZEROING` 适合本函数所有活跃 lane 都由 mask 定义、未命中 lane 不应保留旧值的路径；依赖旧寄存器值的算法应选择并验证 `MERGING`。
- narrowing 的 `CAST_RINT` 是本开发会话 验证过的舍入路线；改成其他 RoundMode 必须重新做精度验证。
- A3/A2 未验证，推断可迁移但未验证，不写入 `platforms`。

## 坏实践

- 依赖默认 `castTrait` 的 `UNKNOWN` 字段，直到设备实现层才失败，增加了模板诊断噪音。
- 为让精度用例通过而放宽容差不能修复编译期枚举约束，且属于测试侧放松。

## 误诊

“widening 是无损转换，所以默认 UNKNOWN 应可用”是错误假设：转换数值是否无损与 mask merge mode 是否为设备支持枚举是两个独立契约。

## 后续增量（2026-09-17，并行策展批：R4——f322f16 vcvt 参数断言与枚举命名空间补充）

R4（BesselI1e Branch-1 FP16 上溯 FP32 的 VF 计算链，ascend950/dav_c310 设备实现，CANN 9.0.0-beta.2）：默认 trait 除 merge-mode 断言外，还会在**窄化方向**触发底层 vcvt 的参数断言（由 `SatMode::UNKNOWN`/`RegLayout::UNKNOWN`(-1) 传入 `__VF_VCVTFF_RND_SAT_PART` 引起）：

```text
error: static assertion failed ...: The 5th argument of this vcvt (f322f16) can only be: RS_DISABLE, RS_ENABLE
error: static assertion failed ...: The 6th argument of this vcvt (f322f16) can only be: PART_EVEN, PART_ODD
```

即 f322f16 方向要求 `SatMode` 映射到 RS_DISABLE/RS_ENABLE、`RegLayout` 映射到 PART_EVEN/PART_ODD，UNKNOWN 一律非法（widening 方向走 `__VF_VCVTFF_PART`，无 rnd/sat 断言）。本轨迹验证可用的完整显式 trait（f16↔f32 双向编译通过 + Cast 往返探针数值正确）：

```cpp
constexpr AscendC::Reg::CastTrait castZERO = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE
};
// 调用侧：Cast<float, T, castZERO>(t_x, srcF16, mask); Cast<T, float, castZERO>(dstF16, t_r, mask);
```

配套事实（同轮编译修复）：这些模式枚举（`CMPMODE`/`RoundMode`/`SatMode`）定义在 `AscendC` 顶层命名空间而非 `AscendC::Reg`，`using namespace AscendC::Reg` 不会引入它们——`Compares<float, CMPMODE::LE>` 报 `use of undeclared identifier 'CMPMODE'; did you mean 'AscendC::CMPMODE'`，`AscendC::Reg::RoundMode` 报 `no member named 'RoundMode' in namespace 'AscendC::Reg'`，须写 `AscendC::CMPMODE::LE`、`AscendC::RoundMode::CAST_NONE` 全限定（同族卡：[Reg CastTrait RoundMode 命名空间卡](casttrait_roundmode_is_in_ascendc.md)、[Reg Compare 的 CMPMODE 命名空间卡](reg_compare_cmpmode_is_in_ascendc.md)）。排查路径：报错栈指向 `dav_c310/kernel_reg_compute_vec_vconv_impl.h` 的 `CastImpl` 与 `__clang_cce_vector_intrinsics.h` 宏展开，据此对照 `kernel_reg_compute_utils.h` 的 CastTrait 枚举逐字段补全。

R4 验证信号：

- 修复前：`current Cast api only supported Mode ZEROING/MERGING on current device!`、`The 5th argument of this vcvt (f322f16) can only be: RS_DISABLE, RS_ENABLE`、`The 6th argument ... PART_EVEN, PART_ODD`（编译器输出原文）（来源平台：950；证据等级：tool_observed）。
- 修复后：同一 VF 文件编译通过 INFERCHANNEL 阶段，全量构建成功（来源平台：950；证据等级：tool_observed）。
- 功能探针：CastRoundtripVF（LoadAlign→Cast f16→f32→Cast f32→f16→StoreAlign）输出与输入一致；Cast→Abs 输出 |x|（来源平台：950；证据等级：tool_observed）。

R4 边界补充：CastTrait 聚合初始化按 `{layoutMode, satMode, mrgMode, roundMode}` 声明序，写错顺序会得到难以理解的断言（本例因 `RegLayout::UNKNOWN` 漏改多费一轮编译）；`ZEROING` vs `MERGING`、`CAST_NONE` vs `CAST_RINT` 对掩码外 lane 数值与精度的影响未对比；bf16↔f32 的等价 trait 组合未验证。