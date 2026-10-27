---
type: "Runbook"
title: fp32→int32越界NPU饱和与numpy不一致
description: "归约结果 fp32 值 ≥ 2^31 时 NPU Reg::Cast<int32_t>(CAST_TRUNC,NO_SAT) 硬件仍饱和到 INT32_MAX，而 numpy golden 走 x86 cvttss2si integer-indefinite 对一切越界/Inf/NaN 返回 INT32_MIN，逐位 0% 匹配；用 Reg::Compare+Reg::Select 把越界元素改写为 INT32_MIN 对齐 golden。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:56:56Z'
updated_at: '2026-09-17T00:56:56Z'
platforms: ["950"]
---

# fp32→int32越界NPU饱和与numpy不一致

## 现象

euclidean_norm 算子（Reduction 范式，`sqrt(sum(x²))`，fp32 内部累加后按输出 dtype 还原；多分支 tilingKey 设计，arch35 MicroAPI/Reg 向量实现）黑盒全量 636 例中 54 个 int32 用例失败，`precision` 列全部 `0.0%`、`precision_status=FAIL`。失败用例的输入分布高度一致：31 例 x 全 = INT32_MAX(2147483647)、13 例 x 随机全域 [-2147483648, 2147483647]、9 例 x 全 = INT32_MIN、1 例负向单点——即 **x 取极端值时归约结果必然超出 int32 表示范围**；而小值域（如 [-2,10]、[0,1]）的 int32 用例全部通过（157 个 int32 用例中 103 PASS / 54 FAIL）。

对最小失败用例（shape `(5,1,2,1,1,6)`、axes=[0]、x 全 INT32_MIN、keepDims=false）做 dump 三方对比：kernel 输出全为 `2147483647`(INT32_MAX)，golden 全为 `-2147483648`(INT32_MIN)，cast 前 fp32 sqrt 值为 `4.8019195e+09`（= sqrt(5×2^62)，远超 INT32_MAX）。golden 侧 numpy 同时打印 `RuntimeWarning: invalid value encountered in cast`。

## 适用场景

- 工程形态：aclnn ST / TTK 黑盒精度评测中，"fp32 中间累加 + 输出前 Cast 回 int32（CAST_TRUNC 向零截断）"的归约/逐元素算子，且用例数据可触发放大溢出（极端值输入、大 N 累加）。
- API 路径：AscendC arch35 MicroAPI `AscendC::Reg::Cast<int32_t, float, kCastTraitFromFp32Int32>` + `Reg::Compare` / `Reg::Select` / `Reg::Duplicate`；golden 为 CPU numpy `np.trunc(v).astype(np.int32)`。
- 验证过的平台与 CANN 版本：Ascend950PR（950）+ CANN 9.1.0。
- 适用粒度：operator_family（机制适用于一切 "fp32→int32 Cast + numpy golden" 的算子，阈值推导对 int32 通用）。
- 路线优先级：recommended_fastpath（与 numpy golden 逐位对齐的唯一 kernel 侧路径；golden 改 fp64/saturate 语义属规格层变更，不在 kernel 控制范围）。
- 明确不适用/未验证场景：输出 dtype 为 fp16/bf16/fp32（Cast 语义不同）未验证；int64 输出（spec 外）未验证；golden 不用 numpy x86 语义（如 torch on ARM）时溢出值可能不同，须先探针确认 golden 行为再套用。

## 根因

**NPU 向量 Cast 的越界饱和语义与 numpy/x86 cvttss2si 的 integer-indefinite 语义不一致**（单句归因）。

机制链（全部为 dump + numpy 探针实测值）：

1. x 全 INT32_MIN 时 Σx² = 5×2^62，sqrt 的 fp32 值 = 4.8019195e+09 ≥ 2^31=2147483648.0，超出 int32 可表示范围（fp32 可表示的最大合法值是 2147483520.0，其上一位即 2^31）。
2. kernel 路径 `Reg::Cast<int32_t, float, kCastTraitFromFp32Int32>`（CAST_TRUNC + NO_SAT）：硬件对越界值**仍饱和到 INT32_MAX=2147483647**——NO_SAT 并不改变 fp32→int32 正向越界的饱和结果（实测）。
3. golden 路径 `np.trunc(v).astype(np.int32)`：x86 `cvttss2si` 对一切越界/±Inf/NaN 输入返回 **INT32_MIN=-2147483648**（"integer indefinite" 值）。numpy 探针实测：`2147483647.0→-2147483648`（注意 2147483647.0 在 fp32 中已舍入到 2^31，同样越界）、`2147483648.0/3000000000.0/1e38/inf/nan/负向越界 → 全部 -2147483648`。
4. 两个溢出语义相反 → 溢出元素逐位全错 → `precision=0.0%`，且 TTK 对 int32 的 isclose(rtol=1e-4) 也无法容纳 4294967295 的绝对差。

## 解决配方

在输出侧 int32 Cast 后追加"越界检测 + 改写"：`sqrt ≥ 0` 保证只有正向上溢/+Inf 一种越界形态，用 `Reg::Compare<float, CMPMODE::GE>` 与阈值 `2147483648.0f`(2^31) 生成越界掩码，再 `Reg::Select` 把越界元素改写为 INT32_MIN。两处输出路径（normal 的 `PostElewiseVfImpl` 与 group Phase2 的 `Phase2PostElewiseVfImpl`）都要改。

before（`op_kernel/arch35/euclidean_norm.h`，`PostElewiseVfImpl` 的 int32 分支，修复前）：

```cpp
// post-elewise: 读 cache 树根 fp32 → Sqrt → Cast→D_T → outBuf
template <typename DType>
__simd_vf__ inline void PostElewiseVfImpl(
    __ubuf__ float* rootPtr, __ubuf__ DType* outPtr, uint32_t laneN, uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    constexpr bool IsB16 = (sizeof(DType) == 2);
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
        AscendC::Reg::Sqrt(f32Reg, f32Reg, mask);
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
        } else if constexpr (IsB16) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::Cast<DType, float, kCastTraitFromFp32Fp16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<DType, AscendC::Reg::StoreDist::DIST_PACK_B32>(outPtr + off, b16Reg, mask);
        } else {  // int32
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::Cast<int32_t, float, kCastTraitFromFp32Int32>(iReg, f32Reg, mask);
            AscendC::Reg::StoreAlign(outPtr + off, iReg, mask);
        }
    }
}
```

after（同函数 int32 分支，修复后；`Phase2PostElewiseVfImpl` 做完全相同的分支改写）：

```cpp
        } else {  // int32: CAST_TRUNC; out-of-range (sqrt ≥ 2^31 或 +Inf) 须输出 INT32_MIN,
            // 与 golden(numpy cvttss2si "integer indefinite") 逐 bit 对齐 (DESIGN §4.3 int32 bitwise)。
            // sqrt ≥ 0, 故只可能正向上溢 / +Inf；fp32 可表示的最大 < 2^31 值为 2147483520.0,
            // 故 v ≥ 2147483648.0f (== 2^31) 即判溢出。
            AscendC::Reg::RegTensor<float> threshReg;
            AscendC::Reg::Duplicate(threshReg, 2147483648.0f);
            AscendC::Reg::MaskReg ovrMask;
            AscendC::Reg::Compare<float, CMPMODE::GE>(ovrMask, f32Reg, threshReg, mask);
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::Cast<int32_t, float, kCastTraitFromFp32Int32>(iReg, f32Reg, mask);
            AscendC::Reg::RegTensor<int32_t> minReg;
            AscendC::Reg::Duplicate(minReg, static_cast<int32_t>(-2147483647 - 1));
            AscendC::Reg::Select(iReg, minReg, iReg, ovrMask);
            AscendC::Reg::StoreAlign(outPtr + off, iReg, mask);
        }
```

group Phase2 输出路径（`Phase2PostElewiseVfImpl`，与 PostElewise 同型修复，修复后 int32 分支 verbatim）：

```cpp
        } else {  // int32: CAST_TRUNC; 溢出 → INT32_MIN, 与 golden 逐 bit 对齐 (同 PostElewise)
            AscendC::Reg::RegTensor<float> threshReg;
            AscendC::Reg::Duplicate(threshReg, 2147483648.0f);
            AscendC::Reg::MaskReg ovrMask;
            AscendC::Reg::Compare<float, CMPMODE::GE>(ovrMask, f32Reg, threshReg, mask);
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::Cast<int32_t, float, kCastTraitFromFp32Int32>(iReg, f32Reg, mask);
            AscendC::Reg::RegTensor<int32_t> minReg;
            AscendC::Reg::Duplicate(minReg, static_cast<int32_t>(-2147483647 - 1));
            AscendC::Reg::Select(iReg, minReg, iReg, ovrMask);
            AscendC::Reg::StoreAlign(dstPtr + off, iReg, mask);
        }
```

改动后重编译（`./build.sh`，exit 0）并重跑。

## 设计决策

- **为何选 Compare+Select 软件改写而非换 SatMode/CastTrait**：探针实测证明现行 trait（CAST_TRUNC、NO_SAT）在硬件上本就饱和到 INT32_MAX，没有可表达的 "越界→INT32_MIN" 模式；软件掩码改写是唯一能在 Reg 向量层精确复刻 x86 语义的方式。
- **阈值取 2147483648.0f(2^31) 而非 2147483520.0**：fp32 中 2147483647 会舍入为 2^31，故 `v ≥ 2^31` 恰好覆盖全部越界值（含边界舍入），且 `Duplicate` 一个 2 的幂无精度损失；sqrt 结果非负，负向溢出/NaN 分支被问题域排除（sum of squares ≥ 0）。
- **API 选型来源**：`Reg::Compare<float, CMPMODE::GE>(dstMask, src1, src2, srcMask)` 与 `Reg::Select(dst, src0, src1, mask)` 的签名取自 CANN 9.1.0 内置实现源码（`asc/impl/basic_api/dav_3510/kernel_operator_vec_cmpsel_intf_impl.h` 等）与内置算子用法（`abs_complex_dag.h` 中 `Reg::Compare<float, CMPMODE::GT>` + `Reg::Select` 组合、`anti_mx_quant_tail_axis.h` 中 `Reg::Duplicate`）；`CMPMODE::GE` 枚举值与 `Select` 按掩码逐元素选 src0/src1 的语义均经头文件核实。
- **被放弃的方案**：改 golden（加 clip/saturate）或放宽容差——任务约束禁止（不改 golden/不松容差），且会掩盖语义差异；在 host 侧预判溢出改走其他分支——溢出是逐元素数据依赖，host 无法预判。
- **当前方案属哪档路线**：recommended_fastpath（kernel 侧唯一正确闭合路径）。

## 预防

- 写 "fp32 中间精度 + int 输出" 算子时，先跑 numpy 探针确认 golden 的越界 Cast 行为（x86：一律 INT32_MIN），再决定 kernel 是否需要 Compare+Select 对齐。
- 评测侧为 int32 输出补**边界用例**：x 全 INT32_MAX / 全 INT32_MIN / 全域随机（放大后必溢出的值域），避免溢出语义只在黑盒全量里被动暴露。
- 复审所有 `static_cast<int32_t>` / `Reg::Cast<int32_t,...>`：确认输入值域上界（本例 sqrt(Σx²) 上界由输入值域平方放大决定）。

## 验证信号

来源：euclidean_norm 黑盒失败修复会话（Ascend950PR，CANN 9.1.0）——在 TTK 黑盒全量失败用例上 dump 三方对比定位，kernel 侧修复后重编译并重跑代表用例与全量。

- 5 个代表失败用例（覆盖 INT32_MIN×all / INT32_MAX×all / 全域随机、int32 与 int64 axes、keepDims T/F、rank4-8）修复后全部 `100.0% PASS`（来源平台：950；证据等级：tool_observed）。
- 全量 int32 子集 157 例重跑 `157 PASS / 0 FAIL`（含原 54 个失败全部转绿）（来源平台：950；证据等级：tool_observed）。
- 修复后黑盒权威全量重跑 636 例 `PASS 620 / FAIL 16`，int32 输入用例 0 FAIL（余 16 例为 float32 巨型 reduce 与 L2 异常，另行归因）（来源平台：950；证据等级：tool_observed）。
- 白盒 aclnn 回归 53/53 PASSED（全部分支门禁通过，含 int32 CAST_TRUNC 逐位用例），无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 阈值与"只判正向溢出"依赖 `sqrt(Σx²) ≥ 0` 这一算子语义；对可能产出负值/NaN 的 fp32→int32 Cast 场景（如差值、开方负数），须补负向 `Compare LT -2^31` 与 NaN 分支，本卡配方不直接适用。
- golden 换平台（非 x86 cvttss2si 语义，如 ARM saturating cast 返回 INT32_MAX）时，对齐目标值要重探针——本卡对齐的是 numpy/x86 的 INT32_MIN。
- fp16/bf16 输出路径未加同类保护（其 Cast 走 `kCastTraitFromFp32Fp16`，饱和语义未在本开发会话触发失败）；其他 SoC/arch 的 Cast 硬件行为未验证（未验证，推断可迁移但需先做同型 dump 探针）。

**未决项**：

- 复现并追根因到机制层：NO_SAT 模式下 fp32→int32 Cast 仍饱和到 INT32_MAX 的硬件语义（对照 CANN Cast 文档/微架构说明）——本卡只实测了现象未取得文档证据（优先项）。
- int64 axes 传入时 tiling 侧 `axes` 解码路径的越界行为仅被用例覆盖，未单独验证。
## 坏实践

- 见到 int32 输出 0% 匹配就怀疑数据搬运/shape 错误——本例搬运与 shape 全对，差异只在溢出元素的单个值上；不做 dump 对比会走弯路。
- 试图通过放宽容差让 `|INT32_MAX - INT32_MIN| = 4294967295` 的差通过 isclose——不可能且属测试侧放松捷径，掩盖真实语义差异。

## 误诊

- 初始假设 numpy 是"模 2^32 回绕"（4.8e9 mod 2^32 = 506952204，正值）——被 golden 实测 -2147483648 直接否定，正确机制是 x86 integer-indefinite（该假设为分析叙述，已被工具观测推翻）。
- 曾怀疑 NO_SAT 应该不饱和而硬件"违反"了 SatMode——实测 NO_SAT 下仍饱和到 INT32_MAX；NO_SAT 的确切微架构语义未从文档验证，以实测为准。
