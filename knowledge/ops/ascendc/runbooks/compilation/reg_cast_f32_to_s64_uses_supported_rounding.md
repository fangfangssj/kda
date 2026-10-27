---
type: "Runbook"
title: "f32→s64 Cast 使用 CAST_NONE 触发 vcvt 静态断言"
description: "dav-3510 的 f322s64 VCONV 不接受 CAST_NONE；精确整数浮点索引可用 CAST_TRUNC 通过编译。"
tags: ["task.debugging", "topic.compute.elementwise.type_conversion", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# f32→s64 Cast 使用 CAST_NONE 触发 vcvt 静态断言

## 现象

把 fp32 暂存索引转换为 int64 输出时，`AscendC::Reg::Cast<int64_t, float, kCastF32ToI64>` 的 trait 使用了 `AscendC::RoundMode::CAST_NONE`。远端 bisheng 编译器在 `f322s64` 的 VCONV 模板实例化处直接静态断言：

```text
The 4th argument of this vcvt (f322s64) can only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z
```

## 适用场景

- 工程形态：Ascend C kernel 直调构建；
- API 路径：`__simd_vf__` 内使用 `AscendC::Reg::Cast<int64_t, float, trait>`；
- 验证过的平台与 CANN 版本：Ascend950PR_957c / dav-3510，CANN 9.0.1；
- 适用粒度：`platform_wide`，适用于该 SDK 的 f32→s64 RegBase Cast 模板实例化；
- 路线优先级：`recommended_fastpath`，只表示已验证可通过编译，不宣称运行时正确或性能最优；
- 输入前提：fp32 输入是 int64 范围内且 fp32 可精确表示的整数；
- 明确不适用/未验证场景：带小数的浮点值、超出 fp32 精确整数范围的值、A2/A3 与其他 CANN 版本。

## 根因

`CAST_NONE` 在这条 f32→s64 VCONV 实例化中对应不受支持的舍入编码。编译器明确限制该转换的第四个模板参数只能是 `ROUND_R`、`ROUND_A`、`ROUND_F`、`ROUND_C` 或 `ROUND_Z`。因此，即使调用方认为输入是“无损失的整数浮点数”，也不能用 `CAST_NONE` 表达这个前提。

对于轨迹里的精确整数 fp32 索引，`CAST_TRUNC` 对应向零截断：输入没有小数部分时，截断不改变数学整数值，同时满足 f322s64 的模板约束。

## 解决配方

保留完整转换函数，只把 trait 的舍入模式从 `CAST_NONE` 改成 `CAST_TRUNC`。下面是轨迹中的完整 trait 与完整函数 before/after。

### before

代码证据：revision=轨迹修复前快照（源轨迹未绑定 Git revision）；path=`operators/cummin/cummin.asc`；symbols=`kCastF32ToI64`、`cummin_cast_i64_vf`；line=84-93、216-237。

```cpp
// float(64 lanes)->int64(32 lanes): low-half widen. Idx values are exact small
// integers in fp32 (R << 2^24), so CAST_NONE ("无损失时不舍入") needs no rounding.
// Carrying the argmin index as float lets the whole idx path reuse the proven
// fp32 Reg ops (Duplicate/Select/LoadAlign/StoreAlign); see cummin_vf notes.
constexpr static AscendC::Reg::CastTrait kCastF32ToI64 = {
    AscendC::Reg::RegLayout::ZERO,        // float(64 lanes)->int64(32 lanes): low half
    AscendC::Reg::SatMode::NO_SAT,        // non-negative indices fit int64
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,        // exact-integer fp32 -> no rounding
};

// float -> int64 widening pass (chunk = VL_int64 = 32). Reads the low 32 lanes
// of each 64-lane float register (Cast reads lanes of the larger dtype = int64,
// 32), so no high/low split is required. fp32 indices are exact small integers,
// so CAST_NONE needs no rounding.
__simd_vf__ inline void cummin_cast_i64_vf(
    __ubuf__ float *tmpIdxAddr, __ubuf__ int64_t *argminAddr,
    uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL_I64 = AscendC::VECTOR_REG_WIDTH / sizeof(int64_t);  // 32
    AscendC::Reg::RegTensor<float> idxReg;
    AscendC::Reg::RegTensor<int64_t> argReg;
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL_I64;
        uint32_t rem = count - off;
        // LoadAlign<float> reads 64 lanes; we only consume the low 32 via the
        // int64-width Cast mask (sizeof-larger = int64 → 32 lanes).
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(idxReg, tmpIdxAddr + off);
        AscendC::Reg::MaskReg maskI64 = AscendC::Reg::UpdateMask<int64_t>(rem);
        AscendC::Reg::Cast<int64_t, float, kCastF32ToI64>(argReg, idxReg, maskI64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(argminAddr + off, argReg, maskI64);
    }
}
```

### after

代码证据：revision=CAST_TRUNC 修复后快照（源轨迹未绑定 Git revision）；path=`operators/cummin/cummin.asc`；symbols=`kCastF32ToI64`、`cummin_cast_i64_vf`；line=84-95、218-239。

```cpp
// float(64 lanes)->int64(32 lanes): low-half widen. Idx values are exact small
// integers in fp32 (R << 2^24); the f322s64 vcvt rejects CAST_NONE (ROUND::H),
// allowing only RINT/ROUND/FLOOR/CEIL/TRUNC — TRUNC (向零截断) is the canonical
// float->int and is a no-op for exact-integer inputs. Carrying the argmin index
// as float lets the whole idx path reuse the proven fp32 Reg ops
// (Duplicate/Select/LoadAlign/StoreAlign); see cummin_vf notes.
constexpr static AscendC::Reg::CastTrait kCastF32ToI64 = {
    AscendC::Reg::RegLayout::ZERO,        // float(64 lanes)->int64(32 lanes): low half
    AscendC::Reg::SatMode::NO_SAT,        // non-negative indices fit int64
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,       // f322s64 vcvt: only R/A/F/C/Z; exact ints -> no-op
};

// float -> int64 widening pass (chunk = VL_int64 = 32). Reads the low 32 lanes
// of each 64-lane float register (Cast reads lanes of the larger dtype = int64,
// 32), so no high/low split is required. fp32 indices are exact small integers,
// so CAST_NONE needs no rounding.
__simd_vf__ inline void cummin_cast_i64_vf(
    __ubuf__ float *tmpIdxAddr, __ubuf__ int64_t *argminAddr,
    uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL_I64 = AscendC::VECTOR_REG_WIDTH / sizeof(int64_t);  // 32
    AscendC::Reg::RegTensor<float> idxReg;
    AscendC::Reg::RegTensor<int64_t> argReg;
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL_I64;
        uint32_t rem = count - off;
        // LoadAlign<float> reads 64 lanes; we only consume the low 32 via the
        // int64-width Cast mask (sizeof-larger = int64 → 32 lanes).
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(idxReg, tmpIdxAddr + off);
        AscendC::Reg::MaskReg maskI64 = AscendC::Reg::UpdateMask<int64_t>(rem);
        AscendC::Reg::Cast<int64_t, float, kCastF32ToI64>(argReg, idxReg, maskI64);
        AscendC::Reg::StoreAlign<int64_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(argminAddr + off, argReg, maskI64);
    }
}
```

`after` 中函数头部仍提到 `CAST_NONE` 的注释是轨迹真实源码里的历史残留；实际模板实参取自上方 `kCastF32ToI64`，其生效字段已经是 `CAST_TRUNC`。这里不改写注释，以保持代码证据 verbatim。

## 设计决策

- 选择 `CAST_TRUNC`，因为它属于编译器列出的合法模式，且在“输入本来就是精确整数”的前提下不改变数学值；
- 不用其他舍入模式，是为了避免在未来意外出现小数输入时引入四舍五入、向上或向下取整语义；
- 保留 `NO_SAT`，因为本次索引应为非负且处于 int64 范围内；该前提失效时应另行设计饱和与越界策略；
- 本卡只收敛编译契约。轨迹后续的 int64 索引运行结果仍错误，不能据此证明该转换路径在目标硬件上的数值正确性。

## 系统信息

- toolchain：CANN 9.0.1，bisheng / ASC CMake；
- 目标：Ascend950PR_957c，`NpuArch=dav-3510`；
- 相关文件：算子 `.asc` 中的 RegBase fp32 索引转 int64 输出函数。

## 预防

- 为 `Reg::Cast` 选择舍入模式时，以目标 dtype 组合的编译器/头文件约束为准，不把 `CAST_NONE` 当成通用无操作模式；
- 对“浮点中保存整数”的路径显式校验精确表示范围和目标整数范围；
- 把编译闭环与数值正确性闭环分开记录，编译通过后仍需在 NPU 上验证输出。

## 验证信号

来源是一次 Ascend950 Flash kernel 的远端编译：

- 修复前，bisheng 在 f322s64 VCONV 处输出上述静态断言（Ascend950 / CANN 9.0.1，证据等级: tool_observed）；
- 将 trait 改为 `CAST_TRUNC` 后，紧接着的远端构建输出 `[100%] Linking ASC shared library` 与 `[100%] Built target`（Ascend950 / CANN 9.0.1，证据等级: tool_observed）；

本段证据边界止于编译闭环，不支持“运行时转换正确”的结论。

## 失效边界

若 fp32 输入含小数，`CAST_TRUNC` 会向零截断；若整数绝对值超过 fp32 的精确整数范围，源值已经可能失真；若超出 int64 范围且仍使用 `NO_SAT`，结果也不受本卡保证。A2/A3 和其他 CANN 版本需要重新核对支持的转换模式。

## 坏实践

- 因为输入“理论上是整数”就使用 f322s64 不支持的 `CAST_NONE`；
- 看到目标构建成功便宣称 Reg::Cast 的运行时 int64 输出正确；
- 忽略 fp32 精确整数范围和 int64 范围约束。