---
type: "Runbook"
title: "LoadAlign/StoreAlign 源类型须匹配 __ubuf__ 指针"
description: "Reg::LoadAlign<T,dist>/StoreAlign<T,dist> 的首模板参数是源/目的数据类型，必须与 __ubuf__ 指针元素类型完全一致；泛型 __simd_vf__ 模板把源类型写死 float 时 half/bf16 实例化报 no matching function（__ubuf__ half* 无法转 __ubuf__ float*）；LoadDist/StoreDist/CastTrait 均为模板参数，dtype 路径必须用 if constexpr 编译期分派。"
tags: [task.implementation, topic.compilation, topic.compute.elementwise.arithmetic]
status: stable
sources: []
created_at: '2026-09-17T00:51:56Z'
updated_at: '2026-09-17T00:51:56Z'
platforms: ["950"]
---

# LoadAlign/StoreAlign 源类型须匹配 __ubuf__ 指针

## 现象

在 Ascend950（arch35，CANN 9.2.0）上给逐元素算子写 regbase 风格 kernel：一份 `__simd_vf__` 模板函数 `AdamWVF<T>`（T ∈ float/half/bfloat16_t，由 `asc_vf_call<AdamWVF<T>>` 调用，每 dtype 一个编译单元），最小复现版本里对 `__ubuf__ T*` 输入指针直接调用 `AscendC::Reg::LoadAlign<float>` 与 `StoreAlign<float>`。构建 `./build.sh <Op>_package` 时 fp32 实例正常，fp16/bf16 实例化处报错（原文）：

```
<op>_kernel.h:221:9: error: no matching function for call to 'LoadAlign'
        AscendC::Reg::LoadAlign<float>(regVar, varAddr, aReg);
        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<op>_kernel.h:176:21: note: in instantiation of function template specialization 'AdamWVF<half>' requested here
        asc_vf_call<AdamWVF<T>>(
                   ^
.../kernel_reg_compute_datacopy_intf_impl.h:139:29: note: candidate function template not viable:
  no known conversion from '__ubuf__ half *' to '__ubuf__ float *' for 1st argument
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg, AddrReg offset, MaskReg& mask)
```

当时代码状态：VF 函数是 dtype 泛型模板，但 Load/Store 调用把源类型写死为 `float`，没有按 T 做编译期分派。

## 适用场景

- 工程形态：AscendC regbase/MicroAPI kernel（`__simd_vf__` + `asc_vf_call` + `AscendC::Reg::` 向量寄存器 API），一份 VF 模板服务 fp32/fp16/bf16 多 dtype 编译单元（`-DDTYPE_VAR=float/half/bfloat16_t` 由构建系统注入）。
- API 路径：`AscendC::Reg::LoadAlign<T, LoadDist>` / `StoreAlign<T, StoreDist>` / `Cast<T_dst, T_src, CastTrait>`（`asc/include/basic_api/reg_compute/kernel_reg_compute_*_intf.h`）。
- 验证过的平台与 CANN 版本：Ascend950（arch35/DAV_3510），CANN 9.2.0，bisheng（clang 15.0.5）（与 frontmatter 一致）。
- 适用粒度：platform_wide——Reg:: API 的模板类型规则与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：单一 dtype 的 VF（无分派需求）；SIMT 路线（`--enable-simt`，reg_compute 头文件本身禁用该编译开关）；其他 CANN 版本的 Reg:: API 签名（未验证，推断机制相同）；其他位宽 dtype（int8 等）的 LoadDist/StoreDist 取值未验证。

## 根因

`Reg::LoadAlign<T, dist>(dstReg, __ubuf__ T* srcAddr, ...)` / `StoreAlign<T, dist>(__ubuf__ T* dstAddr, ...)` 的第一个模板参数是**源/目的数据类型，必须与 `__ubuf__` 指针元素类型完全一致**；`__ubuf__` 限定指针之间无隐式转换，泛型 VF 模板实例化为 `T=half` 时 `__ubuf__ half*` 无法绑定 `__ubuf__ float*` 形参，重载决议失败报 no matching function。`LoadDist`（fp16/bf16 装载需 `DIST_UNPACK_B16`）、`StoreDist`（`DIST_PACK_B32`）、`CastTrait` 都是模板参数，运行期无法选择——dtype 路径必须在编译期用 `if constexpr`（或模板特化）分派。

## 解决配方

修复前（VF 内不分派，源类型写死 float；fp16/bf16 实例化即失败）：

```cpp
template <typename T>
__simd_vf__ inline void AdamWVF(
    __ubuf__ T* yAddr, __ubuf__ T* varAddr, ...)
{
    AscendC::Reg::RegTensor<float> regVar;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::LoadAlign<float>(regVar, varAddr, aReg);          // T=half 时 no matching
        AscendC::Reg::StoreAlign<float>(yAddr, regVar, aReg, mask);     // 同上
    }
}
```

修复后（`if constexpr` 按 T 编译期分派：fp32 直 load/store；half/bf16 走 `DIST_UNPACK_B16` 装载 + `Cast` 升 fp32 计算 + `Cast` 降回 + `DIST_PACK_B32` 存储）：

```cpp
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(count);
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::LoadAlign<float>(regVar, varAddr, aReg);
            AscendC::Reg::StoreAlign<float>(yAddr, regVar, aReg, mask);
        } else if constexpr (std::is_same_v<T, half>) {
            AscendC::Reg::RegTensor<half> regVarT;
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regVarT, varAddr, aReg);
            AscendC::Reg::Cast<float, half, CT_HALF_TO_FP32>(regVar, regVarT, mask);
            AscendC::Reg::RegTensor<half> regYT;
            AscendC::Reg::Cast<half, float, CT_FP32_TO_HALF>(regYT, regVar, mask);
            AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr, regYT, aReg, mask);
        } else {
            AscendC::Reg::RegTensor<bfloat16_t> regVarT;
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regVarT, varAddr, aReg);
            AscendC::Reg::Cast<float, bfloat16_t, CT_BF16_TO_FP32>(regVar, regVarT, mask);
            AscendC::Reg::RegTensor<bfloat16_t> regYT;
            AscendC::Reg::Cast<bfloat16_t, float, CT_FP32_TO_BF16>(regYT, regVar, mask);
            AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(yAddr, regYT, aReg, mask);
        }
    }
```

配套的 CastTrait 常量（升精度不饱和不舍入，降精度 SAT + CAST_ROUND）：

```cpp
static constexpr AscendC::Reg::CastTrait CT_HALF_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::Reg::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait CT_FP32_TO_HALF = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::Reg::RoundMode::CAST_ROUND};
// bfloat16_t 同型：CT_BF16_TO_FP32 / CT_FP32_TO_BF16
```

## 设计决策

- 用 `if constexpr` 而不是三份非模板 VF 重载：实测 `if constexpr` 在 `__simd_vf__` 内编译通过且三个 dtype 实例均正常（按 dtype 拆三份函数更繁琐且 `asc_vf_call` 调用点还要分派，放弃）。
- 循环内统一用 `CreateAddrReg<T>(i, oneRepeatSize)` 生成 `AddrReg` 偏移而不是 `varAddr + i * oneRepeatSize` 指针算术：指针算术形式未在本轨迹单独验证，AddrReg 形式是最终验证通过的写法，故推荐 AddrReg（同平台其他轨迹的 `inAddr + i * VL` 指针算术形态亦编译通过，两种形态均可，此处仅声明实测形态）。
- `asc_vf_call` 直接转发实参，标量与 `__ubuf__` 地址（`(__ubuf__ T*)localTensor.GetPhyAddr()`）都可以传。

## 预防

- 泛型 VF 模板里一切与 dtype 相关的模板参数（LoadAlign/StoreAlign 源类型、LoadDist/StoreDist、CastTrait）都套 `if constexpr` 分派；先写最小 VF（load/store）打通编译与上板，再加计算链。
- 看 candidate note 的转换方向即可定位：`no known conversion from '__ubuf__ half *' to '__ubuf__ float *'` 说明首模板参数与指针元素类型不一致。

## 验证信号

以下信号来自该算子 kernel 实现会话（Ascend950，CANN 9.2.0）：VF 类型不匹配修复后逐级构建并在设备上回归。

- 修复前：最小复现构建报 `error: no matching function for call to 'LoadAlign'/'StoreAlign'`，candidate note `no known conversion from '__ubuf__ half *' to '__ubuf__ float *' for 1st argument`，实例化点 `AdamWVF<half>`（来源平台：950；证据等级：tool_observed）。
- 修复后：`if constexpr` 分派版最小复现构建 `SUCCESS`，EXIT 0（来源平台：950；证据等级：tool_observed）。
- 修复后：带完整计算链的最终 kernel 构建 `SUCCESS`，按 dtype 生成 3 份 ascend950 kernel 二进制；ST 上板 21/21 PASSED（0 FAILED），覆盖 fp32/fp16/bf16 × 多组 shape（小/非对齐/大 shape）与 NaN/Inf/wd=0/maximize/step=1/eps=0 边界用例（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖 Load/Store/Cast 的**类型匹配**根因；若报错伴随 `bisheng: error: clang frontend command failed with exit code 139` 且无前置 error:，先按 ICE 诊断流程定位（同会话另有 `__simd_vf__` 内运行期 if 触发 ICE 的记录），不要把本卡当 ICE 的解。
- `DIST_UNPACK_B16` / `DIST_PACK_B32` 是 16 位 dtype 的装载/存储分布参数；其他位宽 dtype（int8 等）的 LoadDist/StoreDist 取值未在本轨迹验证。
- 其他 CANN 版本 Reg:: API 签名可能有差异（未验证，推断机制可迁移）。
- VF 内 `__ubuf__ T*` 指针算术形式是否可用未做隔离验证（当前推荐 AddrReg 形式）。

**未决项**：

- VF 内 `__ubuf__ T*` 指针算术形式是否可用未做隔离验证（当前推荐 AddrReg 形式）。
## 坏实践

- 在泛型 VF 模板里把 LoadAlign/StoreAlign 的源类型模板参数写死为计算精度类型（float），指望指针隐式转换——`__ubuf__` 限定指针之间无隐式转换，直接重载失败。
- 试图用运行期 `if (std::is_same_v<...>)` 选择 LoadDist——它是模板参数，运行期不可选，且 VF 内运行期 if 本身会触发编译器 ICE。

## 误诊

- 曾怀疑 `__ubuf__` 指针算术（`varAddr + i*oneRepeatSize`）不保地址空间限定导致 no-match：实际根因是源类型模板参数与指针元素类型不一致；指针算术形式也未被验证，最终代码统一改用 AddrReg。
- 曾把本 no-match 错误当作同会话 bisheng ICE（exit 139）的根因：完整计算链恢复实验证明 ICE 另有触发点（VF 内运行期标量 if），两者是不同根因。
