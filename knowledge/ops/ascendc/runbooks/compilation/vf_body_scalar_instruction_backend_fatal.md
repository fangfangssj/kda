---
type: "Runbook"
title: "VF 体内不支持标量指令——外提标量计算"
description: "__simd_vf__ 函数体内做标量解引用/除法/开方报 Unsupported scalar instruction in AIV loop；标量计算须外提到调用方按参数传入。"
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.simt_api, topic.programming_model.execution_model.simt]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# VF 体内不支持标量指令——外提标量计算

## 现象

Ascend 950 SIMT kernel 中用 `__simd_vf__ inline void XxxVF(...)` 定义向量函数、`asc_vf_call<XxxVF>(...)` 调用。初版为了让 RMSNorm 的缩放因子"就地"算出来，把标量计算写进了 VF 体内——解引用标量缓冲指针、做标量除法和开方（当时代码已将标量开方改为 CCE 内建 `sqrt`）：

```cpp
__simd_vf__ inline void RmsNormRowVF(
    __ubuf__ float* dst, __ubuf__ float* src,
    __ubuf__ float* stat, __ubuf__ float* w,
    float eps, float invD, uint16_t totalElems, uint16_t repeat)
{
    constexpr uint32_t VL = 64;
    AscendC::Reg::RegTensor<float> regSrc, regW, regOut, regInvRms;
    AscendC::Reg::MaskReg mask;
    float statVal = *stat;
    float rms = sqrt(statVal * invD + eps);
    float invRms = 1.0f / rms;
    AscendC::Reg::Duplicate(regInvRms, invRms);
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeat; ++i) {
        uint16_t off = i * VL;
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(regSrc, src + off);
        AscendC::Reg::LoadAlign(regW, w + off);
        AscendC::Reg::Mul(regOut, regSrc, regInvRms, mask);
        AscendC::Reg::Mul(regOut, regOut, regW, mask);
        AscendC::Reg::StoreAlign(dst + off, regOut, mask);
        remaining = (remaining > VL) ? (remaining - VL) : 0;
    }
}
```

同样的写法还出现在 gate 分支的 `BcastMulCastRowVF`（`float gateVal = *gatePtr;` 在 VF 体内解引用 gate 指针）。编译（在 kernel_meta 目录手工重跑编译命令）报：

```text
fatal error: error in backend: .../<generated>_kernel.cpp:48:5: Unsupported scalar instruction in AIV loop.
bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
```

## 适用场景

- 工程形态：算子仓 aclnn 包 `op_kernel` 的 SIMT kernel（`__simd_vf__`/`asc_vf_call`）开发。
- API 路径：`__simd_vf__` 函数定义、`asc_vf_call` 调用、`AscendC::Reg::Duplicate`/`Muls` 标量广播。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.2.0。
- 适用粒度：operator_family——所有 SIMT VF kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证：纯标量 kernel（不含 VF）不受此约束；传统 Vector（高阶 API）路径未验证。

## 根因

`__simd_vf__` 函数被编译成 AIV 上的向量循环体，后端拒绝在该循环体内生成标量指令——标量指针解引用（`*stat`、`*gatePtr`）、标量除法（`1.0f / rms`）、标量数学函数都属于"scalar instruction in AIV loop"。标量计算必须放在调用方（普通标量代码区）完成，结果作为标量参数传进 VF；VF 体内只允许向量指令和消费传入标量值的广播类指令（`Duplicate`/`Muls`）。

相邻卡片与区分条件：

- [__simd_vf__ if constexpr 致 hoisted 报错卡](simd_vf_if_constexpr_lowering_banned_exit_70.md)——同为 `__simd_vf__` 后端 exit 70 家族，但那卡由 if constexpr/constexpr trait 触发 `Unsupported Inst must be hoisted`，本卡由标量指令触发 `Unsupported scalar instruction in AIV loop`；错误串不同，先对串再套卡。
- [SIMT 内读块号致 Bisheng 崩溃卡](simt_move_block_metadata_to_vector_shell.md)——同属"VF 体内不能出现的代码"家族，那卡是块号 API（BlockIdx/BlockNum）触发前端异常终止，修复同为外提到 `__aicore__` 调用侧。
- [SIMT VF 栈数组动态索引卡](../precision/simt_vf.md)——VF 体裁相反形态的坑：栈数组动态索引静默产出全零（不报错），本卡是硬编译错误；排查 VF 问题时先分清"编译不过"还是"静默错误"。
- [device 标量开方卡](device_sqrt_uses_vector_api.md)——本卡现象里的标量开方须先按该卡用 CCE 内建 `sqrt`（`sqrtf`/`AscendC::Sqrt` 标量重载均不可用）；即便开方写法正确，放进 VF 体内仍会命中本卡。

## 解决配方

把标量计算外提到调用方，VF 签名改为直接接收标量值。修复后（verbatim）：

```cpp
__simd_vf__ inline void RmsNormRowVF(
    __ubuf__ float* dst, __ubuf__ float* src,
    float invRms, __ubuf__ float* w,
    uint16_t totalElems, uint16_t repeat)
{
    constexpr uint32_t VL = 64;
    AscendC::Reg::RegTensor<float> regSrc, regW, regOut, regInvRms;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::Duplicate(regInvRms, invRms);
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeat; ++i) {
        uint16_t off = i * VL;
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(regSrc, src + off);
        AscendC::Reg::LoadAlign(regW, w + off);
        AscendC::Reg::Mul(regOut, regSrc, regInvRms, mask);
        AscendC::Reg::Mul(regOut, regOut, regW, mask);
        AscendC::Reg::StoreAlign(dst + off, regOut, mask);
        remaining = (remaining > VL) ? (remaining - VL) : 0;
    }
}
```

gate 分支同理，指针参数改成标量值参数：

```cpp
// before: __ubuf__ float* gatePtr 参数 + VF 体内 float gateVal = *gatePtr;
// after:
__simd_vf__ inline void BcastMulCastRowVF(
    __ubuf__ float* dst, float gateVal, __ubuf__ bfloat16_t* src,
    uint16_t totalElems, uint16_t repeat)
```

调用方先在标量区算好，再传值进 VF（verbatim）：

```cpp
// S4: scalar reduce sum_sq = Σ_d keys[d]² → stat
auto b0Bf16 = bufB0_.Get<T>();
float sumSq = 0.0f;
for (int64_t d = 0; d < D_; d++) {
    float v = (float)b0Bf16.GetValue(d);
    sumSq += v * v;
}
// Compute invRms = 1/sqrt(sumSq * invD + eps) in scalar
float invRms = 1.0f / sqrt(sumSq * invD_ + eps_);

// S5: RmsNormCastRowVF: cast keys bf16→fp32 + normalize → B1 (normed_k)
asc_vf_call<RmsNormCastRowVF>(
    (__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),
    (__ubuf__ bfloat16_t*)bufB0_.Get<T>().GetPhyAddr(),
    invRms,
    (__ubuf__ float*)wBuf_.Get<float>().GetPhyAddr(),
    (uint16_t)dAlignElems_, dRep);
```

消费端：VF 内用 `AscendC::Reg::Duplicate(regInvRms, invRms)` 把传入标量复制成向量寄存器，或 `AscendC::Reg::Muls(regOut, regSrc, gateVal, mask)` 直接做向量×标量。

## 设计决策

- 选"调用方标量 + 参数传入"：VF 内标量计算直接不可编译，没有备选余地；该结构同时把归约（sumSq/dot）留在标量循环，满足与 golden 逐位对齐的要求。
- 边界归纳为"标量值可用、标量计算不可用"：传入的标量经 `Duplicate`/`Muls` 广播是合法的，非法的是 VF 体内自行做标量算术/解引用。

## 预防

写 `__simd_vf__` 函数前过一遍清单：函数体内是否出现 `*指针`、`/`、标量 math 调用、标量比较分支？有则全部外提到调用方，VF 体内只留向量指令 + 传入标量值的广播消费。

## 验证信号

来源说明：以下信号来自同一次 kernel 开发会话的连续编译循环（Ascend 950，CANN 9.2.0）。

- 修复前编译报 `fatal error: error in backend: ... Unsupported scalar instruction in AIV loop`，bisheng exit 70（来源平台：950；证据等级：tool_observed）。
- 标量外提后同一构建不再报该错误，编译错误推进到下一独立问题（`not support bf16 type cast`，见 [bf16 cast 位运算替代卡](bf16_scalar_to_fp32_bit_extension.md)），证明 AIV-loop 标量错误已消除（来源平台：950；证据等级：tool_observed）。
- 外提后的 VF 结构保留至最终可编译版本，`bash build.sh` 输出 `SUCCESS`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 验证于 ascend950 + CANN 9.2.0 SIMT 路径；实测覆盖的"标量指令"至少包括指针解引用、标量除法、标量开方三类。
- 传统 Vector/高阶 API kernel 不经 `__simd_vf__`，未验证是否同类受限。
- 错误定位指向的是 codegen 产物 `<generated>_kernel.cpp` 的行号，需回 `op_kernel/arch35/*.h` 源头对照，不要改生成文件。

**未决项**：

- "AIV loop 内允许/禁止指令"的官方完整清单未找到；本卡边界来自实测三类指令的外推。
## 坏实践

- 为"省一次标量↔向量数据往返"把 rms/gate 计算塞进 VF 体内——直接不可编译。
- 看到错误指向生成文件行号就在生成文件里修——改源头 VF 定义才有效。
