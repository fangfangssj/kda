---
type: "Runbook"
title: "arch35 VSEL_CMPMASK_SPR 标量批跑 mask 污染"
description: "arch35 上 Compare + Select(VSEL_CMPMASK_SPR) 在标量/小 count 批跑时出现 Actual=Golden/2，单跑 PASS；切换 VSEL_TENSOR_TENSOR_MODE 仅将失败从 121 降至 52。MaskUnPack 读取 stale padding 是待独立验证的解释，不是已确认机制。"
tags: ["task.debugging", "topic.precision", "topic.implementation.masking", "topic.compute.elementwise.comparison_logic"]
status: stable
sources: []
created_at: "2026-08-17T02:46:24Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# arch35 VSEL_CMPMASK_SPR 标量批跑 mask 污染

## 现象

在 Ascend950PR（arch35/RegBase）上开发 `softplus_v2_grad` 算子时，kernel 用 `Compare`（CMPMODE::GT）生成 mask，再用 `Select`（VSEL_CMPMASK_SPR）按 mask 选 gradOutput 或 grad_sig。标量输入（count=1，count_padded=64）在**批跑**时出现 `Actual = Golden / 2`——Select 选择了 grad_sig（src1）而非 gradOutput（src0），即 mask 被当作 FALSE。但**同一标量 case 单独跑时 PASS**（mask 正确）。

kernel 调用：
```cpp
AscendC::Select<float, uint8_t>(bBx, bMask, bG, bThr, AscendC::SELMODE::VSEL_CMPMASK_SPR, count_padded);
```
其中 bMask 来自 `Compare<float, uint8_t>(bMask, bBx, bThreshold, CMPMODE::GT, count_padded)`。

## 适用场景

- **工程形态**：kernel 直调（ops-registry-invoke，RegBase 编程模型，arch35/Ascend950PR）。
- **API 路径**：`AscendC::Compare` + `AscendC::Select`（VSEL_CMPMASK_SPR 模式），T=float，mask 类型 uint8_t。
- **验证平台与版本**：Ascend950PR（arch35），CANN 9.0.0-beta.2；Python ST 581 例实测。
- **适用粒度**：`platform_wide`——arch35/Ascend950PR 上所有用 Compare+Select（VSEL_CMPMASK_SPR）处理标量/小 count 的 kernel 均可能受影响。
- **路线优先级**：`diagnostic_only`——切换 `VSEL_TENSOR_TENSOR_MODE` 只用于隔离 CMPMASK_SPR 路径：失败从 121 降至 52，但没有形成可交付修复。
- **不适用或未验证场景**：a2/a3 平台未验证；count >= vector lane size（64 for float）的大 shape 场景不受此问题影响（无 padding stale）；非 arch35 芯片未验证。

## 根因

arch35（dav_c310）上 `Select` 的 `VSEL_CMPMASK_SPR` 模式（6-arg overload，T=float）内部实现路径：

1. `Reg::LoadAlign<uint8_t, Reg::MaskDist::DIST_US>(selMask, sel)`——从 mask tensor（bMask，uint8_t*）加载到 selMask 寄存器。
2. `Reg::MaskUnPack(selMask, selMask)`——对 T=float（4 字节）将 packed bitmask 解包。
3. 循环 `repeatTime` 次：`Reg::Select(dstReg, src0Reg, src1Reg, selMask)`——按 selMask 选 src0/src1。

而 `Compare`（Level2，T=float）的输出：

1. `Reg::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, mask)`——计算 packed bitmask。
2. `Reg::StoreUnAlign(dstT, dstReg, uReg)`——存储 packed bitmask 到 dst tensor，**但 dstT = reinterpret_cast<__ubuf__ T*>(dst)** 把 uint8_t* 当 float* 存。
3. 64 个 float 元素只需 64 bit = 8 字节 packed bitmask，但 dst buffer 为 64 字节，**剩余 56 字节为 stale**。

**机制推断**（部分未追到硬件层）：Compare 只写前 8 字节 packed bitmask，Select 的 `LoadAlign` 可能读取超过 8 字节（mask 寄存器宽度），多出的 stale 字节经 `MaskUnPack` 后可能污染有效元素的 mask bit。标量 case（count=1）时 padding 区域最大，stale 数据影响最显著。批跑时前序 kernel launch 的残留数据落在 padding 区域，导致 `Actual=Golden/2`（mask 被翻转为 FALSE）。

> 注：此机制链中「LoadAlign 实际读取字节数」和「MaskUnPack 对 stale 字节的处理」未独立测量，属推断。agent 的 scalar GetValue/SetValue bypass（试图绕过 Select）反而产生 568 失败（更差），部分证伪了「Select mask 污染是唯一根因」的假设。

CANN header verbatim 代码（SelectMode0Level2，VSEL_CMPMASK_SPR，6-arg，T=float；原文读自 `…/asc/impl/basic_api/dav_c310/kernel_operator_vec_sel_impl.h`）：

```cpp
} else {  // sizeof(T) != 8
    RegT src0Reg, src1Reg, dstReg;
    Reg::MaskReg maskReg, selMask;
    if constexpr (sizeof(T) == 1) {
        Reg::LoadAlign<U>(selMask, sel);
    } else {
        Reg::LoadAlign<U, Reg::MaskDist::DIST_US>(selMask, sel);
        if constexpr (sizeof(T) == 4) {
            Reg::MaskUnPack(selMask, selMask);
        }
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        maskReg = Reg::UpdateMask<T>(sreg);
        Reg::LoadAlign<T>(src0Reg, src0 + i * repeatElm);
        Reg::LoadAlign<T>(src1Reg, src1 + i * repeatElm);
        Reg::Select(dstReg, src0Reg, src1Reg, selMask);
        Reg::StoreAlign<T>(dst + i * repeatElm, dstReg, maskReg);
    }
}
```

CANN header verbatim 代码（Compare Level2，T=float；原文读自 `kernel_operator_vec_compare_continuous_impl.h`）：

```cpp
Reg::RegTensor<T> src0Reg, src1Reg;
uint32_t offset = GetVecLen() / sizeof(T) / 8;
__ubuf__ T *dstT = reinterpret_cast<__ubuf__ T*>(dst);
for (uint16_t i = 0; i < repeatTime; ++i) {
    mask = Reg::UpdateMask<T>(sreg);
    Reg::LoadAlign(src0Reg, src0 + i * repeatElm);
    Reg::LoadAlign(src1Reg, src1 + i * repeatElm);
    Reg::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, mask);
    if constexpr (sizeof(T) == 1) {
        Reg::StoreAlign(dst + i * offset, dstReg);
    } else {
        Reg::StoreUnAlign(dstT, dstReg, uReg);
    }
}
if constexpr (sizeof(T) > 1) {
    Reg::StoreUnAlignPost<T, Reg::PostLiteral::POST_MODE_NORMAL>(dstT, uReg, 0);
}
```

> 注：Select 语义 `Reg::Select(dst, src0, src1, mask) = mask ? src0 : src1`（从 IsFinite 参考实现推导）：

```cpp
Reg::Duplicate(vReg0, ZERO);    // vReg0 = 0
Reg::Duplicate(vReg1, ONE);     // vReg1 = 1
// cmpMaskReg = mask of "is NaN or Inf" (NOT finite)
Reg::Select(vDstReg0, vReg0, vReg1, cmpMaskReg);
// Expected: vDstReg0 = is_finite ? 1 : 0
// If dst = mask ? src0 : src1: vDstReg0 = cmpMaskReg ? 0 : 1 = is_finite ? 1 : 0 ✓
```

## 判别方法

**触发**：kernel 用 `Compare` + `Select`（VSEL_CMPMASK_SPR）处理标量或 count < vector lane size 的输入，批跑时部分 case 出现 `Actual = Golden / 2` 或 Select 选错分支。

**探针**：
1. 在 Compare 与 Select 之间注入 `AscendC::Printf`——若失败数变化（本 trace：119→83），则确证 Compare→Select 有时序/同步敏感性。
2. 将 `VSEL_CMPMASK_SPR` 改为 `VSEL_TENSOR_TENSOR_MODE`——若失败数显著下降（本 trace：121→52），则确证 VSEL_CMPMASK_SPR 的 MaskUnPack/SPR 路径是问题源。
3. 标量 case 单独跑 PASS、批跑 FAIL——确证为跨 launch 的 state leakage / padding stale。

**原始观测**：
- VSEL_CMPMASK_SPR：121/581 失败。
- VSEL_TENSOR_TENSOR_MODE：52/581 失败。
- PRINTF 注入：83/581 失败。
- L1_001 标量 debug：count=1 pad=64 mask0=0xFF（TRUE）g0=0.115906 sig0=0.0 res0=0.115906（gradOutput，正确）——单独跑 PASS；批跑 Actual=Golden/2。

**解释**（interpretation，属推断、未独立验证）：`VSEL_CMPMASK_SPR` 对 T=float 走 `MaskUnPack` 路径，读取的 mask 字节超出 Compare 实际写入范围（8 字节 packed bitmask），stale 字节经 MaskUnPack 后污染有效元素 mask bit。`VSEL_TENSOR_TENSOR_MODE` 直接从 mask tensor 读取，不经 MaskUnPack/SPR，stale 影响减小但未完全消除（52 剩余）。

## 验证信号

- VSEL_CMPMASK_SPR 模式：121/581 失败（来源平台:Ascend950PR；证据等级:tool_observed）
- VSEL_TENSOR_TENSOR_MODE 模式：52/581 失败（来源平台:Ascend950PR；证据等级:tool_observed）
- PRINTF 注入 Compare→Select 间：83/581 失败（来源平台:Ascend950PR；证据等级:tool_observed）
- 标量 case（L1_001）单独跑 PASS、批跑 FAIL（Actual=Golden/2）（来源平台:Ascend950PR；证据等级:tool_observed）
- Compare 副作用：设置 `Internal::g_cmpMaskLow` / `Internal::g_cmpMaskHigh` SPR（来源平台:Ascend950PR；证据等级:artifact_verified）

## 设计决策

- **切换模式能证明什么**：`VSEL_TENSOR_TENSOR_MODE` 不经同一 SPR/MaskUnPack 路径，失败数下降说明故障与该路径相关；52 个剩余失败表明它不能作为修复。padding stale 的解释仍是推断，需用独立 mask dump 验证。
- **放弃的候选**：
  - 纯算术 step 函数（无 Compare/Select，用 Muls/Mins/Maxs 构造 mask float 0/1）：57 失败，劣于 VSEL_TENSOR_TENSOR_MODE（52），因 step 函数的浮点精度问题（bx-threshold 极大时 t*1e30 溢出为 inf，min(inf,1.0)=1.0 应为 mask=1 但结果 mask=0）。
  - 标量 GetValue/SetValue bypass（count==1 时用标量路径）：568 失败，远劣于 baseline——GetValue/SetValue 在 arch35 向量 kernel 中的 scalar/vector API 混用导致严重破坏。
  - MTE3_MTE2 SetFlag/WaitFlag prime 同步：57-59 失败，劣于 baseline——同步 flag 不能解决 mask 数据层面的 stale 问题。
- **当前结论**：本卡只提供故障签名与对照实验；标量/极小 count 场景尚无已验证修复。

## 坏实践

- 标量 GetValue/SetValue bypass：在 arch35 向量 kernel 中对 count==1 用 `LocalTensor::GetValue` / `SetValue` 做标量计算，产生 568/581 失败——scalar/vector API 混用在 arch35 上有严重副作用。
- 纯算术 step 函数：用 `(bx-threshold)*1e30` + `min/max` 构造 float mask 替代 Compare+Select，因 FP32 溢出（inf）精度问题反而更差。
- MTE3_MTE2 prime SetFlag：在 kernel 开头「预置」MTE3_MTE2 flag 试图解决跨 launch 同步，反而更差（57-59 失败）。

## 预防

- 在 arch35 RegBase kernel 中，标量/小 count 使用 `Compare` + `Select` 时必须纳入批跑探针；不能因切换模式后失败减少就把 `VSEL_TENSOR_TENSOR_MODE` 作为推荐实现。
- mask tensor 的 padding 区域在批跑时可能含前序 launch 的残留数据；如有可能，在 Compare 前清零 mask tensor 的 padding 区域。
- 标量输入（count=1）在 arch35 向量 kernel 中是高风险场景——向量 API（Compare/Select/LoadAlign）的内部实现对 count < vector lane size 的处理依赖 padding 区域，可能受 stale 数据影响。

## 失效边界

- a2/a3 平台：未验证，推断可迁移（Select/Compare 的 MaskUnPack/LoadAlign 行为是 dav_c310 架构特有，其他 SoC 架构的 Select 实现可能不同，需独立验证）。
- count >= 64（float）的大 shape 场景：不受此问题影响（无 padding stale）。
- VSEL_TENSOR_TENSOR_MODE 的 52 剩余失败：仍为标量/极小 count 场景，根因（MaskInterleave/pintlv_b16 在 padding 区域污染有效 bit）未追到硬件机制层。