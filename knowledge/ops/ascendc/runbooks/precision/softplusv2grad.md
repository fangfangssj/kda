---
type: "Runbook"
title: "SoftplusV2Grad 非默认θ线性区分支缺失致超差"
description: "以 1/(1+exp(-z)) 除法改写实现 SoftplusV2Grad 的 kernel，若省略 z>θ 线性区分支，非默认（小/负）θ 时 exp(-z) 不下溢为 0、分母≠1，与 golden ratio=1 超差；判别：默认 θ 用例全过、仅非默认 θ 用例集中超差且 VF 链无 CompareScalar/Select 阈值分支即命中。"
tags: ["task.debugging", "topic.compute.elementwise.activation", "topic.implementation.api_layer.basic_api", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T16:07:08Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["division_rewrite"]
---

# SoftplusV2Grad 非默认θ线性区分支缺失致超差

## 现象

在开发 SoftplusV2Grad（广播二元梯度算子，`grad_input = grad_output / (1 + exp(-β·self))`）的白盒 ST 时，39 条用例中 5 条精度超差（case00027/28/31/44/46），匹配率 45%~97%。这 5 条的共同特征：全部带**非默认 beta/threshold**，且 threshold 为小/负值（-1.925, -2.333, -9.812, -28.5, 4.586），而非默认的 20.0。默认 threshold（θ=20.0）的用例全部通过（baseline 41/41 PASS）。添加 5 条非默认 θ 复现用例后，未修复 kernel 跑出：

- case00044 (fp32, β=1.266, θ=-1.925): 497/11760 mismatch
- case00028 (fp16, β=-0.1087, θ=-9.812): 38784/69120 mismatch（56%，最差）
- case00031 (fp16, β=-1.344, θ=4.586): 6124/80640 mismatch
- case00027 (bf16, β=-1.977, θ=-28.5): 3499/80640 mismatch
- case00046 (fp32, β=1.132, θ=-2.333): 358/11760 mismatch

## 适用场景

- **工程形态**：kernel 直调（ACLNN ST 路径，`asc_vf_call` + `__simd_vf__` VF 链）。
- **API 路径**：`aclnnSoftplusV2GradGetWorkspaceSize` / `aclnnSoftplusV2Grad`；VF 链 `SigmoidGradVF<float>` 经 `asc_vf_call` 调用。
- **验证平台与版本**：Ascend950PR (dav-3510, arch35), CANN 9.0.0-beta.2。
- **适用粒度**：`operator_family`（适用于所有以 `ratio = where(z>θ, 1, sigmoid(z))` 形式实现、且设计声称"exp 下溢隐式处理线性区"的 SoftplusGrad 族算子）。
- **路线优先级**：`recommended_fastpath`（显式 threshold 分支是正确实现；依赖 exp 下溢的"除法改写"是 unsafe fallback）。
- **不适用/未验证场景**：非 SoftplusGrad 族的无 threshold 算子；GEIR 图模式（本轮未在该路径验证 threshold 修复）；非 arch35 平台未验证。

## 判别方法

**触发**：以 `1/(1+exp(-z))` 除法改写实现的 SoftplusGrad 族 kernel，ST 出现**分层失败**——默认 θ=20.0 用例全过（baseline PASS），仅带非默认（小/负/零）θ 的用例集中超差，且超差用例的 θ 全部落入使 `exp(-(θ+δ))` 不可忽略的区间。次级触发：超差率随 |θ| 减小而上升（θ=-9.812 时 56% 最差，θ=-2.333 时仅 358/11760）。

**可执行探针（按序）**：

1. **分层复现**：先只跑默认 θ=20.0 用例确认全过，再加入小/负/零 θ 用例（覆盖 θ∈{-28.5, -9.812, -1.925, -2.333, 4.586}）。若默认全过、非默认集中挂且超差率随 |θ| 减小而上升，即命中本判别。

2. **检查 kernel VF 链是否存在 threshold 线性分支**：定位计算 `ratio` 的 VF 链，识别是否含 `CompareScalar<CMPMODE::LT>` + `Select`（或等价的 `z > θ` 短路）。待识别的坏模式（BEFORE verbatim，取自 trace file_read 全文 `op_kernel/arch35/softplus_v2_grad_kernel.h`，`SigmoidGradVF` 4-op 链，无 threshold 分支）：

```cpp
template <typename T>
__simd_vf__ inline void SigmoidGradVF(
    __ubuf__ T* dstAddr, __ubuf__ T* srcGradAddr, __ubuf__ T* srcSelfAddr,
    T negBeta, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> selfReg, gradReg, denomReg, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);           // 尾块 count 可能 < oneRepeatSize
        AscendC::Reg::LoadAlign(selfReg, srcSelfAddr, aReg); // Load self
        AscendC::Reg::LoadAlign(gradReg, srcGradAddr, aReg); // Load grad_output
        AscendC::Reg::Muls(selfReg, selfReg, negBeta, mask); // z = self · (-β)        [Scalar reg]
        AscendC::Reg::Exp(selfReg, selfReg, mask);           // e = exp(z) = exp(-β·self) [Unary reg]
        AscendC::Reg::Adds(denomReg, selfReg, (T)1.0, mask); // denom = 1 + e          [Scalar reg]
        AscendC::Reg::Div(dstReg, gradReg, denomReg, mask);  // out = grad / denom      [Binary reg]
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}
```

   该链只算 `out = grad / (1 + exp(-β·self))`，**从不处理 `z > θ` 的线性区**（该区 `ratio = 1`，即 `out = grad`）。配套 `CallSigmoidGradVF` 只传 `neg_beta_`（无 threshold），类成员仅 `float neg_beta_;`。

3. **数值探针验证"下溢假设"是否成立**：对 `z` 略大于 `θ` 的元素（如 `z = θ + 5`）计算 `exp(-z)`。若 `exp(-z)` 显著非 0（如 `exp(-5)≈0.0067`），则分母 `1+exp(-z) ≈ 1.0067 ≠ 1`，"exp 下溢隐式处理线性区"假设崩塌——下溢只在 `z ≥ ~18`（fp32）/`z ≥ ~11`（fp16）时成立。可对 case00044（θ=-1.925, β=1.266, x=5→z=6.33）实测：`1/(1+exp(-6.33)) = 0.99822`，与 golden `ratio=1.0` 相对误差 ≈ 0.178%，远超 fp32 rtol `1e-5`。

4. **对照 golden 分支语义**：torch `aten::softplus_backward` 的 oracle 显式分支 `if beta*x > threshold: return grad_output`（ratio=1 exactly），否则 `return grad_output * sigmoid(beta*x)`。若 kernel 未复现此分支，则非默认 θ 必然超差。

**原始观测（tool_observed）**：默认 θ=20.0 baseline 41/41 PASS；加入 5 条非默认 θ 后未修复 kernel 41/46 PASS，5 条全挂（case00044 497/11760 mismatch, case00028 38784/69120 mismatch, case00031 6124/80640, case00027 3499/80640, case00046 358/11760）（来源平台：950；证据等级：tool_observed）。

**解释（root-cause interpretation）**：kernel 的 `SigmoidGradVF` 只计算 `ratio = 1/(1+e^{-z})`，从不处理 `z > θ` 的线性区（该区 golden `ratio=1`）。设计 §"除法改写" 声称线性区由 `exp(-z)` 在 fp32 下下溢为 0 隐式处理（分母→1→`grad/1=grad`），但该假设**仅对大 θ 成立**：θ=20 时 `exp(-20)≈2e-9 < fp32 ε≈1.2e-7`，分母精确舍入为 1.0。对小/负 θ，大量元素落入 `(θ, θ+18]` 的"近似但不等"区间，`exp(-z)` 不可忽略，分母≠1，`ratio` 偏离 1.0 超过容差。线性区误差 = `1 - 1/(1+exp(-z)) = exp(-z)/(1+exp(-z))`。这是根因解释，明确为 interpretation，非单点实测——实测闭环信号是分层失败 + 修复后全过（见验证信号），数值探针佐证但未单独隔离测量。

## 坏模式

- **依赖 exp 下溢隐式处理线性区（4-op 链，无 threshold 分支）**：对默认 θ=20.0 正确（`exp(-20)` 下溢），但对非默认 θ 失败——`z` 略大于 `θ` 的区间 `exp(-z)` 不可忽略，分母≠1，与 golden `ratio=1` 超差。判别方法·探针 2 已示 BEFORE 结构。该路线标 `banned`（对带 threshold 的算子不得依赖浮点下溢隐式处理线性区）。
- **把"非默认 θ 全超差"误判为测试数据问题**（随机种子/分布）而放宽容差：根因是 kernel 缺分支，放宽容差会掩盖真因。
- **只测默认 θ=20.0 的用例**就认为 kernel 正确：非默认 θ 是触发条件，ST 必须覆盖小/负/零 θ。

## 验证信号

- 默认 θ=20.0 baseline 41/41 PASS（来源平台：950；证据等级：tool_observed）
- 加 5 条非默认 θ 后未修复 kernel 41/46 PASS，5 条全挂（case00044 497/11760, case00028 38784/69120, case00031 6124/80640, case00027 3499/80640, case00046 358/11760）（来源平台：950；证据等级：tool_observed）
- 插入 threshold 线性分支（CompareScalar+Select）后 46/46 PASS（来源平台：950；证据等级：tool_observed）
- TilingUT 57/57 无回归（来源平台：950；证据等级：tool_observed）

## 失效边界

- 推断可迁移至 a2/a3（同 arch regbase API `CompareScalar`/`Select`/`CMPMODE`），但**未验证，推断可迁移**——本卡 `platforms: [950]` 为实测，不得据推断回填平台。
- GEIR 图模式路径未在本轮验证 threshold 修复（该路径有独立 AIVEC 崩溃问题，未收敛）。
- `z` 极大（`z ≥ 18` fp32 / `z ≥ 11` fp16）时 exp 下溢成立，原 4-op 链与修复后链结果一致——修复不改变大 z 行为，仅修正"略大于 θ"区间。