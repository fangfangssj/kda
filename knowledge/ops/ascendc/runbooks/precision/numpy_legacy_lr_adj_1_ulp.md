---
type: "Runbook"
title: "numpy legacy 标量提升致 lr_adj 1-ULP 偏差"
description: "numpy 1.26.x legacy promotion 将 Python float 与 np.float32 标量运算提升为 FP64 再降级回 FP32，与 kernel 纯 FP32 标量路径产生 1-ULP 偏差，在 var_out 近零抵消处被放大为 mare 4e-3。判别要点是 golden 标量运算走了 FP64-then-cast 而 kernel 走 pure-FP32。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-18T07:46:47Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# numpy legacy 标量提升致 lr_adj 1-ULP 偏差

## 现象

在开发 InplaceApplyAdaMax 算子（arch35 / Ascend 950）的黑盒 TTK 精度验证时，大 shape FP32 多核路径出现稳定可复现的精度失败：shape `(9,3,892611,1,1,1,8)` FP32（N≈193M 元素），仅 output idx 0（var_out）的 mare（max relative error）超阈——4.51e-3 ~ 8.59e-3 vs 阈值 1.22e-4；而 idx 1/2（m_out / v_out）全 PASS。mere（mean relative error）极小（~1e-8），说明绝大多数元素正确，只有少数 var_out 近零元素因分母近零而相对误差爆炸。两次独立重跑均命中同一失败，数据相关但稳定可复现。

被诊断的 kernel 标量派生路径（`ReadScalars`，verbatim 取自 trace 中 file_read，artifact_verified）以纯 FP32 计算派生标量：

```cpp
// Compute derived scalars (§1.3 式3, §5 标量预处理):
//   lr_adj = lr / (1 - β1^t)
//   one_minus_β1 = 1 - β1
lr_adj_val_         = lr_val_ / (static_cast<T>(1.0) - beta1_power_val_);
one_minus_beta1_val_ = static_cast<T>(1.0) - beta1_val_;
```

关键行 `lr_adj_val_ = lr_val_ / (static_cast<T>(1.0) - beta1_power_val_);`——先做 FP32 减法 `1.0f - b1p`（已舍入），再用舍入后的分母做 FP32 除法。

## 适用场景

- 工程形态：ACLNN 算子开发（ops-registry-invoke-glacier），kernel 直调 + 黑盒 TTK 精度验证。
- API 路径：`AscendC::DataCopyPad` / `LocalTensor::GetValue` / 标量 SP 运算 / `Reg::Div`（`inplace_apply_ada_max_vf.h` 中 VF S5-S8 compute chain）。
- 验证过的平台与 CANN 版本：Ascend 950（arch35），CANN 9.0.0。
- 适用粒度：`operator_family`——任何使用 numpy golden 且在 AI core 上以纯 FP32 计算标量派生量（lr_adj、omb1 等）的 AdaMax 族/自适应优化器算子均可能命中。
- 路线优先级：`recommended_fastpath`——在 AI core 上以 FP32 补偿运算复现 golden 的 FP64→FP32 降级路径，是无需 FMA/double 的正确解法。
- 不适用或未验证场景：numpy ≥ 2.0（NEP 50 后 promotion 规则改变，此 1-ULP 陷阱不复现）；非 numpy golden 的算子（如 PyTorch / C++ golden）不适用；a2/a3 平台未验证（推断可迁移但未实测）。

## 判别方法

### 触发

黑盒精度测试中，特定大 shape FP32 用例仅 var_out（或某个含抵消运算的输出）mare 超阈，mere 极小，且两次独立重跑命中同一组用例（数据相关、稳定可复现）。触发信号：失败集中在一个含近零抵消的输出，其余输出 PASS。

### 探针

在算子开发目录的 env.sh 环境下运行以下 Python 探针，检查 golden 使用的是 numpy legacy promotion 还是 NEP 50：

```bash
source docs/<OP>/develop/env.sh 2>/dev/null; python3 -c "import numpy as np; print('numpy version:', np.__version__); b1 = np.float32(0.9); r = 1.0 - b1; print('1.0 - np.float32(0.9) dtype:', r.dtype, 'value:', r)"
```

若输出 `numpy version: 1.26.4` 且 `1.0 - np.float32(0.9) dtype: float64`，则 golden 的标量运算（`1.0 - b1`、`lr / (1.0 - b1p)`）先在 FP64 中计算标量再降级，与 kernel 纯 FP32 存在 1-ULP 偏差。

### 观测

- numpy 版本：1.26.4（pre-NEP 50）。
- `1.0 - np.float32(0.9)` → `np.float64(0.10000002384185791)`（Python float `1.0` 将 np.float32 提升为 FP64）。
- 但 `np.float64_scalar * np.float32_array` → `np.float32_array`（numpy legacy "value-based casting" / weak scalar：标量值在 FP32 范围内时被降级到数组 dtype）。
- 所以 golden 的 `m_out`/`v_out`/`var_out` 最终是 FP32，但 `lr_adj` 和 `omb1` 作为 FP64 标量计算后再降级参与数组运算。

### 判别

判别结论（root-cause interpretation）：golden（numpy 1.26.4 legacy promotion）计算 `lr_adj = float32(float64(lr) / (1.0 - float64(b1p)))`——即在 FP64 中做标量除法，再舍入到 FP32；kernel 计算 `lr_adj = float32(lr) / (float32(1.0) - float32(b1p))`——即先做 FP32 减法（已舍入分母），再做 FP32 除法。两条路径的舍入顺序不同，约 5% 的 lr/b1p 组合产生 1-ULP 偏差。当 `var_out = var - lr_adj * (m_out / (v_out + eps))` 中 `var ≈ lr_adj * m_out / (v_out + eps)` 时 var_out 近零（抵消），1-ULP 的 lr_adj 绝对偏差（~3e-8）传到 var_out 后相对误差 = 绝对误差 / |var_out|，当 |var_out| ~ 2.5e-6 时相对误差 ~ 1.2e-2，远超阈值 1.22e-4。N≈193M 元素时，命中极端近零 var_out 的概率显著上升。

判别要点：`omb1 = 1 - beta1` 的 FP64-then-cast 与 pure-FP32 路径始终相等（`1.0 - b1` 当 b1 ∈ [0.01, 0.1] 的 FP32 减法已精确舍入），无需补偿；偏差只在 `lr_adj` 的除法路径出现。这把症状（仅 var_out 失败、mere 极小）与根因（lr_adj 标量除法的 1-ULP 差异被近零抵消放大）对应起来。

## 验证信号

- shape `(9,3,892611,1,1,1,8)` FP32 仅 var_out mare 4.51e-3~8.59e-3 vs 阈值 1.22e-4，mere ~1e-8，两次独立重跑命中同一失败（来源平台：950；证据等级：tool_observed）
- 探针输出 numpy 1.26.4，`1.0 - np.float32(0.9)` dtype 为 float64（来源平台：950；证据等级：tool_observed）
- kernel `ReadScalars` 纯 FP32 标量派生代码（`lr_adj = lr / (1.0f - b1p)`）（来源平台：950；证据等级：artifact_verified）
- 以补偿除法替换 lr_adj 标量路径后，黑盒 567/567 PASS 两轮独立运行、白盒 ST 53/53 PASS（来源平台：950；证据等级：tool_observed）

> 上方 1-ULP 机制链的具体数值（约 5% 组合、~3e-8 绝对偏差、|var_out| ~ 2.5e-6）来自 trace 中推理事件的根因分析，属 interpretation，不作为 tool_observed 证据条目。补偿除法的 verbatim C++ 实现（after/closure）未从 trace 逐字恢复。

## 失效边界

- 已验证：Ascend 950（arch35）CANN 9.0.0，numpy 1.26.4 golden + 大 shape FP32 用例触发，补偿除法后两轮黑盒全 PASS。
- 未验证，推断可迁移：numpy ≥ 2.0（NEP 50）后标量 promotion 规则改变，`1.0 - np.float32(0.9)` 不再提升到 FP64，此 1-ULP 陷阱不复现，kernel 纯 FP32 路径与 golden 一致；a2/a3 平台未实测，推断 AI core SP 的 FP32 运算语义一致可迁移（TwoProd splitter 常数 2^13+1 假设 FP32 为 23 位尾数，平台无关）。
- 不适用：非 numpy golden（PyTorch / C++ reference）若 golden 以纯 FP32 计算标量，则无此偏差。
