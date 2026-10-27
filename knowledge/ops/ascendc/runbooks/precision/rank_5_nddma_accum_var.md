---
type: "Runbook"
title: "RANK大于5时NDDMA漏维只解释accum错误"
description: "fused_mul_apply_momentum 在 Ascend950 上 7D Nesterov 的 NDDMA 外循环漏维会导致 accum 错；修复后 accum 转正而 var 仍错，证明 var 不属于同一根因，本卡不推断 var 的具体机制。"
tags: ["task.debugging", "topic.data_movement.memory_access", "topic.implementation.tiling", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["nddmaOuterIters"]
---

# RANK大于5时NDDMA漏维只解释accum错误

## 现象

在 Ascend950PR（arch35）上，`fused_mul_apply_momentum` 的 ST NPU L0 用例 `L0_030_FP32_NEST_7D`（shape=[1,2,3,4,5,6,7]，momentum=[-100,100]，x2=[-100,100]）出现分裂式精度偏差：**accum 输出完全正确（MERE=0），只有 var 输出巨大误差**（actual=-3697.47，expected=-2.725）。FP32 即出错，非 FP16 Cast 问题。

公式上 Nesterov 路径：`grad = x1 * x2`；`accum_new = accum * momentum + grad`；`delta = accum_new * momentum * lr + grad * lr`；`var_new = var - delta`。accum_new 正确意味着 grad 与 accum_new 两步对，误差在 `delta`（涉及 lr、momentum 两个标量读取）或 `var_new`。

## 判别方法

### 触发

高维（RANK>5）逐元素算子出现「accum 正确、var 巨大误差」的分裂式精度偏差，且 FP32 也错（非 FP16 Cast 问题）。本例触发信号：L0_030 accum MERE=0 但 var MERE=2.16e+03。

### 探针

1. 核对 NDDMA 外循环覆盖范围：在 kernel `Init` 与 `CopyInBrc` 中打印 `ndma_start = RANK - nddma_dims_` 与外循环上下界。若外循环未覆盖到 `ndma_start-1`，则 split axis `k` 与 NDDMA 覆盖区间 `[ndma_start, RANK)` 之间的维度未被迭代。
2. 单独比对 accum 与 var 的 MERE：accum MERE=0 + var MERE 巨大 → 误差不在共用搬运的 grad/accum_new，而在只用标量的 `delta`（lr/momentum 读取）或 `var_new`。
3. 修复 NDDMA 外循环后重跑：若 accum 由错转正确、var 仍错，则排除「var 误差与 accum 同根因」。

### 观测

- NDDMA 外循环修复后 L0_030 accum MERE=0（由错转正确），var MERE=2.16e+03（仍错）（来源平台：950；证据等级：artifact_verified）
- var actual=-3697.47 vs expected=-2.725，输出巨大偏差（来源平台：950；证据等级：artifact_verified）

### 判别

NDDMA 描述符在 arch35 上最多覆盖 5 维。当 RANK>5 时，split axis 与 NDDMA 覆盖区间 `[ndma_start, RANK)` 之间可能存在未被外循环迭代的维度，`CopyInBrc` 只搬了部分数据——这解释 accum 为何错。修复 `ndma_start = RANK - nddma_dims_`、外循环覆盖 `k..ndma_start-1` 后 accum 归零，反证 accum 误差根因为 NDDMA 漏维。

但 var 仍错，**排除 var 误差与 accum 同根因**：grad 与 accum_new 共用 NDDMA 搬运，搬运已修对；现有观测只能把 var 问题收窄到只用标量的 `delta`（lr/momentum 读取）或 `var_new`，不足以继续判定具体机制。

## 适用场景

- 工程形态：kernel 直调 + aclnn ST（ops-registry-invoke 工作流产物）。
- API 路径：自定义 NDDMA 广播搬运 `CopyInBrc`（`params.loopInfo.loopSize[nd] = a_i_seg`）+ 标量 `ReadScalar(coord, IN_LR, lr_val_t)` / `ReadScalar(coord, IN_MOM, mom_val_t)`。
- 验证平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35），CANN 9.0.0。仅 950 实测。
- 适用粒度：`operator_family`——NDDMA 描述符 ≤5 维的硬件约束适用于任何 RANK>5 且用 NDDMA 广播搬运的高维逐元素算子。
- 结论边界：NDDMA 外循环修复是让 accum 正确的必要修复，但不足以让 var 正确；var 必须作为独立问题继续采集证据。
- 不适用或未验证场景：RANK≤5 不触发漏维；a2/a3 未验证，推断可迁移（NDDMA 描述符维数上限是 arch 级约束）。

## 验证信号

- L0_030 accum MERE=0（NDDMA 外循环修复后由错转正确）（来源平台：950；证据等级：artifact_verified）
- L0_030 var MERE=2.16e+03（修复后仍错）（来源平台：950；证据等级：artifact_verified）
- var actual=-3697.47 vs expected=-2.725（来源平台：950；证据等级：artifact_verified）

## 失效边界

- NDDMA 外循环修复只解决 accum 漏维，**不解决 var 误差**——不要假设修了 NDDMA 外循环就解决了 7D Nesterov 全部精度问题。
- RANK≤5 不触发漏维。
- 仅 950 实测；a2/a3 未验证，推断可迁移（NDDMA 描述符维数上限是 arch 级约束）。
- 本卡没有确认 var 的具体根因，不能从 accum 修复结果外推标量读取机制。

## 坏实践

- 在没有标量实测值和对照实验时给 var 指定具体根因；应先在 kernel 内记录 `lr_val`/`mom_val`，再分别隔离 `delta` 与 `var_new`。
- 修了 NDDMA 外循环见 accum 转正就收工——var 仍错说明另有根因，须继续排查标量读取侧。

## 预防

- 任何用 NDDMA 广播搬运的高维（RANK>5）算子，设计阶段就要确认外循环覆盖 split axis 到 NDDMA 覆盖区间的全部维度，不要默认 NDDMA 描述符覆盖全部 RANK。
- 双输出算子出现「一个输出对一个输出错」时，优先怀疑只用于其中一个输出的标量读取，而非整体搬运逻辑——grad/accum_new 共用搬运，错只会出现在只用标量的步骤。