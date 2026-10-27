---
type: "Runbook"
title: "EMA 公式 stat+m*(batch-stat) 在 inf 时 NaN"
description: "bn_training_update 的 S10/S11 EMA 步用 stat+m*(batch-stat) 形式，当 mean/variance=±inf、batch 为有限值时 inf+(-inf)=NaN 污染 runningMean/runningVariance；golden 的 (1-m)*stat+m*batch 在 inf 时保持 inf，二者数学等价但 IEEE 754 不等价。"
tags: ["task.debugging", "topic.precision", "topic.compute.normalization"]
status: stable
sources: []
created_at: "2026-08-18T15:16:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# EMA 公式 stat+m*(batch-stat) 在 inf 时 NaN

## 现象

bn_training_update 精度调试中，blackbox TTK 含 inf 统计量的用例（如 `(3,12)` fp16、`variance` 范围 `(inf, inf)`；`(1,15,1,2,3)` fp16、`mean` 范围 `(-inf,-inf)`）报 `runningVariance` 或 `runningMean` `GOLD: FAIL`，reason 为 `NaN/Inf mismatch`。机制链：golden 用 `runningMean = (1-m)*mean + m*batchMean`、`runningVar = (1-m)*var + m*batchVar`；kernel 用数学等价但 IEEE 754 不等价的 `mean + m*(batchMean - mean)`，当 `mean=inf`、`batchMean=finite` 时 `batchMean - inf = -inf`、`m*(-inf) = -inf`、`inf + (-inf) = NaN`，污染 runningMean/runningVariance 输出。

## 适用场景

- **工程形态**：kernel 直调 + aclnn TTK 黑盒验证。
- **API 路径**：`aclnnBNTrainingUpdate` → `BNUpdateKernel::ComputeStats()` 的 S10/S11 EMA 步（`op_kernel/arch35/bn_training_update_kernel.h`）。
- **验证过的平台与 CANN 版本**：Ascend950（Ascend950PR），CANN 9.0.0-beta.2。
- **适用粒度**：`same_operator`（bn_training_update 的 EMA 输出步骤 S10/S11）。
- **路线优先级**：`recommended_fastpath`（`(1-m)*stat + m*batch` 是与 golden 运算顺序一致的唯一正确形式）。
- **不适用或未验证场景**：有限输入下两种形式数学等价、都正确，本判别仅在 stat 或 batch 为 ±inf 时触发；a2/a3 未验证。

## 根因

IEEE 754 下 `inf + (-inf) = NaN`。EMA 公式 `stat + m*(batch - stat)` 在 `stat=±inf`、`batch=finite` 时：`batch - stat = ∓inf`，`m*∓inf = ∓inf`，`stat + ∓inf = inf + (-inf) = NaN`。而 golden 的 `(1-m)*stat + m*batch`：`(1-m)*inf = inf`（或 -inf），`m*finite = finite`，`inf + finite = inf`（正确）。两形式数学等价但 IEEE 754 下不等价，inf 场景结果不同。

## 判别方法

**触发（tool_observed）**：bn_training_update 在含 ±inf 统计量输入的 blackbox TTK 用例上，`runningVariance`/`runningMean` 输出 `GOLD: FAIL`，reason `NaN/Inf mismatch`；而 `y`/`batchMean`/`batchVariance` 同用例可能 PASS，定位到 EMA 输出步骤而非整算子。

**探针**：跑 TTK blackbox smoke，挑选 `mean`/`variance` 范围含 `±inf` 的用例（如 variance=inf 的 fp16 用例、mean=-inf 的 fp16 用例），`grep "GOLD:"` 取逐用例 PASS/FAIL 与 reason；再读 `op_kernel/arch35/bn_training_update_kernel.h` 的 S10/S11 块，判别 EMA 是否写成 `Sub(batch, stat) → Muls(m) → Add(stat, …)`（即 `stat + m*(batch - stat)` 形式）。

**观测（tool_observed）**：`stat + m*(batch - stat)` 形式下，含 inf 输入用例 `runningMean`/`runningVariance` `GOLD: FAIL`（`NaN/Inf mismatch`）；将 EMA 改写为 `Muls(stat, 1-m) → Muls(batch, m) → Add`（即 `(1-m)*stat + m*batch`）后，同批 inf 用例 `runningMean`/`runningVariance` 由 FAIL 转 PASS。

**判据（interpretation，agent 根因分析）**：失败仅出现在 stat 或 batch 为 ±inf 的用例，且 reason 恒为 `NaN/Inf mismatch` 而非数值容差越界，即可判定为 EMA 运算顺序触发的 IEEE 754 inf 相消：`inf + (-inf) = NaN`。根因是 `stat + m*(batch - stat)` 与 golden 的 `(1-m)*stat + m*batch` 在有限域数学等价、在 inf 域 IEEE 754 不等价。改写为 `(1-m)*stat + m*batch` 与 golden 运算顺序对齐后 inf 用例通过，反证判据成立。

## 验证信号

- 修复前含 inf 输入用例 `runningVariance`/`runningMean` `GOLD: FAIL`（reason `NaN/Inf mismatch`）（来源平台：Ascend950；证据等级：tool_observed）
- EMA 改为 `(1-m)*stat + m*batch` 后，含 inf 用例 `runningMean`/`runningVariance` 由 FAIL 转 PASS（来源平台：Ascend950；证据等级：tool_observed）
- 改写后 AscendC `Muls→Muls→Add` 与原 `Sub→Muls→Add` 指令数一致，无新增指令开销（来源平台：Ascend950；证据等级：artifact_verified）

## 失效边界

判别与改写仅验证于 Ascend950 / CANN 9.0.0-beta.2、bn_training_update 的 S10/S11 EMA 步。a2/a3 未验证，按 IEEE 754 inf 运算语义平台无关推断可迁移，未实测。本判别不覆盖 fp16 极小统计量（非 inf）下 `runningVariance` 的 ulp 精度边界，那是另一根因。本判别也不覆盖 `y` 输出路径的融合标量运算顺序（`-inf + inf = NaN`），那是同算子另一独立根因。

## 预防

凡 EMA/加权和 `a*stat + b*batch`（a+b=1）在 kernel 实现，必须用 `(1-m)*stat + m*batch` 直接形式，与 golden 运算顺序对齐；禁用数学等价的 `stat + m*(batch - stat)`，后者在 stat 为 ±inf 时产生 `inf+(-inf)=NaN`。AscendC V 链 `Sub→Muls→Add` 改 `Muls→Muls→Add` 不增加指令数。

## 坏实践

把数学等价当 IEEE 754 等价：`a + b*(c - a)` 与 `(1-b)*a + b*c` 在有限域等价，在 inf 域结果不同（前者 NaN，后者 inf）。判别此类 bug 时不能只看数值容差，须检查 reason 是否为 `NaN/Inf mismatch` 且仅出现在含 inf 输入的用例。