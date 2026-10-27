---
type: "Runbook"
title: "arch35 sigmoid 大负 z 下 exp 溢出 inf 致 sig=0"
description: "arch35 上把 sigmoid 改写为 1/(1+exp(-z)) 后，z≈-88 时 Exp(88) 溢出为 inf 致 sig=1/inf=0，且 sig≈6e-39 子正常数被 FTZ 清零；单独跑 L1_018 即 FAIL（非跨调用污染），Actual=0 量级吻合 subnormal/FTZ 而非分支选择错。诊断指向数值形式而非控制流。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.activation", "topic.compute.elementwise.math_function"]
status: stable
sources: []
created_at: "2026-08-17T18:13:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# arch35 sigmoid 大负 z 下 exp 溢出 inf 致 sig=0

## 现象

在 Ascend950PR（arch35/DAV_3510，CANN 9.0.0-beta.2）调试 `softplus_v2_grad` 时，弃用 cmpmask 路径、改纯算术 hard-step blend 后，仍有一类用例**单独跑就 FAIL**（非跨调用污染）。典型 `aclnnSoftplusV2Grad_L1_018`（`beta=17.0095, thr=0.0040128, self∈[-100,100], gradOutput=±FLT_MAX=±3.4028234663852886e+38`）：单独跑输出 `[FAIL] MERE=4.898552e-03 MARE=1.000000e+00 mismatch=2203/441000 nanBoth=0`，最大差异位置 `[11310]: Golden=2.152225 Actual=0.000000`。该位置对应 `z ≈ -88`（`self ≈ -5.2`），golden 期望 `gradOutput * sigmoid(-88) ≈ FLT_MAX * 6.06e-39 ≈ 2.06`，kernel 实际给出 `0`。早期算术版（用 `Reciprocal` 求 `sig` 中间值、exp 输入未 clamp）PyTorch ST 失败 237/238，其中含此类 subnormal 失败。

## 适用场景

- **工程形态**：kernel 直调（RegBase，arch35/DAV_3510 同步向量 API），算子含 sigmoid/softplus 类分支。
- **API 路径**：`AscendC::Exp`、`AscendC::Reciprocal`、`AscendC::Div`、`AscendC::Mins`、`AscendC::Muls`、`AscendC::PipeBarrier<PIPE_V>`。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.0.0-beta.2。
- **适用粒度**：`operator_family` —— 任何在 arch35 上把 sigmoid 实现为 `1/(1+exp(-z))` 且输入含大负 z / 极大 gradOutput 的 kernel。
- **路线优先级**：`recommended_fastpath` —— 对 `exp(-z)` 输入加 `Mins` 上界 clamp≤50 防 Exp 溢出，弃 `Reciprocal` 中间 sig 改 `Div(g*(1-step), denom)` 直除让结果在常规量级产生。
- **不适用或未验证场景**：a2/a3 的 `Exp` 溢出阈值与 FTZ 行为未实测，本卡结论仅对 arch35 成立；若 gradOutput 始终为常规量级（非 ±FLT_MAX），subnormal 被 FTZ 也不影响相对误差（在本 trace 容差下通过），本卡诊断仅在极端 gradOutput 下必要。

## 判别方法

**触发（tool_observed）**：在 arch35 上把 sigmoid 从 `exp(bx_clamped)/(1+exp(bx_clamped))`（exp 输入已被 `Maxs(bx, -50)` clamp，sig 量级 ≥1.9e-22 常规数）改写成自稳定的 `1/(1+exp(-z))` 形式后，含大负 z + ±FLT_MAX gradOutput 的用例（如 L1_018，z≈-88）单独跑即 FAIL，最大差异位置 `Actual=0.000000` 而 golden 期望 `sigmoid(-88)×FLT_MAX ≈ 2.06`，`mismatch=2203/441000`、`MARE=1.000000e+00`。

**探针**：单独跑 L1_018 隔离复现（`python3 /tmp/opencode/iso18.py`），确认**单独跑即 FAIL**（排除 cmpmask 跨调用泄漏，见姊妹诊断 `arch35_select_vsel_cmpmask_spr_mask_reuse_count_over_64`）；检查最大差异位置 `Actual` 是否为 `0` 且 golden 期望是否为 `sigmoid(大负 z)×FLT_MAX` 量级（≈2），而非分支段选择错导致的整段错。

**原始观测（tool_observed，verbatim 测试输出，L1_018 单独跑 BEFORE）**：

```
[FAIL] MERE=4.898552e-03 MARE=1.000000e+00 mismatch=2203/441000 nanBoth=0
最大差异位置[11310]: Golden=2.152225 Actual=0.000000
```

**观测**：L1_018 单独跑即 FAIL（无跨调用污染），最大差异位置 `Actual=0.000000` 而 `Golden=2.152225`；该位置 `z≈-88`，golden 期望 `sigmoid(-88)≈6.06e-39 × FLT_MAX ≈ 2.06`（与 Golden=2.152225 吻合）；early 算术版（`Reciprocal` 求 sig + exp 输入未 clamp）PyTorch ST 失败 237/238，含本卡 subnormal 类。

**判别与解释（interpretation，agent 根因分析，非 API 规范主张）**：`1/(1+exp(-z))` 形式引入两个独立的数值问题——(1) arch35 上 `Exp(88)`（`z=-88` 时 `-z=88`）溢出为 `inf` 而非 `1.65e38`，使 `denom = 1 + inf = inf`、`sig = 1/inf = 0`；原形式因 `Maxs(bx, -50)` 把 exp 输入 clamp 到 `≥ -50` 而规避了大正输入溢出，新形式对 `exp(-z)` 的输入未做上界 clamp。(2) 即便 exp 不溢出（`z` 略大于 -88 时 `exp(88)=1.65e38` 可表示），`sig = 1/1.65e38 ≈ 6.06e-39` 是 fp32 子正常数（subnormal）；若先 `Reciprocal` 算出 `sig` 再 `Mul(g, sig)`，中间 `sig` 在 arch35 上被 FTZ（flush-to-zero）清零，`g*0=0`。算式：`z=-88 → exp(88)=1.65e38 → denom=1.65e38 → sig=1/1.65e38=6.06e-39（subnormal，arch35 FTZ→0）→ g*sig=FLT_MAX*0=0`；golden=`FLT_MAX*6.06e-39=2.06`，误差被 ±FLT_MAX gradOutput 放大到 MARE=1.0。`Actual=0` 的量级与 subnormal/FTZ 签名吻合（非分支选择错），且单独跑即 FAIL 排除 cmpmask async 泄漏，故指向数值形式而非控制流。单点归因：`exp(-z)` 形式的 exp 大输入溢出 + subnormal 中间值 FTZ，与 cmpmask async 无关。稳定写法方向：对 `exp(-z)` 输入加 `Mins` 上界 clamp≤50 防 Exp 溢出，弃 `Reciprocal` 中间 sig 改 `Div(g*(1-step), denom)` 直除让结果在常规量级产生。

## 验证信号

- L1_018 单独跑（early 算术版，BEFORE）：`[FAIL] MERE=4.898552e-03 MARE=1.000000e+00 mismatch=2203/441000 nanBoth=0`，最大差异位置[11310] Golden=2.152225 Actual=0.000000（来源平台：950；证据等级：tool_observed）
- early 算术版（BEFORE，`Reciprocal` 求 sig + exp 输入未 clamp）PyTorch ST 失败 237/238（含本卡 subnormal 类）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 仅 Ascend950PR（arch35/DAV_3510，CANN 9.0.0-beta.2）实测。
- a2/a3 的 `Exp` 溢出阈值与 FTZ 行为未实测，clamp 上界 50 与 Div 直除策略推断可迁移但未验证。
- 本卡为诊断规则，修复（Div 直除 + exp 输入 `Mins` clamp≤50）的隔离验证未在 trace 内单独确认（仅 bundled 回归）。

## 坏实践

- 在 arch35 上把 sigmoid 实现为 `1/(1+exp(-z))` 且对 `exp(-z)` 输入不做上界 clamp——`exp(88)` 溢出为 inf 致 `sig=0`。该路线标记为 `banned`。
- 先 `Reciprocal` 求 sig 中间值再 `Mul(g, sig)`——sig 为 subnormal 时被 FTZ 清零；应改 `Div(g, denom)` 直除让结果在常规量级产生。该路线标记为 `banned`。

## 误诊

- 早期把 L1_018 的 `Actual=0` 归因于 cmpmask 分支选错（sigmoid 段被误选）——但 L1_018 **单独跑就 FAIL**（无跨调用污染），且最大差异位置的 `Actual=0` 与 `sig` 的 subnormal/FTZ 量级吻合，分支是对的、数值是错的，才转向 subnormal/Exp 根因。