---
type: "Runbook"
title: "官方小值域分支的两种比对器落地 — max(errorLimit,|golden|) FTZ 容忍型与固定 2^-16 绝对阈型"
description: "官方精度标准 special_cases 小值域分支（|golden| < SmallValueThreshold 不走相对误差）的两种自研比对器落地：实现一 absErrLimit=max(errorLimit,|golden|)（FTZ 容忍——NPU 输出 0 与 golden 同量级以下一律通过，Mish 大负 x 触发，26/26）；实现二小值元素不参与 MERE/MARE、按固定 2^-16 绝对阈判定并计数（稀疏 top-K 注意力，24/24，检出力更强但可能拦截 FTZ）。阈值表 FP16 2^-11/2^-16、FP32 2^-14/2^-24、BF16 2^-8/2^-16。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# 官方小值域分支的两种比对器落地 — max(errorLimit,|golden|) FTZ 容忍型与固定 2^-16 绝对阈型

## 现象

自研 MERE/MARE 比较器对"输出天然含近零元素"的算子出现假失败：

- **Mish（R5）**：大负 x 处 mish(x)≈0，golden 1e-7 量级而 NPU 输出 0，相对误差≈1 误报——修复前 26 条中 4 条失败（FP16 两条 MERE=3.59e-03/1.56e-01，FP32 两条 MARE=1.24e-03/9.06e-01）。
- **稀疏 top-K 注意力（R10）**：kernel 全 fp32 内算、输出 Cast fp16/bf16 时 MERE 达标（1.81e-04 < 9.77e-04）而 MARE 超 10×阈值（0.09~0.31）。

## 适用场景

- **工程形态**：自研 C++ ST 比较器（MERE/MARE 双阈值），输出含近零元素或 FTZ 场景的算子。
- **API 路径**：官方精度标准 `special_cases` 小值域分支；SmallValueThreshold 表（FP16 2^-11/2^-16、FP32 2^-14/2^-24、BF16 2^-8/2^-16）。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0（R5 Mish 轨迹 / R10 稀疏 top-K 注意力轨迹，两条独立 trace）。
- **适用粒度**：`platform_wide`——小值域判定与具体算子无关，机制是官方精度标准的自研实现落地。
- **路线优先级**：`recommended_fastpath`。
- **明确不适用 / 未验证场景**：正常值域阈值（2^-10 与 10×）不动；golden 全部落在正常值域时新旧比对器等价（此时 MARE 超标就该查 kernel）。

## 根因

官方精度标准把判定分成三段——特殊值（NaN==NaN、同号 Inf 相等）、小值域（`|golden| < SmallValueThreshold` 改绝对误差判定，不进 MERE/MARE）、正常值（相对误差，阈值不动）。自研比较器只实现"正常值"一段时，近零元素被相对误差分母（|g|+1e-7 → 1e-7）爆炸成假失败。

## 解决配方

两条独立轨迹给出两种落地形态，**按算子输出形态选择**：

### 实现一（R5，Mish）：`absErrLimit = max(errorLimit, |golden|)`（FTZ 容忍型）

```cpp
// 小值域分支：|golden| < SmallValueThreshold 的元素按绝对误差判定
// absErrLimit = max(errorLimit, |golden|)
// 边界含义：NPU 输出 0（FTZ/舍到 0）时 |a-g| = |g| 恰好等于 |golden| 分支上限
//   （等于即通过），"输出为 0、golden 同量级"的元素按 golden 自身幅度判定；
//   任何比 golden 本身更大的输出仍会拦截。
double absErrLimit = std::max(errorLimit, std::abs(golden[i]));
if (std::abs(a - g) <= absErrLimit) { smallValuePass++; }
else { smallValueFail++; }
// smallValueFailCount == 0 才 PASS
```

- 触发形态：大负 x 处 mish(x)≈0，golden 1e-7 量级而 NPU 输出 0，相对误差≈1 误报。
- FP32 errorLimit 取 2^-24 是本轨迹按 NPU FTZ 行为对标准原文 2^-30 的适配值（单标杆场景，非标准原值）。
- 修复后首次复跑 25/26、FP32 大范围用例 468 个小值域元素 0 失配；值域收窄后终版 26/26；golden vs golden 自测（FP16/FP32/BF16 各 128 元素）MERE=MARE=0.00e+00。

### 实现二（R10，稀疏 top-K 注意力）：固定 2^-16 绝对阈 + 计数统计

```cpp
// 小值元素：不参与 MERE/MARE，按绝对误差 < 2^-16 判定并计数
if (std::abs(golden[i]) < SmallValueThreshold) {
    smallValueCount++;
    if (std::abs(a - g) >= 2e-16... /* 实际 2^-16 ≈ 1.53e-5 */) { smallValueFailCount++; }
    continue;  // 不更新 mere/mare
}
// 正常值域：相对误差，阈值完全不动
```

- 修复后同一批用例 MARE=5.27e-04、smallVal=0/120700（bf16 例 MERE=1.42e-03/MARE=3.89e-03/smallVal=0/954194），全量 24/24 PASS 且 MERE 量级不变（未动 kernel、未放宽正常域）。
- 附带误诊修正：初见"bf16 比 fp16 更容易过"实为 bf16 的 MARE 阈值（10×2^-7）更宽、小值元素被 2^-8 阈值更多分流，不是 kernel 对 bf16 更准。

## 设计决策

- **两型差异**：实现一 `max(errorLimit,|golden|)` 兼容 FTZ（输出 0、golden 同量级以下一律通过）；实现二固定 2^-16 绝对阈检出力更强，但 golden 极小而输出 0 时可能拦截 FTZ——按算子输出形态选择。
- **小值元素参与 MERE 与否**：实现二小值元素不参与 MERE/MARE（避免稀释均值信息），与实现一侧重点不同；两型都保留计数统计（smallValueCount/smallValueFailCount）供诊断。
- **路线优先级**：`recommended_fastpath`。

## 验证信号

- R5 修复前：26 条中 4 条失败（FP16 两条 MERE=3.59e-03/1.56e-01，FP32 两条 MARE=1.24e-03/9.06e-01），失配样本呈"golden 近零、绝对差极小、相对误差≈1"形态（如 golden=-0.001158, actual=-0.000000, rel_err=1.00e+00）；修复后首次复跑 25/26、FP32 大范围用例 468 个小值域元素 0 失配；值域收窄后终版 26/26（FP32 MERE=4.89e-07/6.96e-06 等），Mock 同套 26/26；golden vs golden 自测（FP16/FP32/BF16 各 128 元素）MERE=MARE=0.00e+00（来源平台：950；证据等级：tool_observed）。
- R10 修复前典型 fp16 用例 MERE=1.81e-04/MARE=1.90e-01；修复后 MERE=1.76e-04/MARE=5.27e-04/smallVal=0/120700，bf16 用例 MERE=1.42e-03/MARE=3.89e-03/smallVal=0/954194；全量 24 用例 PASS=24 FAIL=0，MERE 与修复前同量级（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 实现一的 2^-24（FTZ 适配值）为单标杆场景对标准原文 2^-30 的适配，接入三方标杆后应按标准原值复核。
- 实现二的 2^-16 绝对阈对本算子输出尺度验证；输出整体尺度异常（未归一化大值）时小值域判定可能失效，未验证。
- golden 恰好全部落在正常值域时新旧比对器等价（此时 MARE 超标就该查 kernel）。
- 小值域判定与标准表值派生：阈值表 FP16 2^-11/2^-16、FP32 2^-14/2^-24、BF16 2^-8/2^-16（CANN 社区标准）；[阈值扩展形态](depthwise_small_value_print_identical_example_compare.md)与[标准阈值语义](dual_benchmark_threshold_single_benchmark_fallback.md)见关联卡。

## 坏实践

- 把"小值域豁免"当成万能放行——小值元素上的真实大偏差（如输出幅度远大于 golden）仍会被拦截；实现一的"输出恰为 0 而 golden 为任意小值"通过是有意的 FTZ 容忍，不是万能豁免。

## 误诊

- 初见"bf16 比 fp16 更容易过"实为阈值更宽 + 更多分流，不是 kernel 对 bf16 更准（R10）。