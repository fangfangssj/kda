---
type: "Runbook"
title: "fp16 小值输出 MARE 假失败的两条边界 — 元素豁免 allAbsOk（正）与大 K 事后放宽（反例红线）"
description: "分组矩阵乘 fp16 输出含大量贴 atol 边界小值元素（golden 1e-4~4e-3、abs diff 1e-6~8e-6）使 MARE 假失败：正处置=MARE 只统计 |g|>atol 元素 + 全元素绝对差兜底 allAbsOk，真缺陷仍 FAIL 作反证；反例红线=fp16 大 K（K∈[1023,8192]）MARE 回退条件改逐元素 absDiff<=atol OR rel<=rtol 属失败发生后的事后放宽（测试侧放松），正确次序=判别形态 → dtype/tw 对照钉死非 bug → 事先声明阈值模型落地。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# fp16 小值输出 MARE 假失败的两条边界 — 元素豁免 allAbsOk（正）与大 K 事后放宽（反例红线）

## 现象

分组矩阵乘 fp16 ST（CANN 9.1.0）出现两类同源失败：

- **L0 阶段（R4）**：纯相对误差（分母 |g|+1e-7、全元素最大）口径下 MERE=2.16e-06 而 MARE=1.18e-01（阈值 9.77e-03）FAIL，失配元素全部小 |golden|（1e-4~4e-3）且 abs diff 1e-6~8e-6。
- **L1 阶段（R14）**：12 条失败全部 fp16、K∈[1023,8192]、MERE 比阈值小 2-3 个数量级而 MARE 略超 10×阈值（L1-0289 K=8192 MERE=7.46e-06/MARE=1.03e-02），失配元素 absDiff（1e-6~1e-5）<< atol 且 golden 贴 atol 边界。

## 适用场景

- **工程形态**：fp16 分组矩阵乘 ST 验收（MERE/MARE 双阈值、allAbsOk 兜底），Real 模式。
- **API 路径**：比较器 `rel = |a-g|/(|g|+1e-7)`；MARE 聚合与 allAbsOk 兜底。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0（R4 L0、R14 L1 同 trace）。
- **适用粒度**：`operator_family`——fp16 输出含大量贴阈值小值元素的归约/矩阵类算子。
- **路线优先级**：正处置 `recommended_fastpath`；事后放宽属反例（banned）。
- **明确不适用 / 未验证场景**：K<1000 量级与其他 dtype 组合未验证；MERE 也超阈或 absDiff 大（>atol 且 rel>rtol）时逐元素判定照样判负。

## 根因

fp16 输出天然含大量近零 golden 元素（分组矩阵乘的尾块/对齐区），纯相对误差分母 `|g|+1e-7` 在 |g| 贴 atol 边界时把 FP32 舍入噪声（abs diff 1e-6~1e-5）放大成 MARE 超标。kernel 侧对该 MARE 逐位不变（R4：SetBType/SetTail/SetOrgShape/SetFixSplit 全无变化）——旁证根因在比较器而非 kernel。

## 解决配方

### 正处置（R4）：MARE 只统计 |g|>atol + 全元素绝对差兜底

```cpp
// MARE 只统计 |g| > atol 的元素；atol = threshold
// 另加全元素绝对差兜底 allAbsOk：任何元素 |a-g| > atol 即 FAIL
bool allAbsOk = true;
for (size_t i = 0; i < size; ++i) {
    double g = abs(golden[i]);
    double a = abs(actual[i]);
    double absErr = std::abs(a - g);
    if (g > atol) {  // 只在正常量级元素上更新 MARE
        double rel = absErr / (g + 1e-7);
        if (rel > mare) mare = rel;
    }
    if (absErr > atol) allAbsOk = false;  // 全元素绝对差兜底
}
bool pass = (mere < threshold) && (mare < 10*threshold) && allAbsOk;
```

修复后 13 用例子集 13/13 PASS、对齐 shape 无回归；**反证**：同会话真实缺陷（跨用例缓存干扰 MARE=2.53e+03、多输出失败）在混合判据下仍 FAIL——证明豁免不掩盖真缺陷。

### 反例红线（R14）：大 K 事后放宽必须按正确次序

12 条 fp16 大 K 失败后，比较器 MARE 回退条件由 allAbsOk 改逐元素 `absDiff<=atol OR rel<=rtol`，L0+L1 全量 687/687——**该改动属失败发生后的事后放宽（测试侧放松），留档为反例**：

- dtype 对照（同 shape bf16 全过：L1-0495 MARE=7.81e-03 压线 PASS、K=8192 bf16 7.46e-03/5.49e-03 PASS）与 tw=false 对照（K=8192 fp16 MARE=3.38e-03 PASS）构成独立旁证；
- 正确次序是：判别形态 → 用 dtype/tw 对照等独立证据钉死"非 bug" → 再以事先声明的阈值模型落地比较器；
- "非 bug"结论当时为推断（kernel 侧 fp32 累加对照未做），不得当作修复配方复用。

## 设计决策

- **为何豁免的同时保留 allAbsOk 兜底**：近零元素豁免只消"假失败"，allAbsOk 保证近零区真实大偏差（>atol）仍被拦截——两者互补，缺一不可。
- **为何 R14 的逐元素放宽是反例而非另一配方**：它把"真 bug"与"指标病"的区分从判别前移到结果后（用放宽让全量绿），且无 kernel 侧累加对照的独立证据；R4 的 allAbsOk 是事先声明的模型，R14 是事后补的判据。
- **路线优先级**：R4 属 `recommended_fastpath`；R14 属测试侧放松的反模式（文档中作为红线记录）。

## 验证信号

- R4 修复前：MERE=2.16e-06 而 MARE=1.18e-01（阈值 9.77e-03）FAIL，失配元素全部小 |golden|（1e-4~4e-3）且 abs diff 1e-6~8e-6；kernel 侧修改对该 MARE 逐位不变（旁证根因在比较器）；修复后 13/13 PASS、对齐 shape 无回归；反证：真缺陷（跨用例缓存干扰 MARE=2.53e+03、多输出失败）在混合判据下仍 FAIL（来源平台：950；证据等级：tool_observed）。
- R14：12 条 fp16 大 K 用例失败全满足"MERE 小 2-3 个数量级 + MARE 超阈 ≤37% + absDiff（1e-6~1e-5）<< atol"；dtype/tw 对照通过；逐元素混合判定后批次 162/162、164/164 转绿，L0+L1 合计 687/687（来源平台：950；证据等级：tool_observed——转绿信号与对照观测为实测，"非 bug"归因为推断）。

## 失效边界

- R4 的 atol=threshold 取值对"输出幅值远大于 1"的算子偏松（1e-3 绝对差在大输出上无感）——输出幅值大且关注小值精度时 atol 应按输出尺度归一（未验证）。
- R14 的 fp16 大 K 形态判别只覆盖三条件同时满足（MERE 小 2 个数量级以上 + MARE 超阈幅度小 + absDiff << atol）；MERE 也超阈或 absDiff 大（>atol 且 rel>rtol）时逐元素判定照样判负。K<1000 量级与其他 dtype 组合未验证。

## 坏实践

- 调指标不留反证用例（无法证明没有放行真缺陷——R4 以已知真缺陷用例在混合判据下仍 FAIL 作反证）。
- 失败发生后"顺手"放宽比较器判据到全绿（R14 被记录为反例红线）。
- 见到精度 FAIL 先做失配元素形态分析（MERE 量级 + abs diff 量级 + |golden| 分布）再决定动哪一层——R4 曾按"非对齐维度 tail 处理缺陷"与"转置 tiling 不一致"方向修 kernel（SetTail/SetOrgShape/SetFixSplit/SetBType），MARE 逐位不变，判定根因不在 kernel。

## 误诊

- R4 曾按"非对齐维度 tail 处理缺陷"与"转置 tiling 不一致"方向修 kernel，MARE 逐位不变——先做失配元素形态分析再动层。
- R14 初版被疑为"长 K 累加漂移致真缺陷"，dtype/tw 对照才确认"非 bug"（当时为推断，缺 kernel 侧 fp32 累加对照，未完全钉死）。

## 预防

- 写 fp16 输出算子的比较器时，近零豁免（MARE 只统计 |g|>atol + allAbsOk 兜底）应在 ST 落地前按精度模型声明，而非失败后补。
- dtype/tw 对照是"非 bug"判别的低成本独立证据，遇到近零 MARE 假失败先做对照再动判据。

## 未决事项

- R14 的"fp16 大 K 累积漂移"机制估计（sqrt(K)×2^-11）为推断，未做 kernel 侧 fp32 累加对照验证。