---
type: Guide
title: 精度复检的 Bootstrap 中位数区间与样本合同
description: 保存误差比样本和 Bootstrap 中位数判据，区分复检次数、重采样次数、均值区间及文档与实现的分位点差异。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/engine/src/scripts/orchestrator/precision/cannbot_standard/reference/special_cases.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/engine/src/scripts/orchestrator/precision/cannbot_standard/scripts/confidence_interval.py
  role: implementation
created_at: '2026-09-14T07:15:54Z'
updated_at: '2026-09-14T07:15:54Z'
platforms: []
aliases:
- Bootstrap
- confidence_interval
- analyze_recheck_ratios
---

# 精度复检的 Bootstrap 中位数区间与样本合同

固定 AOG 复检接收已计算的误差比样本，分析它们的中位数是否表现出系统性恶化；函数不生成输入、不运行算子，也不计算逐元素误差。来源示例采用 `ratio_i=npu_error_i/max(benchmark_error_i,1e-10)`，而函数注释另写 golden_error；必须先固定实际分母的含义。分子、分母需来自同一输入、同一误差指标、同一有效域与参考语义，不能混用 [三方误差诊断](three_way_comparison_and_diagnostic_hypotheses.md) 中不同指标的比值。

分母为零或极小时，下限会改变比值和门限的含义。它不是数学精度真值；双方同错、参考本身不可靠、样本选择偏差和有意移除失败运行都不会因比值统计而自动消失。

## 样本数、重采样数与统计量

令 N 为实际有效复检样本数，B 为 Bootstrap 重采样次数。固定脚本把一维比值数组按有放回方式抽取 N 项，计算每次样本中位数，共做 B=1000 次，再从排序后的中位数集合取区间端点。来源流程图写 2000 次，不能与当前函数的 B=1000 混同。`analyze_recheck_ratios` 中又分别调用区间函数和系统误差检查，导致一次调用实际跑两批各 1000 次；最终报告来自第二批，增加随机状态消耗不等于把 B 变成单批 2000。

默认 confidence_level=0.95、alpha=1-confidence_level，排序结果为 m，实际端点索引公式为：

```text
lower_idx = max(0, ceil(alpha/2 * B) - 1)
upper_idx = min(B-1, ceil((1-alpha/2) * B) - 1)
CI = [m[lower_idx], m[upper_idx]]
```

公式使用零起始索引，不能直接换成任意库的插值分位数并声称结果完全一致。当前 Python 浮点计算下，默认值给出索引 25 和 974；来源注释写 24 和 975，二者不一致，其中下端还受 `1-0.95` 浮点舍入后 `ceil` 的影响。需要复现同一判定时，应保存实现版本、随机种子/状态、N、B、置信水平和实际端点规则，而不是只写“95% CI”。

另一个 `parametric` 分支计算 `mean ± t_critical*std(ddof=1)/sqrt(N)`，以均值为统计量；它不等于 Bootstrap 中位数区间，也不被默认复检函数选用。参数假设与样本结构必须分别满足。

## 判据与有效性

该策略以 `CI_lower<=1.0` 通过，只有下限大于 1.0 才按该规则不通过。区间跨过 1 时仍可能通过，例如 200 项由 100 个 0.5 和 100 个 1.5 构成，固定 CPU 探针得到区间 [0.5,1.5] 并通过。它表示该门限没有拒绝当前样本，不证明性能/精度等价、全部用例合格，或高置信度地保证误差不大于参考；不得替代必须逐例满足的精度验收。

N<200 在来源实现中直接返回无效/复检不通过，是该工具的最小样本政策，不是所有 Bootstrap 的普遍可靠性分界。N>=200 也不充分：函数没有检查有限性、非负误差比、独立性或置信水平合法域；全 NaN 数据可得到 `is_valid=True` 和 NaN 端点，二维数据会在 NumPy 一维采样处失败。`len(array)` 只取首维长度，不等于总样本数。

先验证一维、有限、合法比值和实际有效运行数；保留异常/失败运行的独立状态，不将其静默删除或写成 1。复检样本的生成机制与 Bootstrap 内部重采样分别记录；相关样本不因随机抽取自动独立。验证应覆盖 N=199/200、零分母、非有限值、二维输入、固定随机状态和门限两侧，并确认整体测试仍遵守原算子合同。以上是固定代码的统计行为与 CPU 反例，不是 NPU 实测或已验证的总体统计保证。
