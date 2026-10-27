---
type: Guide
title: 性能目标的方向、单位与基线
description: 明确时延降低、加速比、百分点及复合目标的数学含义。
tags:
- task.optimization
- topic.performance.analysis
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/standards/performance_threshold_guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/architecture/API_AND_MIGRATION.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/architecture/ARCHITECTURE.md
  role: supporting
created_at: '2026-09-11T04:58:58Z'
updated_at: '2026-09-11T04:58:58Z'
platforms: []
---

# 性能目标的方向、单位与基线

指标目标应注明名称、单位、方向、基线条件和比较符。对正基线b，越小越好的相对改善为(b-x)/b，越大越好为(x-b)/b；b=0或未知时该比值未定义，不能默认通过。来源check_success相对分支只实现越小越好，不能直接用于利用率增加，且数据模型GoalConfig.meets_goal只检查相对门限，并未实现另存的绝对/复合合同。

时延8.1→4.5us的降低率为44.44%，加速比1.8x，在相同工作量下吞吐增长80%。来源将performance_delta=0.44同时写44% faster、avg_improvement写speedup，语义含混，应分别命名。百分比从23%升到50%是27个百分点，相对增长117.39%；默认15% absolute increase应明确为15个百分点而非乘1.15。

绝对目标如x<4us和相对目标如改善≥38%不等价，尤其严格小于与小于等于在边界有差异。基线6.5us到4us降低38.46%，89.2到60降低32.74%；文档四舍五入33%不能作为完全等价替代。AND/OR复合条件要明确括号、优先级及每项适用范围，主指标未达标时“次指标全达标即可成功”是可选目标合同，不是性能优化普适准则。

来源默认时延20%、利用率15个百分点、lower ratio降低20%、higher ratio提高25%，以及按M*N*K分10^6/10^9和5/50us档位都是启发式预算。相同体积可能有不同形状、布局、dtype、核数及瓶颈，不能据此承诺达到硬件极限。高利用率不总是越高越好，降低某pipe活动比也可能来自其他阶段变慢；最终目标需对应端到端工作量和成本。

正确性、回归限制与可复现性能应分别满足；至少2次采集只是来源最低次数，不能自动排除抖动或搜索选择偏差。对shape集合保留逐例结果、聚合权重、容许回退及独立复验，不能以总体平均掩盖未授权的关键shape回退。

初始baseline与每轮current baseline应分别保存同一制品、输入及配置的测量。多轮时延降低r1/r2的累计为1-(1-r1)*(1-r2)，不能直接相加；更新baseline不能改写最初承诺。缺失指标、NaN、错误单位或未执行记录应保持未决，不经任意eval表达式解释为成功；可用明确定义的比较与布尔语法表达合同。
