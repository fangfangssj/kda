---
type: Guide
title: Tiling 搜索参数角色与语义约束
description: 从参数分析流程中抽取可复用约束，纠正只按常量来源放宽搜索。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/parameter-analysis.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/aiss-tiling-solver/SKILL.md
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# Tiling 搜索参数角色与语义约束

可区分固定问题输入、独立可调参数和派生字段；派生字段随候选重算，Host与Kernel必须一致。Kernel tiling字段是重要入口，但仅在Host使用的参数也可能影响布局变换、算法选择、launch或workspace，不能一律排除为非调优维度。

参数限制可能来自硬件容量/接口字段、SDK支持、算法语义、并发协议、物理布局或经验性能阈值。字面量不自动是可放松经验值：如mask宽度、16×16转置基本块、事件编号、数组容量和quant group都可能硬编码却承担正确性合同。平台配置字段也可能表示用户保留或软件上限，需追踪实际消费者与定义，而非二分“平台=硬件、常量=经验”。

枚举应保持所有正确性约束，只有有证据属于性能策略的截断才适当扩大；还需检查dtype/format/shape注册、地址积溢出和资源同时存活。多个算法能力谓词可同时命中，比较额外参数和覆盖范围，fallback标签不保证真的支持全部输入。占位导航与已存在脚本可能不同步，不据索引一句暂未收录丢弃正文知识。

UB、L1、L0A/B/C和通信GM分别按真实物理布局核算，Cube不一定使用UB作为每次输入暂存，Vector也可参加混合路径；“主切分总在L1”不是跨所有融合算法的硬规则。算子名分类只是检索入口，GroupNorm本身是归一化归约，不因路由表放在FA就具有Attention语义。

验证候选的原始参数、派生值、全部分支与实际Kernel消费，区分可行性与排序成本模型。特定JSON/YAML输出模板、先后步骤和强制搜索范围属于Skill编排，不进入知识规则。

## AISS 求解状态的范围

固定来源 AISS 接口给出 ok、unsat、timeout 等状态。应相对于送入求解器的变量、约束、平台参数和搜索范围解释：ok 表示该模型返回候选，不能单独证明目标 Kernel 实现正确或全局最优；unsat 不能证明硬件无法实现，可能是当前布局、算法、取值范围或错误约束排除了实现；timeout 只表示给定预算内未完成所需求解，不等同于 unsat。

候选还需重算派生字段并核对真实 Kernel 的容量、对齐、同步、分支和整数范围，模型中的字段必须由实现实际消费。来源外部 latest 工具没有在本文固定二进制实现，因此不把其算法、最优性或安装过程作为已验证知识。
