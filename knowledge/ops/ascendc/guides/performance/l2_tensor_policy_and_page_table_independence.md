---
type: Guide
title: L2 Tensor 策略与页表语义的独立性
description: 保留选择性缓存的可测条件，区分框架标志、实际访存 hint 和分页地址映射。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P52_l2_cache_hint.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P61_l2_cache_optimization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P52_l2_cache_hint.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P61_l2_cache_optimization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P52.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P61.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/data_structures/global_tensor/set_l2_cache_hint.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md
  role: constraint
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# L2 Tensor 策略与页表语义的独立性

SetL2CacheHint配置对应GlobalTensor的缓存访问策略，已有接口记录默认NORMAL、可DISABLE，而PERSISTENT是所述版本暂不支持的预留模式。不是全局一键关闭芯片L2，也不能因另一个句柄已配置就推断所有别名访问采用相同策略；应追到真正消费该配置的搬运路径。

重复使用的权重/KV可能受益于L2复用，一次性流式数据也可能挤出更有用内容。blockDimM==1、输出大小大于L2或多核读同一权重，只能提出候选假设，不能分别证明缓存抖动；共享读取同样可能提高命中。数据总量、活跃工作集、复用距离、写入策略及并发访问应结合实际测量判断，不能一律“小KV关闭、大KV开启”。

来源l2CacheOffFlag约定true关闭、false开启，是该框架字段的语义，不是独立SDK调用。仅改变一个未被后续代码读取的bool不会改变硬件行为。来源将该字段同时解释为“双页表开关”，但没有给出地址映射、双页表分支与hint的完整关联；不能据此宣布关闭双页表等于开启L2。

PageAttention的逻辑页到物理页映射属于正确性合同。任何页表模式变化都需保持有效token、K/V地址、shape及生命周期，不能为了缓存实验简单删掉一种必需映射。不同Tensor/路径可以采用不同策略，P52与P61既不是全局互斥，也不保证在同一个访问上叠加仍生效；取决于最终传递到指令的配置。

来源报告250μs→210μs，时延下降16%；缺少本批输入、硬件、版本和采集原始证据，只保存为待复现案例，不作为通用收益。15–25%带宽提升同样不是保证。比较NORMAL/DISABLE时应固定语义与输入，测命中、GM流量、带宽和完整时间，并观察并发算子的影响。

缓存hint不替代数据发布与同步；修改后仍须检查输出。核对API版本的具体模式名字，不能由grep中的CACHE_MODE_ENABLE猜出真实枚举。基础入口见[SetL2CacheHint](../../apis/simd_api/basic_api/data_structures/global_tensor/set_l2_cache_hint.md)。
