---
type: Guide
title: Matmul 搬运模板与 K 重排合同
description: 保存 MDL、NBuffer33、预取和 UnitFlag 的不同前提及 Host/Kernel 一致性。
tags:
- task.optimization
- topic.memory
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P86_matmul_api_transport_efficiency_tun.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P93_matmul_k_stagger_access.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P86_matmul_api_transport_efficiency_tun.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P93_matmul_k_stagger_access.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P86.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P93.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_nbuffer33.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_multi_core_k_stagger.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_unit_flag.md
  role: constraint
created_at: '2026-09-11T02:13:08Z'
updated_at: '2026-09-11T02:13:08Z'
platforms: []
---

# Matmul 搬运模板与 K 重排合同

MDL大包搬运、NBuffer33调度、Preload预取、UnitFlag和K轴错峰分别解决粒度、流量分布、等待或地址争用问题，不能因都在Matmul内部就机械同时开启。每项需要目标SDK模板能力、Host tiling与Kernel实例一致；存在CFG_NORM回退也不说明另一分支的优化失效。

已有NBuffer33说明中，A的最多3×3基本块在L1保持，B以相应3×1块并配合double buffer推进N。其合同为MDL配置、A/B在GM、纯Cube模式、IterateAll输出，stepM/stepKa/stepKb<=3且stepKa=stepKb=ceil(singleCoreK/baseK)，A常驻与B实际加载空间合计不超L1。Host在GetTiling前配置scheduleType，Kernel还需相应policy，仅在Host填写N_BUFFER_33不足以完成选择。

Preload把下一轮加载安排到可利用的MTE2间隙，仍需有效预取范围、可用slot和消费完成后回收；并不是消除首轮加载或无限增加L1。UnitFlag用于Mmad与Fixpipe的细粒度交接；已有高阶案例限制Norm/IBShare/MDL，并对混合L0C→GM与L1→GM流水、L0C累加时多Iterate单GetTensorC给出限制。实际版本需核对，不能只设置一个名为UnitFlag的变量。

K错峰通过MatmulConfig.enableKdimReorderLoad改变不同核的K搬入起点。来源及已有案例要求MDL、多核、A或B在GM且K方向数据不能同时搬入并保持L1。全载判断需包括单核M/N、dtype、A/B分区、step/depth及容量，不能拿K元素数直接与L1字节数比较；K<=4096不构成普适无收益阈值。

若手动实现K置换，A/B对应的K片必须一致、全部有效K贡献恰好覆盖，并处理尾块。加载顺序和计算累加顺序应区分：若实现也改变累加顺序，可能有浮点误差变化，不能用“不改数学逻辑”排除该原因。单核缺少多核错峰目的，多个核也未必存在同一热点，需要实际地址与带宽证据。

来源90→69.87微秒为MTE2约下降22.37%，98.72→85.68为整体约下降13.21%；34.4%→41.7%是增加7.3个百分点、相对约21.22%，不是同一种提升口径。本批没有复现这些性能。验证tail、模板分支、组合兼容和结果后，分别测MTE2、Cube/Fixpipe等待及完整时间，不将配置赋值顺序当作所有字段的固定依赖。
