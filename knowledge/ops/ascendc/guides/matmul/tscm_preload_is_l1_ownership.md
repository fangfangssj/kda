---
type: Guide
title: TSCM 预加载与 L1 所有权
description: 明确 TSCM 对应 L1 的逻辑空间，避免把预加载理解成额外缓存容量。
tags:
- task.optimization
- topic.memory
- topic.data_movement
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P78_onchip_cache_matmul_acceleration.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P78_onchip_cache_matmul_acceleration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P78.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/tscm_input_matmul.md
  role: constraint
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# TSCM 预加载与 L1 所有权

TSCM输入Matmul由用户管理并预先填充对应L1逻辑空间，Matmul直接消费所传地址，省去对象内部重复GM→TSCM搬入。TSCM不是独立于L1的一份可缓解L1容量不足的额外硬件缓存；预加载的GM流量仍真实发生，应考察跨对象/任务复用和流水重叠，而非声称免去全部输入搬运。

设置为TSCM输入的相应矩阵应在该接口要求的范围内全载并保持到最后消费。来源两个TBuf槽仅表示一个策略布局，需按实际Q/K/V/中间状态尺寸及并发寿命分配。headDim大可能增加容量，也要同时考虑M/N/K、dtype、padding和其他使用者，不能仅据headDim或“训练/Decode”决定是否可用。

源骨架设置A/B均为TSCM却只展开Q的加载，实际实现必须保证B也已加载且地址正确；UNSPLITK条件下才预载时，其他分支需有自己的合法输入路径。SetAType/SetBType属于所用tiling或框架对象合同，不能将其与Kernel IterateAll参数混成一个通用运行时接口。

NZ仅在消费端要求该布局时准备；若原数据已经是正确物理格式，无需重复转换。ND→NZ的准备、完成发布、片上slot回收及跨对象double buffer均由真实地址和事件管理。来源L1 Carry只给概念和场景标签，没有完整接口，不能据此补造API。可与GM路径通过合法分发共存，源码仍含GM并不表示优化失败。

验证全部输入全载、不同Matmul对象复用、slot轮转、非整分形尾部及回退，比较含准备和转换的完整时延；来源“带宽远高于GM→L1”没有同口径路径测量，不能当作量化收益。
