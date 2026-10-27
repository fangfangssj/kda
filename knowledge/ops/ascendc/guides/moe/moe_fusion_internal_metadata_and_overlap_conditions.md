---
type: Guide
title: MoE 融合内部元数据与重叠条件
description: MoE 融合内部元数据与重叠条件的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/common-reading-method.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/mega-moe-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE 融合内部元数据与重叠条件

Dispatch负责route/slot/发送/ready，expert阶段负责输入组织、GEMM/FFN/激活/量化，combine负责来源恢复/加权/输出。一个kernel可合并职责、消除外部expandX/expandIdx或计数tensor，但对应映射与有效长度仍须保存在内部地址表、workspace、寄存态或状态协议中，不能因接口消失就省掉来源恢复。

跨阶段复用buffer需注明每阶段承载的数据shape/dtype和最后消费者。通信payload、expert输入/中间、聚合acc、输出staging及共享控制区各有生命周期。最优发送布局未必适合GEMM和最终token恢复，布局转换成本需全链计入；一个函数可以同时包含Scalar、DMA及Vector工作。

按chunk/expert ready逐步驱动FFN或聚合可减少空泡，但提前回搬仍需确定全局输出offset。若输出要求expert/source有序，未知前缀会阻止随ready任意追加；可用固定槽、预计算prefix或后续重排换取并行，权衡额外内存与搬运。分层统计可减少全体核等待，但仅在依赖足够时。

pingpong/queue出现不证明已形成稳态重叠：要有独立buffer、先行发下一搬运、consumer完成/复用credit以及实际时间窗口。多职责放同一个kernel也可能仍整阶段串行。核内执行队列时序图的逻辑时间不能当实测耗时，同一队列顺序与硬件流水内部重叠应区分，不以“同lane绝不并行”推广到所有微架构。

优化应检查热点expert/rank、Scalar轮询分支、最终输出顺序与单核统计尾部；按未就绪子集轮询可减少扫描，但完成集合和epoch必须准确。共用粗粒度barrier可能留下瓶颈，去掉同步需证明依赖已由其他机制保障。图片格式、读文档步骤、交付图表数量和Reviewer模板留在Skill。
