---
type: Guide
title: MoE 容量公式与路由假设
description: MoE 容量公式与路由假设的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/spec-template.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/dispatch-dataflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/combine-dataflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/index.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE 容量公式与路由假设

来源A表示本rank最大接收routed item容量，R_actual才是本轮有效数。expandX[A,H_expand]、expandIdx[A,3]及逐条scale只消费前R_actual行；打包fp4x2的H_expand=ceil(H/2)是存储单元数，逻辑H与实际字节另算。payload宽度不含triple，通信记录可Align(payload,32B)后附[srcRank,token,topK]并整体按真实DMA要求预留，不能将元数据写入expandX隐藏维。

MoE均匀部署例localE=moeExpertNum/(EP-sharedRanks)，需分母>0与整除；dstRank=expertId/localE+sharedRanks，localExpert=expertId%localE。shared组例每专家g=sharedRanks/sharedExpertNum个rank，target=srcRank%g+sharedExpertIndex*g，需g>0、整除且合法覆盖。sharedExpertNum=0应走独立无shared分支，不能仍做除法。

源MoE容量下界Bs*EP*min(localE,K)或globalBs*min(localE,K)，依赖每token topK专家不重复及globalBs是实际总batch上界。重复expert若允许，同一expert可收多条路由，按每source每expert仅Bs个slot可能溢出。mask可减少实际量但不自动改变支持域最大容量，drop/capacity策略须显式。

源共享容量Bs*EP*sharedExpertNum/sharedRanks=Bs*EP/g是均匀分组表达；srcRank%g路由在EP不能被g整除时各承载rank可能收到ceil(EP/g)*Bs，不能截断平均数当每rank上界。应证明部署整除或按实际source集合取max。数据区按sourceRank→localExpert→slot，而状态区按localExpert→sourceRank；按上限预留和实际计数寻址不同。

旧overview把assistInfo写成A×128又说前A×3有效，单位/布局不充分，不能据此替换sample expandIdx[A,3]或公开API版本。三元组字段顺序、dtype与实际文件/接口必须固定，返回时检查rank/token/topK值域和唯一对应。
