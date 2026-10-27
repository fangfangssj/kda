---
type: Guide
title: APACE 先计算后通信的三级流水生命周期
description: APACE 先计算后通信的三级流水生命周期的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/fusion.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/architecture.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE 先计算后通信的三级流水生命周期

可复用模式为AIC按轮写完整输出坐标的staging，AIV把各target行段PUT到对端source槽，再在接收端归约。staging为完整[M,N]连续输出时与target chunk布局一致，可避免额外重排。源“compute-first不需回压”仅在staging/Win全量独占且无旧槽重用时成立；ring改造需另加释放依赖。首tile计算暴露，后续mm(t+1)可与comm(t)/reduce(t-1)重叠，不能说整个mm段永远无法掩盖。

严格分离方案后R核通信、前blockNum-R核归约，必须有正数归约核或明确fallback。每轮全AIV门控/SyncAll，通信核Commit+Drain同时归约核处理上一轮，再全核SyncAll、有效单barrier job跨rankfence、SyncAll。最后归约T-1，T=1不漏尾。时分复用全AIV顺序通+归约也能正确，只是开销不同；T1仍有启动/同步，不能无测量宣称零额外开销。

FragmentTensor每轮构R个分段，一次计算R*curTileM行可减少R*T次Params和scheduler重建；32上限是该实现fragment数而非T。localLast把远端逻辑fragment先排、本地后排，在远端完成发flagA，全轮完成发flagB。源规定A/ScaleA重排而cFragAddrs保持原rank序，是其下游映射的专用合同；必须证明scheduler逻辑rank到C表索引转换，不能普遍要求所有A/C表一起重排或永远不重排C。未跨边界核补发通知，不能让receiver缺计数。

localLast是减少通信启动等待的优化，在来源协议中双flag依赖该边界；重设计协议时可以有别的正确实现，不将“移除阻塞级错误”迁作普遍物理规则。减少problem区间与缩小base tile不同，拆问题仍可能额外重复B搬入、重置缓存、增加tail及标量开销，不能保证MTE2总量不变或cache必提高。
