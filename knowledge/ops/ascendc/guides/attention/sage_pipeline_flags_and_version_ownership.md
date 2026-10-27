---
type: Guide
title: Sage 四阶段流水与版本事件所有权
description: 保留mode4槽位协议，限定Bootstrap与双重等待结论。
tags:
- task.implementation
- topic.attention
- topic.precision
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-rules.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-recipe.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-troubleshooting.md
  role: supporting
created_at: '2026-09-11T05:16:48Z'
updated_at: '2026-09-11T05:16:48Z'
platforms: []
---

# Sage 四阶段流水与版本事件所有权

来源混合launch为__mix__(1,2)，每AIC配两AIV且各处理64行；AIV逻辑AIC编号GetBlockIdx()>>1、子块GetSubBlockIdx()仅在该启动合同下解释。独立__vector__ merge不能依赖混合子块索引，候选sub=blockIdx&1、task=blockIdx>>1、步长blockNum>>1要求成对核数且步长非0。

四相位流水按it推进QK(it)、SM(it-1)、PV(it-2)、RS(it-3)，循环到P+3排空；单步数据依赖QK→SM→PV→RS不允许违反，但不同step可重叠。QK/S、PV/O各2槽，P到L1用4槽。来源mode4候选C1_V1={0,1}、V1_C2={2,3,4,11}、C2_V2={5,6}、MM2={7,8}、MM1={9,10}、PV_CONSUMED={12,13,14,15}；AIC通知双AIV时set id及16+id，AIV set单id由该模式路由，AIC等双方id。此表仅适用对应组件/模式，不能挪到A2或其他事件分配。

MM1表示AIV已消费sUb，MM2表示已消费oUb；每步只释放本步槽，两个槽都set会使未来等待提前通过。P在UB经MTE3送L1后，下一Cast覆写pUb需等MTE3完成；AIC读完pL1后还需PV_CONSUMED，两个缓冲的保护不能互相替代。首次填充可以跳过不存在消费者的等待，复用才wait，按槽真实使用次数判定，不能不分任务边界机械用it>4。

来源2026-08所谓“新版”BlockMmadTla在内部wait MM1/MM2；若外层再wait同一token会死锁。需读取实际安装组件确定等待所有权，不能一概删除所有版本外层wait。来源冷PIPE_FIX预置不传播和AIC PIPE_ALL必挂死未附本批底层验证；保留为指定环境故障线索。通用要求是核型支持的pipe集合、实际producer完成及bootstrap/steady/drain计数正确，不能推广成所有AIC永远禁止任何Barrier。

Set/Wait<V_MTE3>表达V生产后MTE3可消费，不是DMA完成后V可覆写；后者需要MTE3→V或实际等价完成协议。来源“kernel退出不保证MTE3排空”不能脱离runtime/接口强制所有出口插PIPE_ALL；需要确认已提交写出与下个kernel/Host读的完成合同。跨launch残留token是单发对、双发错的一个候选，不是唯一根因。

新增tiling字段要保持Host序列化与Device逐字段拷贝一致，漏字段会未初始化，P0只是可能表现。CMake依赖追踪是否缺失依ASC工具链，给add_executable列头文件或touch TU不是所有版本根本修复；应验证真正重编及加载制品指纹。
