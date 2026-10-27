---
type: Guide
title: APACE 等待掩码与 pipe 等待范围
description: APACE 等待掩码与 pipe 等待范围的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/communication.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/fusion.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE 等待掩码与 pipe 等待范围

来源AIC CalcDependTileIdx从计算block末行mPos+blockM-1推算通信tile，钳到总tile-1；uint32 waitedMask记录已Wait位，退出前补等未消费的tile以平衡Set/Wait。该表示只有32位，必须tile总数<=32或替换表示，固定flagId不会扩展位掩码。总tile为0时下溢、位移>=32时未定义/丢位，均需在Host和消费者边界处理。

不同pipe的Set和Wait可以正确配对，不能要求PIPE参数相同。PUT参考是AIV MTE3 Set→AIC MTE2 Wait，目的是门控后续搬入。AIC输出完成FIX Set→AIV标量Commit则需门控标量流，来源建议无模板CrossCoreWaitFlag(flagId)。在某pipe插入wait不自动阻止后续标量下发UDMA。architecture GET图的PIPE_FIX Wait与fusion无模板Wait不一致，应以实际消费者及对应API合同判定，不能原图即用。

全参与核（包括无计算tile核）仍须执行协议要求的通知/消费次数。SyncAll应在所有参与者都可到达处，置于只通信核guard内会缺席；SyncAll<true>的AIV参与与<false>混合参与不同。多个同步阶段要按生产依赖排序，先全producer完成再对consumer发布，画等待边检查环。

flag通道数、每通道未消费积压、软件tile掩码是三个限制。固定flag0/1允许重复轮次但仍需credit/producer节流以防积压上限；源“无回压仍积压总是1–2”没有速度差证明。逐轮id还需避开所用SyncAll保留id。数值13的保守范围不能套所有模式，须按最大实际id和保留集合计算。
