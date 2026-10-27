---
type: Guide
title: APACE barrier 钩子与同通道完成顺序
description: APACE barrier 钩子与同通道完成顺序的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/communication.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/fusion.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE barrier 钩子与同通道完成顺序

此版本GET PostInit为空，DoCommit先CrossCore再CrossDevice后ReadNbi，DoWait为Drain，DoFinalize再Core→Device。PUT PostInit为Device→Core，DoCommit为WriteNbi，DoWait为非self Drain再Device→Core，Finalize为空。NONE/DEVICE/CORE/BOTH是位组合；Init模板不自动决定所有后续Commit/Wait模板实参，需逐调用核对。

data与scale通信对象可共用channel与TeamBarrier，但独立512B UB工作区和窗口段。样例scale先Commit、data后Commit，只对data Wait，依赖同一有序channel中后一次Drain覆盖先前工作；不能推广到不同channel/队列或未知Drain完成语义。共同barrier只保留协议要求次数，不能两个对象各自无意重复导致跨rank轮次错位。

本端指令返回、DMA发送完成、远端内存可消费和全体rank就绪是不同条件。来源说Drain“只保证本端发出”是该文的解释，具体API完成保证需匹配实现；无论如何单rank Drain不能自动证明所有peer已发布本轮数据。源依赖Core/Device rendezvous与完成事件组合，SyncAll只同步本设备，不能单独替代远端就绪。

GET前置barrier和PUT后置barrier服务各自生产/消费合同，不是仅由GET/PUT名字决定必须多一次同步。任何省略都需已有等价顺序证据。每个hcomm调用返回值都需处理，非零Drain不能静默忽略；共享远端可寻址窗口直接MTE读取是另一条需验证注册/可见性/就绪的兜底数据路径，不能自动等价成功ReadNbi。
