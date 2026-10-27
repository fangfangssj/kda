---
type: Guide
title: Varlen L1 Carry 与 SameAB 约束
description: 保存驻留优化条件，修正跨batch无条件复用与容量常量。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-varlen-handcraft-recipe.md
  role: primary
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# Varlen L1 Carry 与 SameAB 约束

来源needL1Carry候选条件为非SameAB、非FP32、无Rope、S1基本块和D/Dv在实现上限内，并切到UnpackInit/ProcessL1Carry及PairGetS1LoopRange。shareL1Size设0表示改变组件间资源分配，不是物理L1变0或驻留数据无需内存。

KV整batch驻留需要真实KV字节加P/Q、阶段和对齐满足目标L1；只看S1/D阈值若未同时限制KV长度仍不足。可以跨同batch的多个Q任务复用相同KV，跨batch通常数据不同需重新加载，除非证明输入内容/身份确实共享。来源“跨task/batch复用”不能理解成保留上个batch的KV继续算。

source首次不开Carry、shareL1Size正常1MB是其工程建议与配置描述，不等于A2物理L1容量1MiB；必须按实际API字段语义和资源查询核算。禁用L1Reuse奇偶对而改needL1Carry也属于两种组织差异，不是varlen数学要求。

SameAB指Q/K/V为同一GM张量且shape/stride/语义兼容，Sq==Sk仅为必要关系之一；一般self-attention也可能三次投影产生不同Q/K/V，不能只因self-attention标签自动共享加载。Q_EVENT0=2/KV_EVENT0=4/P_EVENT0=6是该类自管候选事件分配，不能与其他组件重复占用。

aux max/sum[sum Sq,Hq,8]和TND输出应与每batch有效query总数一致。直接entry的REGIST、混合核型和userWorkspace合同来自选定实现，不因增加qList/kvList两个指针就自动完成全部长度适配。
