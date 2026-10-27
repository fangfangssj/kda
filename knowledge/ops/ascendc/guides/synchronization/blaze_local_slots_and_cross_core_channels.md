---
type: Guide
title: Blaze 局部槽位与跨核通道的分别闭合
description: 保留事件申请、计数轮换与阶段参与者，纠正固定 ID 和全局屏障的过度归纳。
tags:
- task.design
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-sync-patterns.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-framework-overview.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_pipe/alloc_event_id.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_pipe/release_event_id.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_pipe/fetch_event_id.md
  role: constraint
created_at: '2026-09-11T02:57:41Z'
updated_at: '2026-09-11T02:57:41Z'
platforms: []
---

# Blaze 局部槽位与跨核通道的分别闭合

L1复用通常需要MTE1→MTE2释放边及MTE2→MTE1就绪边；L0A/B需要Mmad→MTE1释放边及MTE1→Mmad就绪边；L0C还需要Mmad→Fixpipe结果边及Fixpipe→Mmad复用边。V→MTE3是输出数据的正向边，来源把它列为反向依赖不准确。读完哪块存储后允许谁覆盖，应按实际地址和下一写者决定。

Init可为原本空闲槽位预发复用许可，循环消耗/返还，退出时排空实际在途工作。不能无条件析构等待本轮未发出的事件；zero-row AIV可能无需本地搬出wait，却仍需履行对端真正等待的完成通知。预先清零、VF store也是写者，必须纳入与MTE2竞争的slot生命周期。对象析构只能覆盖其管理的局部协议，不能自动替代Kernel跨核交接或Host ACL释放。

事件身份包含合法HardEvent及ID，不能按来源声称“一切HardEvent共享同一物理寄存器”统一分配或由此推导必须偏移6。来源自身初始化在不同类型使用同ID，已与该说法冲突。应按目标实现的资源分配器、核类型和接口约束管理；同一类型中A/B独立槽位确需避免错误别名/假依赖。AllocEventID占用到对应ReleaseEventID，FetchEventID只返回可用ID并不占用，复杂重叠生命周期不能当独占分配使用。

buf=i&(N-1)仅对正的2次幂N实现取模，双槽XOR1要求状态确为0/1。固定ID范围0..7/0..3、AIV偏移16和mode4均须绑定平台API与具体入口。示例count/15%3按15期轮换3通道，不能解释为所有硬件必须该公式；生产者与消费者在相同逻辑期次的递增位置、初始额度、未消费积压和最终drain需一致，物理buffer槽也不一定等于flag槽。

跨核通知的pipe必须覆盖被释放资源的真正最后消费者。固定Blaze helper若以PIPE_V发布归还，不能因Epilogue还有GM写出就机械改成PIPE_MTE3；但也不能借helper名称忽略其实际存储生命周期。__mix__(1,2)只给执行比例，不证明某个准备阶段有两个活跃producer。空核/空组的通知要与对端跳过数据读取或identity输出的语义一致。

PipeBarrier只约束本地对应流水，不代替跨核可见性或SyncAll。CV1之后按整行重分任务给V2，需要生产者最终完成与声明参与者的阶段协议；SyncAll的AIV-only形式不覆盖未被另行闭合的AIC生产者，硬同步还需目标调度/核数限制。加屏障后通过、两个输出不同或静态Set/Wait数量相等都不是唯一根因证明。

DataCopyPad的GM stride字节与UB stride的32B额外间隙分开推导。只有实际UB pitch等于该重载自动对齐跨度时UB stride才为0；显式padding、Compact或更大pitch另算。padding字段上限不等于有效搬运长度上限，长度超过32B不必然拆成多个32B copy；不能用tile剩余长度填rightPadding。已有[多行padding](../data_movement/datacopypad_rows_and_padding.md)给出完整边界。本批未证明示例全部flag/API组合可编译。
