---
type: Guide
title: 广播紧凑源同步与复用槽身份
description: 补齐TBuf搬运到Vector与复用前的生产消费关系。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/advanced_tiling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/broadcast_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_common.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_add_tiling.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_add_kernel.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/datacopypad_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/nddma_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/ub_broadcast_design.md
  role: supporting
created_at: '2026-09-11T06:37:24Z'
updated_at: '2026-09-11T06:37:24Z'
platforms: []
---

# 广播紧凑源同步与复用槽身份

紧凑源DataCopyPad写TBuf后立即UbBroadcast读取，必须有覆盖MTE2→Vector的依赖。source LoadBroadcastInput中对bufSrc没有显式事件，qy.EnQue在UbBroadcast之后，不能将其当作之前紧凑源已就绪的证据；同槽下一轮MTE2覆盖前还需Vector读完。是否编译器自动插入要检查实际同步合同/生成代码，源码顺序本身不证明跨流水完成。

多个输入可各有紧凑源以允许重叠，也可共享一槽但显式串行并等待上一消费者完成；“每输入永远必须独占TBuf”过强。共享的收益与同步代价应按真实存活期，扩大并行输入需相应增加预算。退化UB→UB拷贝也要完成源搬入并保留对齐后的源/目标空间。

外层广播复用条件至少包括同输入、同GM起点、所需有效范围、dtype/布局、相同版本数据且目标slot仍保留它；offset相同但尾块尺寸变化或内容已被其他计算覆盖不能直接复用。双缓冲需每槽单独记录身份与valid长度，不能用上一轮全局offset替代。改单槽驻留仍要承担MTE2→消费者与最后读者→覆盖协议。

source每轮重新搬连续数据可以成立，是否值得复用由真实MTE2流量与等待决定；GM输入可能被别的writer更新时跨轮offset相同也不代表值相同。平台DAV_3510/C310是源样例范围，本次未做设备编译或性能验证，未把固定延迟/最快声明迁成保证。
