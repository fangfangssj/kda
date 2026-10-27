---
type: Guide
title: APACE AllGather 预触发与 scale 等待边界
description: APACE AllGather 预触发与 scale 等待边界的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/operator-anatomy.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE AllGather 预触发与 scale 等待边界

来源AG在循环前SetFlag(0)通知本地HEAD可用；第round通信完成后SetFlag(round+1)，AIC HEAD依赖0、MAIN依赖round+1、TAIL依赖commTurn。退出时补等t<=commTurn含端点。预触发flag仍须按协议消费，绕过id0不消费会破坏计数；用uint32位掩码时最大有效索引31，因此commTurn<=31而非直接套A2A的tile数<=32。还要满足实际flagId通道/保留id限制。

只有已经预装且本地producer完成的数据才能称HEAD恒就绪；此预触发不能推广到所有本地输入。没有计算tile的核仍执行该版本协议要求的对应等待。用m<512、恰整除、非整除尾、R=1或最小支持R、最大有效dependId检查初始化和尾部drain。

来源AG设计/开发表称“scale的Wait承担跨卡fence”，但同页代码锚点列scale.Commit→data.Commit→data.Wait<BARRIER_DEVICE>()；实际等的是data对象。scale.Init<BARRIER_DEVICE>负责Init阶段barrier，与后续Wait模板分开。入库保存这个冲突及可复用原则：需按真实调用点识别哪个对象在何阶段执行fence，不凭Init实参猜Wait。

窗口示意也一处写[rank][data][scale]，另一处dataRegionBytes后统一scale段。两种都是可设计的布局，但不是相同偏移。地址owner需按真实base+rankStride+regionOffset核验，source/target和逻辑rank排列不能混用。此材料无已拉取原始AG完整实现，本卡不把任选示意当上游实证。
