---
type: Guide
title: MC2 变长块分发、偏移与完成协议
description: 保存实现层结构，补齐零短块字段、FIX和远端完成约束。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/mc2/pipeline_balancing_design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/mc2/guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/comm-compute/index.md
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# MC2 变长块分发、偏移与完成协议

可用longBlockCnt/longMSize、shortBlockCnt/shortMSize和shortBlockPos描述至多一个尾块，tileCnt为两类数量之和。短块在前时，前shortCount块起点tid*shortM，后续为shortCount*shortM+(tid-shortCount)*longM；短块在后时同理交换。所有块长度正、总和等M，Host/通信/AIC对每个tid使用相同规则与真实dtype字节stride。

独立long/short/local Matmul tiling可适配不同M，shortCount0时不能读取未初始化shortTiling.usedCoreNum。launchCoreNum应覆盖全部实际计算路径及通信协议需要的参与者，源只取long/short max却未纳入local可能不足；闲核仍须满足同步协议。均匀M/tileCnt整数除法若余数非0不能直接shortCount0，否则丢行。

通信→计算路径必须保证当前块所有数据与scale已到达且可见，再发布AIV→AIC就绪。CrossCoreSetFlag是本地核间事件，不能单独证明UDMA/shmem远端完成；PutScale先调用也不自动保证Scale早于消费完成。计算→通信则需对应AIC输出真实写回完成，再通知AIV；BarrierAll与Quiet的覆盖对象和参与者依通信API，不能把每块全局barrier当所有通信实现必需。

长短块切换可能重新Init资源，必须等最后消费者，尤其FIX输出仍在用的L0C/L1等。来源MTE1_MTE2和M_MTE1事件未包含FIX，不能仅凭这两个事件断言已等待FIX完成；应从实际producer/consumer和库完成接口证明。skipMmadInit=(tid>0)也需配置不变或合法重配，跨长短不同tiling不能无条件跳过。

mLoopIdx直接作CrossCore event ID需验证平台可用范围和复用协议，不是任意tileCnt都可无限增长；flag轮转需确保上次已消费且slot未被重写。短块存在时源在所有相邻块都执行同步，可能过度串行，优化需基于真实依赖不能直接删。

LocalMatmul只有所需输入已经本地可用、输出分区/累加无冲突且不依赖远端聚合时才可独立；AllGather/ReduceScatter/AllReduce/AllToAllv的语义和输出布局不同，不能把“所有MC2首选Local”当通用可用性结论。完整验证包含rank数1/多、M整除/余1、前/后短块、空任务、不同用核和重复调用。
