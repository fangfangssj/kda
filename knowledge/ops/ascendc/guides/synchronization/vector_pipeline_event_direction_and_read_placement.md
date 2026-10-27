---
type: Guide
title: Vector 流水事件方向与读取位置
description: 纠正MTE方向、跨事件配对与Scalar读后才等待的示例。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_05_pipeline_sync.md
  role: primary
created_at: '2026-09-11T05:59:16Z'
updated_at: '2026-09-11T05:59:16Z'
platforms: []
---

# Vector 流水事件方向与读取位置

同步边应按真实producer→consumer确定：MTE2_V表示GM→UB搬入后V可读；V_MTE3表示V结果完成后可搬出；MTE3_V表示搬出已消费UB后V可复用；V_MTE2表示V已读完输入后MTE2可覆盖；MTE3_MTE2可约束搬出后的同UB复用/后继搬运，不能解释成“GM加载完成供V计算”。来源MTE3ToV称加载同步、V_MTE2称向GM搬出均方向混淆。

V_S保护向量结果被Scalar读取，S_V保护Scalar产生/修改状态后Vector消费；一个反向事件不能替代另一个。source概述先Set<V_MTE2>再Wait<MTE2_V>不是完整配对，两个方向各须相应Set/Wait及自己的实际事件身份，DMA也须位于正确发布之间。PIPE_V只在其实际Vector范围建立顺序，不能泛称同步所有流水或自动处理Scalar读取。

Source Gemma float reduceOut=ReduceSumHalfInterval(...)之后才V_S：若helper内部已执行合法同步再读取，后面的等待可能冗余；若helper未同步而已返回读取值，补在读取之后太晚。正确边界在实际GetValue/GetAccVal之前，而非按外层函数名决定。InplaceAddRMS段V_S在sqx.GetValue前，符合其意图，但名字rstdValue不能证明该位置数据已包含sqrt/epsilon/倒数。

DequantBias先DataCopyPad weight_scale至低精度临时，再MTE2_V、Cast FP32与后续V依赖，是类型转换的正确依赖候选；GetWithOffset的offset单位需接口确认，半区布局必须允许扩宽重叠。此串行片段本身没有独立计算插入，不能证明已经隐藏Cast/DMA延迟。

NormCommon例rowIdx在for局部声明，循环外再引用rowIdx==0无法作为完整可编译代码；gamma是否预取与bufferV→MTE2释放、MTE2→V就绪需全部配套。逐行ReduceSum与GetAccVal需要有效范围及真正同步，src/dst/workspace全同址也须高阶API允许。

Gather先将索引乘sizeof(T_X)再Gather只适用其offset单位为byte的重载；idxLocal与gatherOffsetLocal必须确实是同一计算结果/正确视图。插barrier不能修复读错buffer、错误索引单位或丢失位宽。源关于NPU普遍“乱序执行架构”的解释过宽，独立流水并行和具体依赖即可说明同步需求。
