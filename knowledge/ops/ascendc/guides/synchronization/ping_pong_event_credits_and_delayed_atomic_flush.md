---
type: Guide
title: Ping-Pong 初始许可与延迟 Atomic 刷新
description: 保留手工流水和局部累加设计，补齐缺少的事件与输出协议。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_01_double_buffer.md
  role: primary
created_at: '2026-09-11T06:10:16Z'
updated_at: '2026-09-11T06:10:16Z'
platforms: []
---

# Ping-Pong 初始许可与延迟 Atomic 刷新

手工slot循环要区分可复用许可与数据就绪：MTE3→MTE2表示上一搬出已结束、该别名区域可再搬入，MTE2→Vector表示新数据就绪，Vector→MTE3表示输出可读，MTE3→Vector表示搬出结束可覆盖输出。Dequant预定义ping6/pong7只是样例资源编号，所示代码没有完整CopyOut、初始许可、归还及排空，不能直接运行成完整双缓冲。

MTE3→Scalar/Vector首轮预Set可表达空闲slot初态，不证明有真实搬运或初始化已完成。每条后续Wait须有对应生产，完成后重设许可；同一事件ID还需与其他helper隔离。DynamicBlockQuant的MTE2_V/V_MTE2及V_MTE3/MTE3_V分别管理输入/输出方向，四种方向都出现也不证明跨tile重叠。

DeformableAttention手工TBuf分成两半，以ping*alignedHeadEmbedDims选输出，在写回后发布MTE3_V再切ping。首次Wait需初始化两份许可，Duplicate0必须在对应上次搬出后，最后两份搬出需排空；value双槽还要独立证明输入地址/事件闭环。

SoftmaxGrad示例先CopyIn再等MTE3_V保护Compute重写输出，只在输入与未完成输出不别名时安全。若共享存储则更早的CopyIn可能已覆盖，需MTE3_MTE2等真实依赖。来源Fetch了该方向却未使用，不得凭变量名认为别名已保护。最后一轮没有MTE3_V发布不表示自动完成，函数/调用边界仍需排空保证。

EmbeddingDenseGrad局部累加可把同一目标的多次更新聚成一次GM AtomicAdd。示例每槽计数阈值10、两份本地FP32累积，需追踪各槽目标ID、有效性与计数，在切换目标/达到阈值/最终尾部时写回并重置；CheckIsNeedSwitch、IsLimite若在更新前检查，阈值实际计数取决于helper，不能机械断言恰10。旧槽与当前槽写回地址不能互换。

初始化输出与Atomic累加之间要有所有参与者可见的完成协议；本地AtomicAddInUb函数名并不证明硬件atomic，单owner的UB累加与多核GM atomic不同。连续10次同目标时最多可把相应GM更新次数降约90%，随机目标频繁切换则未必，浮点合并顺序变化也需验证精度/确定性合同。

TransQuantParam的offsetUb中间区与resUb结果区是不同用途，不因此形成相同数据的两份Ping-Pong。SPLIT_SIZE65536、uint64输出每块8192元素仅是该布局；loops=max(scaleLen,offsetLen)/8192只有完整块数，还需余数及两输入广播/长度合同，写回后保护resTensor复用。不能保留来源另一DSL的baseline作为AscendC接口证据。
