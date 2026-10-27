---
type: Guide
title: 双缓冲物理槽与常驻参数预算
description: 按实际队列、参数复用和分支核算容量，区分具备槽位与实现重叠。
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

# 双缓冲物理槽与常驻参数预算

Foreach、AdamW、LayerNorm、RMSGrad、ClippedSwiGLU等示例以TQue深度和InitBuffer物理buffer数共同描述队列。每份分配字节乘实际份数，再加全部常驻/临时区；depth=2本身不证明物理2份，普通CopyIn/Compute/CopyOut源码顺序也不自动证明串行，因为DMA与队列协议可能异步执行。需要实际生命周期与时间线。

Foreach needCopyOut=false可不分配对应输出队列，但要有满足inplace/输出合同的实际写回。AdamW每物理slot分配IN_BUFFER_NUM*numPerLoop*sizeof(T)，额外状态和输出不能只算numPerLoop。InplaceAddRMS Arch35示例x1/x2和Y/X四个双缓冲队列需逐项合计，未展示的outRstd和scratch也不能遗漏；RMSGrad文中rstd队列和代码dx队列不一致，应以完整InitBuffer为准。

Modulate可让X/Y双缓冲而Scale/Shift单缓冲，仅当同一个参数切片被多个数据tile复用且最后消费者完成后再改写。parameterStatus缺失参数分支不能访问空指针；四种组合须明确。其baseline本身已DOUBLE_BUFFER=2，差别是参数常驻安排，不能称从单缓冲新增双缓冲。ClippedSwiGLU Alloc/Free后的物理slot顺序不必然固定0/1轮换。

LayerNorm mean/stat输出队列仍有独立字节数和生产消费协议，LocalMemBar<VEC_STORE,VEC_LOAD>只保护指定局部访存，不等价DMA或跨核同步。BatchNorm PIPE_ALL不是全局多核同步；标量读DMA结果用实际MTE2→Scalar，Vector产生结果用Vector→Scalar。

FakeQuant示例每个CopyIn、Compute、CopyOut后PIPE_ALL，会使阶段重叠受限；额外分配第二slot不能保证更快，且循环calcOffset未展示推进。DeepNorm BUFFER_NUM=1片段不足证明Ping-Pong；必须找另一实际槽或协议，不能从标题补造。

双缓冲减少可用tile容量且增加事件/队列成本，短任务或一轮循环可能无收益。来源30–50%、1.5–2倍、8核8倍等均未给本次可复核配对测量，不作为保证；检查零任务、一轮、多轮、尾块、所有slot与最终写回。
