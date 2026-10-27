---
type: Guide
title: MoE 执行单元与同步证据
description: MoE 执行单元与同步证据的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/kernel-execution-units.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/common-reading-method.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/mega-moe-reading-guide.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE 执行单元与同步证据

Scalar承担地址、循环、分支及指令下发；Vector承担向量计算，Cube承担矩阵计算，DMA/MTE承担数据搬运。AIV/AIC是核类型/执行环境，不能完全等同某一条Vector/Cube流水；复杂dispatch/combine也可能只用AIV侧Scalar+MTE+Vector，只有实际task type、编译分支和Matmul调用才支持AIC参与判断。

GM→UB通常是MTE2，UB→GM通常MTE3；仅看到DataCopy名字不能确定pipe，需看地址空间和具体重载，L1/L0间搬运另有路径。Cast/Add/Mul/Reduce等通常向量路径，但VF/寄存器API的工作数据可在寄存器而非所有输入输出都直接UB。源“Vector所有源目标都必须UB”是传统SIMD简化，不宜覆盖所有AscendC执行模型。

MTE2_V事件表示搬入完成才允许Vector消费，V_S表示向量结果可供Scalar判断，S_MTE3表示Scalar侧准备完成再允许相应输出下发；sample SyncFunc需检查是否确实Fetch/Set/Wait同一事件。PipeBarrier通常是pipe内顺序，HardEvent跨pipe；PIPE_ALL也不是跨rank通信完成或任意cache一致性的万能替代。

代码有同步只证明作者表达了依赖意图，冗余屏障可能存在，缺失屏障也可能由队列或API内部实现提供。是否形成MTE2(token n+1)与Vector(n)、MTE3(n-1)重叠，须结合事件位置、buffer持有、编译和时间线验证，不把示意甘特图当设备观测。
