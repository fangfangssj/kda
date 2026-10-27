---
type: Optimization
title: Matmul 软件预取的任务序列、槽位身份与边界
description: 分离预取、消费和回收游标，处理首轮、K尾段及跨输出tile转换，并区分手工K预取与Matmul高阶API的M/N预加载。
tags:
- task.optimization
- topic.data_movement
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/mte2_preload_design.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_mte2_preload.md
  role: supporting
created_at: '2026-09-11T01:12:01Z'
updated_at: '2026-09-11T01:12:01Z'
platforms: []
---

# Matmul 软件预取的任务序列、槽位身份与边界

软件预取把后续计算需要的搬运提早发射，使独立搬运与当前计算重叠。它能否获益取决于实际发射延迟、资源与依赖，不应仅凭MTE2 busy高、存在gap或固定队列深度解释判定。

## 先区分两种接口范围

既有[高阶API预加载案例](../../guides/operator_practice/best_practices/matmul_tuning_cases/matmul_mte2_preload.md)通过`MatmulConfig.doMTE2Preload`启用M/N方向预加载，案例要求MDL/SpecialMDL、K方向全载及相应M/N双缓冲。Skill reference描述的是手工Kernel在K循环中提前装入PING/PONG。二者不能因名称相同就共享全部开关、前提或参数。

## 为每份预取数据保存完整身份

一个工作项可用 `(输出tile, K分片)` 表示，还应包含A/B地址、有效M/N/K、Scale起点和转置/布局。槽中保存的是哪一个工作项，必须由预取游标明确决定；消费游标只消费已经发布的同一工作项，回收游标在最后读者完成后才允许覆写。

对有限任务序列，启动只发实际存在的前`min(预取深度,任务数)`项；第i项的K起点与有效长度来自其任务描述，不能永远用`A[0,0]`作为每个核的首项。稳态补入下一合法项，排空阶段仅消费已发项，不再等待不存在的新生产者。两槽例的先后关系可表达为：预取0和1；消费0，待槽0释放后预取2；消费1，待槽1释放后预取3；最终消费最后已发项。

这里“预取完成”与“已发起预取”不同：数据被实际消费前必须完成，下一份搬运不一定要在第一份MMAD开始前全部结束。空槽初态、就绪发布、最后消费者释放及收尾事件应与实际协议一致。

## K尾段与跨tile不能共用模糊条件

当前输出tile的K下一段存在，条件是`kIndex+1<kSegmentCount`；跨到下一个输出tile则还要证明该tile存在并重置其K起点、M/N有效尺寸及A/B/Scale地址。奇数K段、不同tile尾形状和连续调用都可能改变槽位奇偶。

来源伪代码的首个tile会预取`iter0+1`，但后续tile分支实际使用`iter0*kL1`和当前槽，而文字称它始终预取下一段并在前一tile末尾已装好下一tile首段。这些并非同一算法。后续分支没有next-K判断不能单独判成越界，因为它加载的可能正是当前合法段；应逐分支写出真实任务身份和数据地址，再核对文字与实现。

不应把`tileIdx/blockNum>0 || ...`当通用尾保护，更不能仅增加一个判断就宣称跨tile流水正确。始终满足：每项恰好装入/消费一次、没有未完成即覆盖、没有等待未发事件、每个尾段只读合法范围。

## 性能验证

两槽已分配时，提前发射可能不增加物理容量，但会拉长某些数据的存活区间；与全载、Scale合并或不同调度顺序组合后须重算占用和复用键。K只有一段时，本tile内没有下一K段可预取；跨tile预取仍是另一个需明确实现的问题，不能一概断言所有预取无效。

来源BF16 `[M,K,N]=[1024,4096,2048]` 的报告中，kernel为66.000→63.288μs，MTE2为37.595→32.722μs，Scalar为2.558→5.924μs。它说明发射改造可能同时降低搬运等待、增加Scalar工作；各流水耗时可能重叠，不能把它们直接相加减来解释全部Task变化。队列阻塞、带宽、缓存和依赖应结合固定构建的流水图验证，3%–10%不是通用保证。
