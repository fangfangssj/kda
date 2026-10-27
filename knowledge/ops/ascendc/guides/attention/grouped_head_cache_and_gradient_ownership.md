---
type: Guide
title: 线性 Attention 的分组 head 缓存与梯度归属
description: 按真实数据依赖限定 HK 缓存，保持跨 stage 的 workspace 映射，并区分共享输入与需累加输出。
tags:
- task.design
- topic.memory
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/a2-a3-linear-attention-stage-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/linear-attention.md
  role: supporting
created_at: '2026-09-10T14:50:50Z'
updated_at: '2026-09-10T14:50:50Z'
platforms: []
---

# 线性 Attention 的分组 head 缓存与梯度归属

在多个 value head 共享一个 key head 的分组合同下，若 `HV % HK == 0` 且各组连续均分，映射为 `hk = floor(hv / (HV/HK))`。这描述的是一种明确的分组约定；不把它推广到任意 Attention 接口或非均匀映射。

## 缓存键包含全部数学依赖

只依赖 `(batch, chunk, hk)` 的 K-side 中间量，可以在对应 value head 组内共享。若中间量还包含按 hv 变化的 gate、beta、scale 或状态，就不能只按 hk 缓存。缓存复用范围应由计算依赖决定，而不是由输入张量名字决定。

来源的 Catlass 多 stage 设计使用稳定 workspace slot 连接生产者与消费者。同一逻辑数据在所有 stage 中必须使用一致的 slot 公式；GM 跨 stage slot 与 UB ping/pong 是不同的生命周期。窗口轮转前需要确认旧消费者结束，新数据发布也需要满足生产完成依赖。固定四槽公式或两个 head 的调度只是特定实现选择，不足以自行证明复用安全。

## 共享读取与梯度累加不同

多个 hv 读取同一份 K-side 缓存不需要对读取做归约；若多个 hv 产生同一 hk 输出的梯度贡献，则需要累加归属。把整组贡献交给唯一执行者，可以在组内按确定顺序初始化、累加和写回；若把同组拆到多个核，必须另行设计原子或分阶段归约，不能继续使用无保护的读改写。

按 batch/chunk 分核并在核内遍历 head 组，可以保持归属，但 batch/chunk 任务过少时会降低并行利用率。增加 head 维并行度时，需同时修改输出归属及 workspace 协议。

## 分块边界

物理 chunk/tile 尺寸与实际 validRows 分开管理，varlen 任务从真实序列和 chunk 索引得到基址。不能仅依据总 T 推断每个序列尾块。L1 resident、scratch 和跨 stage workspace 分别核算存活范围；大 V 不天然等价于 GEMM 的 K 分割，是否累加以及何时最终写回必须按每个矩阵乘的数学轴判断。

验证应覆盖 HK=HV、HV>HK、尾 chunk、不同 gate/状态，以及同组贡献的多核分配。来源提到的固定 tile、分钟级长跑与性能收益缺少本批原始执行证据，不在这里发布为平台规律。
