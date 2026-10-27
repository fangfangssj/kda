---
type: Guide
title: MoE 样例 ready 发布、UB 依赖和复用缺口
description: MoE 样例 ready 发布、UB 依赖和复用缺口的代码事实、前提与失效边界。
tags:
- task.implementation
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/mte_dispatch_comm.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/moe_dispatch.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/include/utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/README.md
  role: supporting
created_at: '2026-09-11T07:32:48Z'
updated_at: '2026-09-11T07:32:48Z'
platforms: []
---

# MoE 样例 ready 发布、UB 依赖和复用缺口

WritePackedToken将padding清零、搬入H个half并写[srcRank,tokenIdx,topKIdx]，MTE3输出等待完成后才统计发送。SetRemoteStatus对每目标/expert发送32B槽，slot0=ready、slot1=count，count与ready通过同一DataCopyPad发布；UB中先SetValue(count)再SetValue(flag)不证明远端32B写具原子可见性或flag必晚于count。需要匹配传输保证，或分开发布元数据与有序ready。

WaitRemoteStatus逐槽判断flag==1，不是sum(flags)==N；未齐备会重读整矩阵，齐备时按source/expert收集count，随后立即ClearLocalStatus，再CopyWindowToOutputs。清ready早于消费不能本身作为producer复用credit；本样例单轮运行也未建立可反复调用的epoch协议。零count槽仍须发布；Process空token范围直接return会跳过状态，泛化为各rank不同batch时可使其他rank永久等待。轮询无超时，count未验证非负/容量且签名间int32/u32转换。

WritePackedToken的Duplicate由Vector写token UB后紧接MTE2写同buffer，代码未显式给出V_MTE2依赖；Alloc/Free但未EnQue/DeQue不能仅凭TQue类型证明同步。MTE2_S之后写triple、S_MTE3也不自动证明此前Duplicate已结束。InitBuffer resetTensor的Vector生成、ClearLocalStatus的DMA消费亦要结合实际中间PIPE_ALL确认。SyncFunc Fetch/Set/Wait表达指定pipe依赖，不能替代远端完成/跨rank barrier。

每条token发送和回搬都等MTE3_S后释放队列，清楚保证当前buffer复用意图，但没有双缓冲稳态重叠证据。各远端data写完再status发布是必要顺序；多核扩展还须把所有核payload完成合并到唯一发布者，不能沿用block0局部发送完成条件。
