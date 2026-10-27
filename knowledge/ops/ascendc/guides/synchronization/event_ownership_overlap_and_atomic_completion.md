---
type: Guide
title: 事件所有权、重叠执行与原子完成
description: 保存逐算子同步点，限定Barrier数量与吞吐关系。
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

# 事件所有权、重叠执行与原子完成

显式事件可让无依赖工作穿插在Set与真正Wait之间，但紧邻Set/Wait只保证对应阶段顺序，不自动产生overlap；双buffer及独立任务仍须真实存在。更多PipeBarrier不必提高吞吐，减少barrier也不必更快，关键是完整依赖和实际时间线。

AdamW在weight decay、momentum、variance不同状态更新间、Foreach在Cast→op→Cast间的PIPE_V是来源实现选择。来源baseline用TQue EnQue/DeQue/Free可能已有隐式协议，不能仅因缺显式SetFlag判有race；同流水是否需每条barrier按实际API/编译规则核验。Axpy后UB DataCopy要看真实执行流水，PIPE_V不能统一替代任意跨引擎copy完成。

DeformableAttentionGrad分配MTE2_V、V_MTE3、weight/x/y梯度等事件可区分多个独立存活buffer。source称8个ID只展示部分，不能推成所有算子固定需要8个或没有上限。Alloc/Fetch的资源模型、event类型与ID作用域、预置token、每次消费、排空和释放都须按真实TPipe协议；数值ID0在不同方向是否可共存也不是单凭名称决定。

RMSQuant gamma搬入→MTE2_V→Cast→V_MTE2描述加载与输入buffer归还；后一等待可允许下次MTE2覆盖，但结果FP32 buffer还要等其最后使用者，不能将两者混同。Scatter小模式MTE3_MTE2→CopyIn→MTE2_V→Cast再PIPE_ALL是保守串行依赖候选，并不单凭这串调用证明计算与搬运重叠。

RMSGrad dgamma原子写回可避免单独partial workspace归并，但输出必须先合法初始化且所有更新完成后才供其他消费者；SetAtomicNone只恢复模式，不等于所有核的加法已完成。ROUND_UP(colVal,ALIGN_32)需明确常量单位是元素还是byte，GM实际allocation及padding写范围也须足够。仅本核HardEvent不能替代跨核发布或整个stream完成。

SyncAll用于某些跨核阶段时需真实支持模式、全参与者和内存可见性，不能让无任务核提前退出。正确性、确定性、buffer复用与性能是不同验证项，源15–30%/30–50%/5–20倍的收益没有本批设备证据，不作为已验证结论。
