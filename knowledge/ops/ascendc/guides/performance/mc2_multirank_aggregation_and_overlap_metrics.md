---
type: Guide
title: MC2 多rank统计口径与重叠指标
description: 保存来源两种统计政策，防止平均值和实际最慢轮混用。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/shared/profiling_mc2.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/shared/pipeline_tuning.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow_integration.md
  role: supporting
created_at: '2026-09-11T07:06:47Z'
updated_at: '2026-09-11T07:06:47Z'
platforms: []
---

# MC2 多rank统计口径与重叠指标

来源shared采集10轮每rank最后5次均值再取跨rank max；APACE工作流又给parse_prof跳3轮、丢大于1.2*min样本、rank均值再平均，同时另报rank max。两者是不同政策，不能混为“官方统一整体性能”。每种均需报告原始记录、保留轮次和剔除规则；慢样本可能是真实抖动，不能无说明去掉以提高成绩。

跨rank需要同轮标识及相同kernel/shape/stream。max_r(mean_t T[r,t])不等于mean_t(max_r T[r,t])，如两rank两轮[1,9]/[9,1]前者5后者9；若关心每轮最慢参与者，应先对齐轮次再max，同时结合真实端到端起止。rankNum不能写死4，PROF目录数也不必严格一rank一目录，需用设备/进程映射核验。

aiv_time≈aic_time不证明时间区间重叠，二者可同长却完全串行；需时间线、等待及阶段完成关系。cube ratio和cube utilization的分母可能不同，不能把某项高当绝对利用充分。op_summary不一定含每物理core记录，核间均衡需实际per-core采集，而不能直接在kernel聚合行推导。

max(commBytes/BW,2MNK/FLOPS)是理想下界模型，通信字节必须按发送/接收、排除或包含self、全轮或单tile、scale/metadata等口径统一，算力使用当前实际参与核及dtype。单tile通信时间与全算子FLOPs不能直接取max比较。同步、标量参数重建、负载不均与memory争用都会扩大间隙，不只CrossCoreWait。

AIV慢时增加headM可能降低启动次数，也可能减少重叠；AIC慢时增baseK/scaleKL1也可能压缩buffer或增加尾浪费。bootstrapping TCP端口与实际UDMA数据链路不是一回事，不应从rank慢就断言127.0.0.1端口丢数据。隔离通信/计算仍须闭合协议，端到端与各阶段结果一起解释。
