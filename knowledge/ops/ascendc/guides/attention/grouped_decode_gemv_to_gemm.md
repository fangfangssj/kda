---
type: Guide
title: 共享 KV head 的 Decode GEMV 合并
description: 以同组 query heads 为矩阵 M 维，保留分组、数值稳定性、任务并行和分块累加。
tags:
- task.optimization
- topic.compute.composite
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/discovered/disc_X5.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/operators/aclnn/transformer/attention/incre_flash_attention.md
  role: constraint
created_at: '2026-09-11T02:29:01Z'
updated_at: '2026-09-11T02:29:01Z'
platforms: []
---

# 共享 KV head 的 Decode GEMV 合并

Decode/MTP即使query序列S只有1或2，也可将共享同一KV head的多个query head排为矩阵行。对Nkv=1，Q为[Nq,Dqk]，QKᵀ为[Nq,Skv]，P·V为[Nq,Dv]，将原来Nq个GEMV合成GEMM。Nkv>1且Nq能被整除时，可逐KV组使用M=G=Nq/Nkv；原资料笼统排除所有Nkv>1过严，但不能把读取不同K/V的所有组直接合成共享同一右矩阵的GEMM。MHA的G=1没有相同的组内行扩展收益。

每个(b,s,kvGroup)任务必须取对应Q行、K/V缓存和输出head位置。Nkv=1时只有B*S个任务，B*S很小仍可能闲核；可进一步按合法维度分块并处理合并成本，不能声称这种网格天然占满设备。不同head可有不同mask/PSE，只有确实相同的部分才共享预处理。S=2的causal位置也不能沿用S=1的可见列。

AIV可负责布局、mask和softmax，AIC负责QKᵀ及PV；交接的GM/片上地址、dtype、发布/消费/回收必须闭合。源码MARSHAL/CS/VP一次性flag是特定串行阶段协议，每任务需重置/消耗正确，不能直接假定事件0–7无冲突；1:2时两AIV行域、到达顺序及空任务必须一致。串行阶段会付同步成本，不自动实现CV重叠。

分数“通常O(1)”不足以跳过max-shift。需按输入域证明或保留稳定softmax与所有KV tile的在线归一化。若PV按Dv分块，每个N tile都可能重新读取P；QK的NC分块也可能重载Q。统计真实重复流量，不能同时宣称Q/K/P/V都只读一次。Fixpipe直接写最终y只在该输出的全K累计和softmax缩放完成后成立，不能让后续tile覆盖部分和。

示例NC=256、KC=256、NV=128，FP32 L0C保存[Nq,128]时Nq=64占32KiB、Nq=128占64KiB；还需计算A/B装载、双缓冲、L1和UB。源L0C128KiB不是所有平台规格。实际NZ布局、尾部、Mmad初始化/续累加和Fixpipe行距按接口合同建立。

已有[IncreFlashAttention](../../operators/aclnn/transformer/attention/incre_flash_attention.md)也采用M=G的矩阵视图，但其公开V4的S1=1、dtype、page及route guards不等于本来源S=2专用实现的已验证支持。分别验证Nkv1/多组、不同heads、Skv/Dv尾块、多tile状态与全masked行。来源decode11–33倍、geomean0.5745→3.527未在本批设备复现。
