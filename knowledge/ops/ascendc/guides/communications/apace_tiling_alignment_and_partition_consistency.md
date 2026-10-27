---
type: Guide
title: APACE 通信切分与计算 tiling 的一致性
description: APACE 通信切分与计算 tiling 的一致性的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/fusion.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/communication.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE 通信切分与计算 tiling 的一致性

通信head/tail、每轮计算problem形状、Fragment逻辑跨度/有效跨度、真实GM分配及输出区间必须表示同一partition。固定baseM/baseN并不意味着整套tiling可无条件复用：scheduler核心数、尾块、mmad计数与L1 shape依赖problem时要重算或有明确动态替代参数。源一处要求T>1多套、另一处说单份通用，二者须由实际kernel消费者区分。

源策略B寻找T整除mSeg以免尾块，但T|mSeg并不保证tileM=mSeg/T按16对齐。例如mSeg=48,T=2得到24；合法候选还须检查tileM的真实对齐要求。回退T1也不自动满足mSeg对齐。单tail<=head时PUT连续源偏移合法，计算tail仍要独立保证有效行；padding逻辑行而不扩实际分配会越界。

padding方案可用fragmentSize=alignedRows、realFragmentSize=actualRows并截尾搬运，且A/ScaleA/C消费者全部遵守有效跨度。尾块tiling需区分R*tailM与单rank tailM。源特定Blaze失败信号是非对齐shape垃圾值，不证明所有Blaze Matmul不能处理任意tail。

来源MX约束CeilDiv(K_mm,64)为偶数，其中K_mm是单次计算的K，不能对globalK检查后就宣称每rank分量合法。K scale编码/物理shape还须匹配真正类型合同。M%R=0属于均分ReduceScatter例；usedCoreNum>=R只满足前R通信核方案，后R专职通信+归约通常还需>R。Win容量包含M*N*输出字节及布局所有附加区，T和tileBytes不替代总容量。
