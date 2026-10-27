---
type: Guide
title: MoE 确定性前缀槽位与原子顺序
description: MoE 确定性前缀槽位与原子顺序的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/dispatch-dataflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/index.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE 确定性前缀槽位与原子顺序

对固定全局路由序index=token*K+topKIndex，每条有效路由发往expert e的dstSlot可定义为此前有效路由中expertId==e的数量。这能让不同发送核独立获得不重叠且确定有序的source/expert槽，不依赖各核局部计数的到达顺序。

来源sample描述先计算curOtherExpertCnt=sum(min(abs(expertIds-e),1))再用index-curOtherExpertCnt得到相等数量；必须完整执行绝对值与0/1截断，单用Sub+ReduceSum不统计相等。整数差/Abs要覆盖值域，INT_MIN绝对值或减法溢出不安全；mask、无效expert与shared路径需在相同有效序定义下计数，不能prefix包含了发送时被跳过的项。

每条路由扫描[0,index)累计工作为O((Bs*K)^2)，向量化只能降低常数。可按expert分块计数、跨核scan或排序生成offset，但要保持实际需要的输出顺序及metadata一致。prefix大于UB时需分段或另算法，index0也不能发非法零长度Reduce。

原子fetch-add可分配唯一slot，但调度到达顺序通常不保证原token/topK序。若后续只靠triple恢复且允许接收顺序变化，可以设计原子路径；若接口要求确定性expert/source/slot排序，就需额外排序或确定性scan。非原子共享计数多写者会丢更新，原子本身也不发布payload；slot分配和ready发布是两个依赖。
