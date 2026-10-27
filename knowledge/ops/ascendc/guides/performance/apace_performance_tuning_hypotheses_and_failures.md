---
type: Guide
title: APACE 调优假设与负收益的适用范围
description: APACE 调优假设与负收益的适用范围的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/troubleshooting/optimization-playbook.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/troubleshooting/failure-navigation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/compute-first-reduce-scatter/development.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE 调优假设与负收益的适用范围

来源报告SWAT在blockNum>=aicNum时不触发AdjustBasicBlock，某些baseM256、tiles/core约1场景可尝试128以增加可调度tile。需M整除、16粒度、尾字段重算（示例mTailTile=1、mBaseTailSplitCnt=1、mTailMain=0）并重新核验stepK/scaleKL1/nBufferNum和L1。减tile增加DMA/fixpipe固定开销，不能仅看cube利用率上升判收益。

dbL0c=2导致fixpipe变慢、增stepK无收益、减baseN反降性能、拆小tile使MTE2饱和、localLast额外重排无收益，是来源若干实验的结果。它们不证明这些方向在所有shape无效；两pipe重叠>=90%也不能推导优化任一侧永无收益，关键看暴露段、关键路径和资源争用。L0C与其他buffer物理归属也不能笼统说都竞争同一空间。

mac_ratio高而cube_util低可能来自标量开销、数据供给或指标分母，不能唯一判MTE2 bound。先观察GM→L1→L0→L0C→GM各阶段、等待区间、真实per-core负载和SCALAR。R*T Params重建、每行*列段*R事件数量有清晰成本模型；批量、地址递推及Fragment视图可减少开销，但减少的成本若已被掩盖不必转为端到端收益。

COMM-only需预填符合原协议的staging，COMPUTE-only跳通信后也需为reducer提供对应输入，禁用flag不能制造不存在的完成条件。分离后资源争用和cache状态会变化，不能直接加两次隔离耗时当融合预计时长。比较同shape、dtype、量化/舍入、rank及缓存条件，保存原样本与负收益；按条件选择不同参数有用，单点阈值不跨平台推广。

文档“TeamBarrier无共享flag竞态”与其communication描述多job共写baseflag冲突，保留参与者/epoch验证而非否定可能竞态；“所有手写远端轮询不可行”也不是普遍定理，合法映射和一致性协议可以不同。SyncAll本地同步不能替远端fence，但冗余同步在已证明等价顺序时可优化，不能把任何次数永久冻结。
