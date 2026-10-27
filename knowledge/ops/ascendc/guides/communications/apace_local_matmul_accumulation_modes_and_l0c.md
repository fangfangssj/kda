---
type: Guide
title: APACE localMatmul 累加模式与 L0C 容量
description: APACE localMatmul 累加模式与 L0C 容量的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/fusion.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE localMatmul 累加模式与 L0C 容量

来源AllToAll PUT的mode0在依赖tile就绪后遍历所有rank，self读取本地A，R份在同一L0C累加并一次fixpipe；mode1先LOCAL写C，再REMOTE将R-1份通过AtomicAdd加入C；mode2 DEFERRED_SYNC先算self进L0C，wait后算remote并一次输出。LOCAL splitKNum=1，REMOTE mode1为R-1、mode0/2为R。源ST只固定0，1/2存在分支不等于全量验证。

此版本UDMA有DEFERRED_SYNC选择，hcomm变体非0都LOCAL+REMOTE但透传localMatmul。其共享kernel只对值1跳过self/启AtomicAdd，故把2传hcomm可能重复self且与splitKNum错配；Host须拒绝不相容组合或明确实现等价分支。R=1时REMOTE R-1=0也须审查空阶段。

同tile沿rank累加复用一个FP32 L0C，容量为baseM*baseN*4字节乘实际驻留份数，不乘R；模式1的GM AtomicAdd不增加这块L0C。示例256KiB L0C、2份各128KiB，128x128=64KiB、128x256=128KiB、256x256=256KiB。真实容量/保留区/份数来自目标平台，临界总量相等也需考虑实现额外占用。

mode1必须保证初始写C完成且后续AtomicAdd的类型/布局受支持，AtomicNone恢复状态。来源报告LOCAL→REMOTE补PipeBarrier ALL可规避错误，但把FixPipe称MTE3、把507015唯一归因旧C均不充分。只有实际事件与资源重用证据可确定所需屏障及代价。mode0与mode2均FP32 L0C累加，不能仅凭模式号宣布2必然比0精度更高；mode1额外GM舍入可能改变结果。
