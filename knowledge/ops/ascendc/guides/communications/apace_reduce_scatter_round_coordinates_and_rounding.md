---
type: Guide
title: APACE ReduceScatter 轮次坐标与舍入点
description: APACE ReduceScatter 轮次坐标与舍入点的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/compute-first-reduce-scatter/design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/compute-first-reduce-scatter/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/development-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/index.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE ReduceScatter 轮次坐标与舍入点

此场景每rank独立A_j[M,K]、共享B[K,N]，先本地完整Matmul得C_j[M,N]；out_r=sum_j C_j[r*mSeg:(r+1)*mSeg,:]，mSeg=M/R。它是独立完整本地乘积的ReduceScatter，不是全局A/B沿K切分Matmul。源staging为BF16时C_j应先经过相应BF16输出舍入，再转FP32求和后回BF16。

第t轮每target取本段tileMOffset起的curTileM行；A row stride应fullK*sizeof(AType)，scale用其真实rowBytes，staging用N*sizeof(CType)。模板aGM+tileMOffset*axisK只在1字节元素的字节指针下巧合正确，FP16扩展需乘sizeof，转置A又需对应layout的地址映射。每个fragment目标rank的C地址保持其真实输出行坐标。

每轮只写R*curTileM的分段子问题并发送本轮，reducer只消费本轮行范围；全量mm重复T次与忽略turn的全量归约会提前读取未就绪区并破坏并行生命周期。模板虽然算了tileT却未示出传入Params，curTileM也未完整绑定，不能复制成可用kernel。head/tail/full三套或动态一套依赖实际scheduler如何消费；按R*tailM而非tailM生成Fragment逻辑problem。

self chunk直接读staging，remote读Win[source*chunkBytes+winOffset]；共享metadata96B向128B对齐只是某布局。设计里的m4032例3*512+480=2016、7*128+112=1008分别对应R2/R4的mSeg，不能以同一mSeg套两例。归约acc每批初始化为0或首source覆盖；来源伪代码直接Add(acc,...)未展示初值，不能认定未初始化acc可用。
