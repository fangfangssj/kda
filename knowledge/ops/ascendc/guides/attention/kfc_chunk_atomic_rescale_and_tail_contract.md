---
type: Guide
title: KFC Chunk 原子累加与尾部合同
description: 保留Prescale顺序和双缓冲状态，限定尾块性能案例。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md
  role: primary
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-11T05:27:29Z'
platforms: []
---

# KFC Chunk 原子累加与尾部合同

来源D128 FP16 BNSD路径每query块128行、KV chunk1024列，S/P两槽，FP32 max/sum按rows*8状态，P16降精度后PV。每chunk QK完成→softmax产生P和alpha；上一chunk PV完成后才能Oacc*=alpha，再提交本chunk AtomicAdd PV。初次Oacc必须初始化0，最后PV完成后才可除最终l和写输出。

原子PV与非原子prescale同时读改写相同Oacc是竞争，即使所有PV自身使用atomic也不能保护Vector写。来源16列组NaN是其表现，不能当所有架构原子RMW粒度事实。每个chunk WaitIterateAll/End、prescale DMA完成及下次PV之间需完整依赖；若使用o2独立槽后显式合并，也需相同数学与生命周期。

alpha=exp(mOld-mNew)要保留旧max再更新，按对应chunk槽存储，不能把下一chunk的alpha提前用于当前PV；若无sink且初m无效/全mask，需专门首步逻辑避免-inf差。16行softmax配两个8行Cast是该容量策略，不是count<8的所有Vector API都会trap。

S按packed cols、P也按同样物理pitch才能供PV，原矩阵K/V的leading dimension与chunk有效宽度分开。Tail需要所有shape字段每次正确设置，不能先短尾再下个task满块沿用旧尾而只在tail条件时调用SetTail。source把非128倍数GEMM一概判坏与API一般tail能力不符，仅作为其路径未支持范围。

pad到128并补K/V合法地址、softmax屏蔽无效score是一种处理候选，但source最终仍声明S%1024≠0的原子路径竞态未解决；不能因前文“标准解法”便声称任意长度已支持。Sq尾与KV尾是独立维，B/H/D和dtype范围也要与测过路径区分。

source总UB约100KiB与F0列S64+P32+tmp32+state17+O16KiB加aux至少161KiB互相矛盾，需按真实同存活区核算。per-worker约1.2MiB GM也需含S/P双槽和Oacc及padding，不直接作为最小配置。状态若经int32逐字拷tiling，须保证POD size对4可整除或补尾字节、对齐及合法别名。

来源流程在F0第7步后结束，并非包含所称11步完整实现；省略段不能补造成已验证代码。历史geomean0.693、458752自注意力22.7s及S1固定Sk128同Sq约5.7ms不是同一工作量，应分别记录接口和输入轴。
