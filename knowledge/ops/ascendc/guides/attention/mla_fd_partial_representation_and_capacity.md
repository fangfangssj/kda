---
type: Guide
title: MLA FD Partial 表示与容量
description: 区分归一化Partial和指数和，补齐全核发布及数据类型。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-mla-paged.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-mla-handcraft-recipe.md
  role: supporting
created_at: '2026-09-11T05:34:07Z'
updated_at: '2026-09-11T05:34:07Z'
platforms: []
---

# MLA FD Partial 表示与容量

FD归并需要先确定partial表示。若保存未归一化a_j、m_j、l_j，则m=max(m_j)、w_j=exp(m_j-m)，O=sum(w_j*a_j)/sum(w_j*l_j)；若保存已归一化o_j=a_j/l_j，则分子为sum(w_j*l_j*o_j)。不能给未归一化a再次乘l，或给已归一化o漏乘l。

也可保存o_j与logZ_j=m_j+log(l_j)，按exp(logZ_j-maxLogZ)加权归并。这解释了为什么某实现只需每row/split一个统计float，但source仅列gl一float同时用m/l公式，未说明其真实编码，本批不能推断gl就是普通sum或已含max。必须核对实际MLARescaleO写端和FDRescaleO读端后绑定格式。

maxSplit来自tiling后再分配：oFd至少Dc*H*totalTokens*maxSplit*sizeof(partialType)，l按该实现实际编码及同split布局。来源较新表partial和OUpdate为FP32，旧表globalo写FP16会少配一半；GemmType输出半精度不决定中间Update类型。源“最小S/P尺寸”比所列slot公式更宽裕，不能称已证明最小。

S公式每core32768FP32、P每core65536元素、OTmp每core131072FP32、Update65536FP32只是地址跨度下界，还要检查最大row/col physical tile是否装入每半槽。资源分配和实际传入Element类型一致，不能同一offset有时字节有时元素。

所有partial写完后才进入全AIV归并，局部HardEvent drain不能单独保证其他核完成。CrossCoreBarrier模式、PIPE_MTE3及参与者按真实launch核型确定，闲任务核不可在函数头return躲开barrier。SetAtomicNone、SetMaskNorm和mask复位需符合后续操作与调用者状态约定。

归并阶段按物理AIV编号分发，不是主循环logical AIC组编号。ComputeEleNum6144、Dc512得到headsPerLoop=min(12,16)=12，H16为12+4尾头；若Dc>6144会得0，须守卫而非继续ceil除零。每batch每qtoken与head段只写一次，空KV/全mask/非有限统计要显式处理，maxSplit比实际大时未写槽不得参与。

来源同篇开头声称20case全过，后文又说只验证未split的4点、FD仍待实现；不能以总标题断言当前recipe FD已实测通过。gate3.047/3.116/约3.19和历史33.5倍分解墙钟属于不同批次/口径，不能拼成新的性能证据。
