---
type: Guide
title: 021 转置的位宽地址表与H分核尾部
description: 补齐8位双半块、32位拆地址及HSplit的读写边界。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_vconv_021.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_transdata_5hd_021axis.template
  role: supporting
created_at: '2026-09-11T06:55:07Z'
updated_at: '2026-09-11T06:55:07Z'
platforms: []
---

# 021 转置的位宽地址表与H分核尾部

021为[N,H,W]→[N,W,H]，参考key10008支持1/2/4B，区别于key10007二维16位专用。H/W>8、面积门槛、有效/对齐面积>1/2、总量70000B以及UseRConv=(H>=W)是路由启发，不能保证所有命中shape收益为正。N少于5、非RConv且H≥N*16时可尝试HSplit，来源Host说明拒绝8位HSplit；Kernel本身没有对应运行时guard，需分发保持一致。

普通分核按NPerCore/末核NTailCore分batch；HSplit让每核处理全部N的不同H段，GM输入起点n*H*W+hStart*W+wTile，输出起点n*W*H+wTile*H+hStart。核间写的是同一batch各行中不重叠H段，不能沿用纯连续batch输出偏移。每份AvailableUbSize加输入输出各num2的总预算，与主尾tile物理面积匹配。

16位每32B块16元素；32位每块8元素，目的16项地址分8行、每行两项分别+0和+8元素，dstStrideFactor2。RConv repeat=r/16，列块循环使用cAlign，因此32位RConv的cAlign应与8元素块及srcRepStride单位一致；CConv repeat=c/blockElem，行组按16。不能把rAlign/cAlign都机械解释为相同16元素块数：普通C切分GM偏移源码仍用cUbSplitPara.UbAlignFactor*16，需与Host该分支的因子定义和实际c逐一对照，否则32位可能跨错列块。

8位使用32元素列块和16行分组，evenUbCount=(rAlign+1)/2、oddUbCount=rAlign/2。实际是交替的16行组，不是逐行奇偶抽取；srcListOdd从第16行开始。even/odd分别写目的low/high half，再分输入srcHighHalf两轮覆盖32列，组切换PipeBarrier<PIPE_V>保护同一块分半写依赖。repeat计数先窄到uint8，必须有上限保护，不能用rAlign很大而静默截断。

普通CopyIn按H剩余真实行数裁剪，W尾按真实validW；UB dstStride为32B块数且block自身隐含向上对齐，因此不能把(factor-valid)*bytes/32的截断单独判为漏padding，应与自动块对齐相加验证。CopyOut有效H/W与UB物理r分离，GM gap用HLen-validH。任何remaining为负转uint32会膨胀，loop与factor先保证不超过真实轴。

HSplit的CopyIn直接使用hCount_，CopyOut才min(HLen-hStart,hCount_)，而Compute的r也直接等hCount_且地址表按16行组展开。因此Host必须同时证明H读范围不超界及计算所需行距/对齐；若尾hCount取上对齐值，输入会越界，若取任意真实非对齐值，Vector末组和输出物理行距仍需另处理。不能靠CopyOut裁尾来证明CopyIn安全。W对齐分支也要确认末c是合法真实宽度而不是过大的padding宽。

验证各位宽两计算方向、两种分核、H/W双尾、多个UB循环，以及源目的位模式。当前资料只有设计/模板，未证明全部Host因子或SDK地址表均已在NPU验证。
