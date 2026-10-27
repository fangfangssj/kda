---
type: Guide
title: FFA 正交 Batch 映射与流式状态
description: 保存双KV对角模型的四路GEMM和完整平面偏移。
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
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-handcraft.md
  role: supporting
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-11T05:27:29Z'
platforms: []
---

# FFA 正交 Batch 映射与流式状态

来源双KV对角FFA使用Q[N,M,D]、K1/V1按N分组、K2/V2按M分组；同一(n,m)输出的score按k为scale*(dot(Q[n,m],K1[n,k])+dot(Q[n,m],K2[m,k]))，输出按该同一概率加权V1[n,k]+V2[m,k]。它不同于将两套KV沿k拼接并扩大softmax域。

可用Q0[N,M,D]及Q1[M,N,D]两个视图/转换，QK1任务(n,mTile)、QK2任务(m,nTile)，PV同理。S0[N,M,K]、S1[M,N,K]各自连续k，Vector用各自地址相加；P也需两种布局供两路PV。正交分组令共同连续排列困难，但并非数学证明所有其他布局/融合均不高效。

PV若复用ColumnMajor B组件，物理V^T[D,K]的leading dimension为K，应按对应layoutVB(K,D)表达，块j起点+j*KB而非+j*KB*D。运行问题K轴长度KB与原始整平面strideK分开，D==K或K==KB会掩盖错误。新增BH后V1平面大小N*D*K、V2为M*D*K，S/P/O与mask各有独立广播/平面规则。

终态候选tile32、KB128：A0/A1分别形成S0[i,r,k]/S1[r,i,k]，Vector16行批量加scale/mask并online更新，C1/C2生成O1/O2，Oacc=alpha*old+O1+O2，末块除l。首步直接初始化输出可避免未初始化NaN乘0；nK需ceil且处理真实尾，来源nK=K/128只支持其整除合同。

per-core S0/S1/P0/P1均按tile²*KB，O1/O2及Oacc按tile²*D并计各自双缓冲，工作区随核数而非全K线性增长。O2[r,i,D]不能对i/r误作连续块；S1/P1的转置访问同理。全K物化可以减少逐块握手但增加空间，两路都可能合法，“大K只有流式可行”取决于可用内存和实际shape。

来源aux[B,H,N,M,8]仅槽0有效且其余0是特定输出合同，Host清零和Kernel只写槽0须配套；不同API可能复制8槽。mask[B,1,N,1,K]可在同n多m行共享，不能跨n误用；强制k0不mask只覆盖非空行，不能代替合法全mask测试。scale默认1.0与1/sqrt(D)不同，必须保留接口默认。

四路GEMM可以用5条分别ready或先合并两QK/两PV后3条ready，但只在所有相关生产者均完成后通知；不是四个GEMM天然必须5个flag，也不是原模板所有事件/slot无需改动。闲核是否可退出取决于pair协议还是全核barrier，source各段1:1、2Set:1Wait及“全核参与”不能混用。
