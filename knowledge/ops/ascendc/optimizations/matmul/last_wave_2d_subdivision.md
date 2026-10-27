---
type: Optimization
title: Matmul 最后一波任务的二维子块分配
description: 将末波各父tile拆成二维输出子块，明确全局轮次、父子编号、有效范围、空子块与同步参与者。
tags:
- task.optimization
- topic.implementation.tiling
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/swat_design.md
  role: primary
created_at: '2026-09-11T01:12:01Z'
updated_at: '2026-09-11T01:12:01Z'
platforms: []
---

# Matmul 最后一波任务的二维子块分配

当基础输出tile总数T不能充分占满最后一波P个核时，可以把该波的每个父tile沿M/N再切分，以提高末波并行度。这是输出维度切分，通常不需要像Split-K那样合并部分和，但会增加重复A/B加载、启动和Scale/输出边界处理成本。

对T>0，令全局末波号`R=floor((T-1)/P)`，末波父tile数`t=T-R*P`，其中1≤t≤P。每个父tile分为`q=mSplit*nSplit`个子块的简单一波方案需满足`t*q≤P`；最终活跃核数是tq，最后一个活跃核号是`tq-1`。也可写成旧末核号`t-1`加新增核数`t*(q-1)`，不能重复计算已有t个核。

T=5、P=24、mSplit=nSplit=2时，q=4，20个核处理5个父tile，其余4核没有子任务。T为P的整数倍时t=P，该方案在不增加波次的前提下只能q=1；T=0必须在计算T-1之前处理。

## 父子编号与坐标

在末波，活跃核i的父tile序号为`floor(i/q)`，子块序号为`i%q`；父tile的全局线性编号为`R*P+floor(i/q)`，再经过实际父tile调度映射。来源采用M方向子块索引变化更快：`mi=(i%q)%mSplit`、`ni=floor((i%q)/mSplit)`。

父tile真实尺寸为(m,n)时，可定义`sm=ceil(m/mSplit)`、`sn=ceil(n/nSplit)`，子块偏移为`(mi*sm,ni*sn)`，有效尺寸分别为`max(0,min(sm,m-mi*sm))`和`max(0,min(sn,n-ni*sn))`。这些是语义切分公式，实际Cube、GM搬运与量化布局还可能要求对齐或限制允许的分割数。

先按边缘重分配求父tile的真实GM起点和尺寸，再叠加子块偏移。不能对已经改变长度的父tile继续用基础baseM/baseN推算子块，也不能只改输出地址而不修改A/B与Scale切片。

## 全局末波与每核最后一次迭代不同

某些核比其他核少一轮，因此“当前是本核最后一轮”并不总等于全局末波。实现若用每核round_判断，要证明本核是新增子任务参与者，或者其拆分参数已回退为1。否则可能把前一波的正常任务错误地按末波重编号。新增核的轮数、旧核的子块归属及仍空闲核的终止条件须一起维护。

几何上为空的子块可以不做数据运算，但若该核仍是跨核同步协议的参与者，不能随意return导致其他核永久等待。输出维度切分也必须保持唯一写权，避免非对齐写回、原子模式或边界补齐覆盖相邻子块。

验证以父tile和每个有效输出坐标的恰好一次覆盖为依据，检查T<P、T=P、T=P+1、多波、m/n小于分割数、非整除尾部及非方形拆分。核数利用率提高不是性能结论；应同时比较最慢核、重复加载、Scalar调度与完整Task时间。
