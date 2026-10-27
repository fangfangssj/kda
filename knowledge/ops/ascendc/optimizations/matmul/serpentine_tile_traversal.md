---
type: Optimization
title: 矩阵分块的蛇形窗口遍历与末窗口映射
description: 通过窗口内M方向并行和窗口间N反向保留访问局部性，明确末窗口高度及线性任务到矩阵坐标的双射。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/swat_design.md
  role: primary
created_at: '2026-09-11T01:12:01Z'
updated_at: '2026-09-11T01:12:01Z'
platforms: []
---

# 矩阵分块的蛇形窗口遍历与末窗口映射

蛇形遍历改变输出tile的访问顺序，使相邻任务更可能复用相同A/B片段。来源SWAT方案把若干M方向tile组成窗口，在窗口内先推进M、再推进N，并对奇数窗口反转N。该顺序有利于保留相邻窗口N边界的访问局部性，但跨核L1不因此变成共享缓存，也不能只从调度顺序证明L2命中或带宽收益。

对非空矩阵，`mCnt=ceil(M/baseM)`、`nCnt=ceil(N/baseN)`，基础任务数为二者乘积；乘法和ceil计算应避免溢出。M/N为零、baseM/baseN非法时须先处理，不能进入后续除法和`totalCnt-1`计算。

## 来源中的末窗口不是简单不足W的余数

设窗口参数W，来源实例为4。定义`w=min(W,mCnt)`、`f=floor(mCnt/w)-1`、`h=mCnt-f*w`。前f个窗口各有w行，最后一个窗口有h行。最后完整窗口与不足窗口被合在一起，所以h可大于W：mCnt=10、W=4时，前窗口4行，末窗口6行。不能把该实现概括成“所有窗口高度≤4”。

对线性任务t，若`t<f*w*nCnt`，令窗口号`q=floor(t/(w*nCnt))`，得到`mTile=q*w+t%w`、`nTile=floor(t/w)%nCnt`。否则令`u=t-f*w*nCnt`、`q=f`，得到`mTile=f*w+u%h`、`nTile=floor(u/h)%nCnt`。最后若q为奇数，将nTile替换成`nCnt-1-nTile`。

这些商和余数分别覆盖对应窗口中的全部M/N组合；反转N只是排列，不增删任务。若改变窗口分组，Host的边缘重排与Device重建必须使用同一套边界，不能只改一处W。W=4是所述实现参数，不是通用硬件常量。

## 逻辑tile、GM坐标和任务身份

未重排尺寸时，GM起点通常为`mTile*baseM`和`nTile*baseN`；尾部尺寸重排后，必须用新尺寸的前缀和计算起点。来源BlockCoord复用部分槽位保存GM起点，另一些槽位保存逻辑tile编号；消费者必须遵守这个私有结构合同，不能把字段名K/B误读成真实K/batch坐标。

遍历改变还会改变同一核驻留数据的复用顺序。与FullLoad或Scale缓存组合时，应重新检查缓存键是否仍相同。验证可枚举全部t，检查逻辑tile集合恰好等于完整二维网格，重点覆盖mCnt小于W、等于W、非W倍数、nCnt=1及多窗口反转；再检查实际GM区间、每核工作量和真实缓存/流水观测。
