---
type: Guide
title: N_LAST 合并搬运必须同时保持两端地址顺序
description: 纠正仅按输出轴相邻就增大blockLen的方案，并保留可测量的优化路径。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_n_last.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_n_last.template
  role: supporting
created_at: '2026-09-11T06:51:44Z'
updated_at: '2026-09-11T06:51:44Z'
platforms: []
---

# N_LAST 合并搬运必须同时保持两端地址顺序

把轴按输出位置排序后检查输出位置连续，对任意完整perm都容易成立，不能证明源UB字节连续。将两轴合并为一个burst，必须同时满足两端内轴stride为元素步长、外轴stride等于内轴长度乘步长，且没有尾块、padding或中间未合并轴破坏关系。可交换循环顺序，但必须同步携带每轴源和目的stride；需要重排时应通过正确UB布局、Gather或可表达该映射的DMA完成。

来源input[bs0,bs1,N,H,W,C]→output[N,H,bs0,W,bs1,C]建议直接把W*bs1*C合成blockLen。取shape[2,2,1,1,3,2]、固定bs0=0，目标连续元素对应源索引[0,1,6,7,2,3,8,9,4,5,10,11]，而同一UB原输入布局连续burst给[0,1,2,3,4,5,6,7,8,9,10,11]，因此仅改blockLen会错序。末轴不变不消除这项约束。

合并后需重算四层软件循环、LoopMode、descriptor块数和两端偏移，且验证切轴主尾的共同连续关系。切轴不必绝对独立成组，但部分tile必须不跨原行；perm非全逆序不是可合并的充分条件。源示意表的loop0轴序与代码实际最内输入轴先不同，应以实际地址为准。

来源N_LAST约58.9/124.1与基线19.4/24.6us属于给定案例，合并后约19/25us是预期而非实测；另一个绑定队列实现13.1/15.4us不能作为本补丁已验证。先以坐标编码/不同dtype/非对齐C和双尾验证，再观察真实DMA次数和时延，不将预期收益提升为事实。
