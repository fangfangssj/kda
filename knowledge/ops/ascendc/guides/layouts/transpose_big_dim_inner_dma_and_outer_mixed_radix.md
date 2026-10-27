---
type: Guide
title: 高维Transpose的内层DMA与外层混合进制
description: 保留大于5维的坐标拆分和尾块，限定状态化helper的复用。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_big_dim.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_big_dim.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_tiling_data.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_base.template
  role: supporting
created_at: '2026-09-11T06:46:43Z'
updated_at: '2026-09-11T06:46:43Z'
platforms: []
---

# 高维Transpose的内层DMA与外层混合进制

BIG_DIM把超过单次NDDMA维数的输出轴留在外层软件循环，内层描述最多5维子张量；并未让单条DMA突破维数上限。source选择在非小shape、非N_LAST且规约rank>5，但该阈值为实现策略。选输出切轴outCut后，其内层轴整搬、切轴取F份，其他轴每次长度1，外层任务数=ceil(outDim/F)*外侧维积。

nddmaIdx记录内层描述所用输入轴，排序后baseNddmaShape必须同步重排并与输入/输出perm对应。SetLoopInfo将内层整轴、切轴、长度1轴分别设loopSize，GM stride来自真实输入后缀积，UB stride来自输出tile布局，再按DMA维序反转。切轴必须恰好出现一次，idx合法、数组与乘积均在字段范围，不可用丢轴/截断维数兜底。

单次主块元素V=F*内层输出维积，源totalNddmaNum用于V，尾块V/F*tail。outTailFactor>0且(loopIdx+1)%ceil(outDim/F)==0时选尾，前提切轴分块是外层编号的最快变化有效位。GM输出起点和输入起点均由同一外层混合进制坐标导出，切轴stride乘F，输入用perm映射；UB布局已经按输出顺序，CopyOut可连续真实字节。

源码单TQueBind一份ubSize，无双缓冲，复制后归还必须晚于写出。DecimalToMixed遇商0提前返回并不清高位，正确使用依赖数组初值0及单次Process内loopIdx单调递增；若重排任务、重新从0或复用对象需清理全部坐标。ProcessPerCore的stride在成员旧值上逐次乘，重复Process也应重置；不能把这些helper当无状态任意调用。

内层选择5维与UB容量、输出连续度和外层任务数共同影响性能。source Host片段在idxNum减到负时部分分支未示完整guard，不能据节选复写任意rank。以坐标编码验证6/7/8维、不同outCut、内层尾及跨核起始非0；GM/UB字段溢出、single-run初始化和重复调用分别检验。
