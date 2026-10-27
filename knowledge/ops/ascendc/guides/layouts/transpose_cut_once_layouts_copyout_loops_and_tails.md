---
type: Guide
title: Transpose 单轴切分布局、写出循环与尾部
description: 保存CUT_ONCE地址推导及完整描述符条件。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_cut_one_axis.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_cut_one_axis.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_base.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_tiling_data.md
  role: supporting
created_at: '2026-09-11T06:46:43Z'
updated_at: '2026-09-11T06:46:43Z'
platforms: []
---

# Transpose 单轴切分布局、写出循环与尾部

CUT_ONCE/CUT_TWICE属于输入/输出共同约束UB的搜索。参考先给输入sqrt(UB元素数)预算、再按实际输入块余量选输出轴；sqrt仅初值，不保证最优或尾部padding可放。outCutIndex与输入切轴在perm中的输出位置比较决定是否独立切两轴，数值下标大表示更内侧，来源对靠外/靠内文字有反向表述，应以索引关系和真实tile为准。

同一切轴投影可合并in/out因子，但乘积和尾需按原维度重新算；outUbFactor已覆盖全轴时重选轴也要重算所有派生形状/区间。扩大任务数以达到核数阈值是性能启发，不能忽略更小DMA事务代价。实际Host实现未附本模板，不能只填两factor就认为完成。

Kernel将rank≤5右对齐扩展，outCutExpanded=outCut+5-rank。GM输入stride来自完整expandedInputShape，UB目的stride来自tile输出形状，在切轴前一维按32B对齐并向外递推，再经expandedPerm和reverseArray适配NDDMA维序。base reverseArray使用uint32临时量，若传入更宽类型含>2^32值会截断；实际描述符允许宽度/范围应先核验。

q/r分配DMA任务后，线性loopIdx经输出块数混合进制解析、按perm映射输入坐标，分别乘真实GM两端stride，切轴步长乘outUbFactor。UB输入输出主/尾shape必须分别匹配，尾不能只改GM长度不改NDDMA loopSize和UB对齐行距。

CopyOut从输出最内轴合并完整连续维到blockLen字节，次外层为blockCount/GM字节gap，更外层用LoopMode loop1/2，再必要时软件loop4；UB方向的参数单位与LoopMode字节stride不能混用。所有SetLoopModePara需Reset，结构体未使用字段也应按SDK初始化合同，blockCount窄化和最大blockLen须验证。

source寻找首个非整轴时endIndex默认0，若所有轴都完整却仍走该分支，会在已合并全长度后再按外轴叠加blockCount/loop，因此调用方须保证确有对应切分或单独走连续输出。使用主shape判断tail的连续边界要求主/尾结构一致。DecimalToMixed早退出需初值与单调任务条件，不能任意倒序复用。单绑定队列实际num1，改双缓冲需重算容量、队列占用与覆盖时序。
