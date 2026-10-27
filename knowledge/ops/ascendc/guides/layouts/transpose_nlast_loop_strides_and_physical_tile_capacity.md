---
type: Guide
title: N_LAST 转置的多层DMA循环与物理容量
description: 保存末轴不变策略的源目的地址、尾块及对齐条件。
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

# N_LAST 转置的多层DMA循环与物理容量

该模板在消除size1和相邻连续轴后末轴保持不变时考虑N_LAST，key10004、末轴≥32元素、约4MB与0.9倍核数属于该Host启发，非API必要条件。实际Kernel以普通DataCopyPad、两层LoopMode及外层软件循环组织最多8维；不能据总指南标签说它一定调用NDDMA MultiCopy。

主UB输入形状为切轴外侧1、切轴F、内侧全量；输出tile经perm重排。CopyIn每块逻辑元素V=F*内层积，尾Vt=r*内层积。若每输入切轴共有L=ceil(D/F)块且r>0，块i源元素偏移为(i-floor(i/L))*V+floor(i/L)*Vt；这表达此前各完整切轴段主尾长度，不能简单i*V。无尾时连续i*V。

末轴字节非32对齐时按每行blockCount和真实blockLen搬入，UB行距align32(lastLen*sizeof(T))/sizeof(T)，更外层stride递推。UB峰值需按这些物理行距而非逻辑shape乘积。TQueBind depth1、num2、每份ubSize/2，总预算不可预除两次；blockCount/长度窄化及DMA描述符限额另核验。

CopyOut维序以输入最内轴先，dstStride来自该轴在输出perm位置的后缀积。srcStride0的主尾设置差异本身不是错误：当前CopyOut不消费这一项；判断应追踪实际参数。输出GM起点由ceil(inputDim/tileDim)混合进制坐标乘对应输出tile步长，不能用输入线性偏移代替。LoopMode及软件循环中源和目的步长必须成对保留，每次设置后恢复模式。

文档描述的主尾分离、局部参数数组无spill、零分支是优化目标；源码仍在循环和CopyIn中判尾，局部数组也不保证编译器放寄存器。32元素与32字节、旧972B与当前928B ABI不可混用。
