---
type: Guide
title: SIMT 一致分支、launch上界与寄存器证据
description: 纠正所有if都会线程分化和编译通过即无spill的推断。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/simt/optimization-guide.md
  role: primary
created_at: '2026-09-11T06:42:38Z'
updated_at: '2026-09-11T06:42:38Z'
platforms: []
---

# SIMT 一致分支、launch上界与寄存器证据

只有同执行组内线程选择不同路径才产生分化；所有线程共享mode的if通常是uniform分支，不必分化。将有限mode模板化可消除该判断并让分支内常量传播，Host/VF调用处必须匹配mode，默认else不能静默接受未支持值。按线程数据产生的条件若改成无条件计算还须避免非法地址/除0等副作用。

source GM_ADDR x/y按x[i]直接读写T，实际应使用匹配T的typed GM指针或明确位级载入；字符地址下标与T元素步长不同。grid-stride起点与step必须正且完整覆盖，int32总长/乘积可能溢出，64位不能只为多线程盲目缩为32位。

LAUNCH_BOUND是目标编译/运行合同的线程上界，需与真实VF_CALL线程配置一致，写1024不代表实际发射1024。source搬运2048/1024、计算512/1024，ND32位256/64位128，以及1025–2048每线程16寄存器、513–1024为32、257–512为64、≤256为127是来源选型表，平台/编译器未核实时不作通用硬件表。

32位索引可能降低寄存器压力，按位宽模板化线程数是一种候选；编译通过不能证明无spill，最大可编译线程数也不一定性能最优。观察实际寄存器分配、spill、访存合并、占用和完整耗时，覆盖高维地址、尾及不同mode。source“uint64初始256”与表中512可作为不同起点而非矛盾能力规则。
