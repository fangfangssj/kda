---
type: Guide
title: Grouped 融合事件 Credit 与排障证据
description: 保留尾段事件故障机制，纠正强制Barrier即可排除竞争的误判。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/grouped-matmul.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/troubleshooting.md
  role: supporting
created_at: '2026-09-11T05:12:50Z'
updated_at: '2026-09-11T05:12:50Z'
platforms: []
---

# Grouped 融合事件 Credit 与排障证据

对复用UB，V→MTE3保证Vector生成数据后才能DMA写出；MTE2→V保证读入完成；MTE3→V或MTE3→MTE2防止最后DMA消费者尚未结束就覆写。事件方向根据真实生产者/消费者而定，不能只选固定三个方向；来源SetFlag<PIPE_V,PIPE_MTE3>形式是示意，需要目标API真正接受的HardEvent语法。

来源故障模型中主循环遗留一个MTE3_V Set token，尾部导出Wait误消费旧token，导致rowMax被提前覆写。修复需画每个buffer、每个循环分支的Set/Wait计数与初始credit：若已有未消费token，在新导出前先消费，再等本次DMA；恢复下轮credit也必须匹配下一消费者。盲目插入Wait可能在没有预置token时死锁，额外Set也可能错位；0、1、多strip及多cell的初始化、稳态、排空都要成立。

插PIPE_ALL后仍错不能证明无竞争：跨核可见性、错误位置、未覆盖事件或其他数据依赖可能未被它解决。插后通过也不能唯一定位某条依赖，时序扰动可能掩盖别的错误。固定groupList稳定复现与改groupList翻转不能排除race，竞争也可以确定性重现。

分别dump计算前后、GM存储前后及消费者结果有助于缩小范围，但dump改变时序、存储和编译，不能凭一次观测唯一认定计算错或写回错。性能均值中Cube短也不唯一证明调度问题，应结合任务数、地址、事件时序与数据正确性证据。
