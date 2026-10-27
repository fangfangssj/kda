---
type: Guide
title: Cube/Vector 缓冲生命周期与输出行距
description: 修正 TBuf 预加载同步并保留 K_L1 分块和 Fixpipe 布局。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_cube.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_vector.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_cv.md
  role: supporting
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# Cube/Vector 缓冲生命周期与输出行距

Cube侧A1/B1为L1输入，A2/B2为L0A/L0B，CO1为L0C累加；典型流水为GM→L1、按baseK拆到L0、Mmad累加、Fixpipe输出。一个已入队K_L1块在所有内层消费者完成后才Free，不能每个baseK再次DeQue同一逻辑项。是否每外层恰好一次由实际队列生产数决定，不是所有循环统一限制。

Mmad需首K初始化、中间累加及尾K mask/padding，L1复用也须等待L0搬运消费者。Fixpipe写大矩阵的目的行距对应真实输出leading dimension，而不是所写tile的baseN；行主序紧密[N]时可为N元素，但不同重载单位、NZ/padded输出/transpose不能统一写N。调用传K_total/dstStride或保存在Init均可，要求每次与实际任务一致。

Vector的VECIN/VECOUT/VECCALC都位于UB但生命周期职责不同。TQue适合表达输入/输出阶段，TBuf适合临时区或跨循环只读缓存，二者并非因代码是否位于T.serial而硬性决定。直接TBuf搬入后需建立MTE2→V依赖，来源只加PipeBarrier<MTE2>不够；使用对应HardEvent或目标API支持的全pipe等待。复用前的V→MTE2以及写出前V→MTE3也不能遗漏。

Queue depth0引用式API是特定重载合同，不等于“队列为空”或buffer数量0；InitBuffer的buffer份数与模板depth应分开。TBuf/Get或ReinterpretCast仅创建视图，不增加存储、不建立同步。count简写支持与否也需对应API及dtype，不能把所有Cast都当相同重载。

C/V共享scheduler需要双方对同一task/core/slot得出同一逻辑身份，AIV子块按有效行域各自处理，ready和release分别对应数据完成和最后消费。主循环中分流Init/Process可以减少无用资源，但ASCEND_IS_AIC/AIV与真实编译宏、核型、模板是否消除分支需查构建结果。
