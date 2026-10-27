---
type: Guide
title: Workspace Ring 的 Credit 与整行统计发布
description: 保存逐 tile 与批量交接，修正 flag 位置信息不足以证明协议。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_cross_core_sync.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_cv.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc.md
  role: supporting
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# Workspace Ring 的 Credit 与整行统计发布

按tile交接可用环形workspace：生产者等待free，写槽后发布ready；消费者等待ready，读完该GM槽后发布free。head/tail按同一逻辑tile推进，depth>0，偏移(head%depth)*slotSize需明确元素/字节单位。每核独立队列还要加core基址，总容量不能只写depth*baseM*baseN*sizeof(T)而漏核数、AIV分区或附加数据。

来源InitFreeSlots把同一个flag连续Set depth次，只有硬件/封装确实有足够credit计数、每参与AIV正确初始化且无旧token时才能代表depth个可用槽。若事件只有有限深度或语义为状态位，重复Set不能当无界计数队列。启动预置、零任务、有限任务、末尾drain、下次launch及计数回绕都需定义。

mode2 CV1:2通常由AIC一次Set给对应两个AIV，各AIV一次Set共同使AIC等待满足。是否逐tile或批量不改变模式的参与者合同；来源“逐AIV不同flag可用于ring但不能bulk”把两件事混淆。不同ID和单flag都必须有完整匹配的参与者/模式，不能按signal放循环内外机械决定合法性；0x8+idx还须检查合法ID和组件占用。

ProducerRelease对Fixpipe完成后ready、ConsumerRelease在MTE2读取完成后free是AIC→AIV只读GM槽的一个模型；若计算还从GM读，或该槽也被其他消费者使用，不能提前释放。默认Wait模板的pipe行为需查重载，来源省略pipe与另一指南强制显式pipe相冲突，不能直接认定完整代码可运行。

需要跨整行N统计时，可先写该row-block全部N块，单次发布后由AIV两遍读取：第一遍求max等，第二遍用最终统计量量化。批量workspace需容纳所有同时在用的row-block，而非总是全M*N或总是每核一行块。下一个row-block覆写前仍需消费者完成的反向依赖；来源bulk代码只有ready，没有复用release，不能直接无限循环使用同一区。

整行统计也能与逐tile流水组合为partial累计加后续阶段，不能因为需要全局统计就断言环形方案一概不可用。是否物化S、重算或保留partial由完整依赖、容量与成本决定。loop中或loop外的源flag位置可作为线索，不能替代数学与资源证明。
