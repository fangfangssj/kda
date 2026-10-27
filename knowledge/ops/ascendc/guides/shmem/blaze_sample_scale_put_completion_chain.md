---
type: Guide
title: Blaze-SHMEM Scale 先行发送与完成链
description: Blaze-SHMEM Scale 先行发送与完成链的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_comm_udma.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/comm_shmem.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/mc2_architecture.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM Scale 先行发送与完成链

PutScaleToAllRanks只下发scale NBI，没有quiet或barrier；随后同block→同remote的PutSegmentToRank下发data并udma_quiet(remote)，再所有参与者BarrierAll，AIV才发MTE3就绪flag，AIC在MTE2等后读取data和scale。scale依赖后续quiet覆盖此前同提交域操作，需确认底层队列/PE完成域；不能说每个NBI后都必单独quiet，也不能删掉唯一完成链。

quiet是对应提交者的操作完成，receiver不通过自己没有提交的quiet就自然知道他人写入完成。远端完成、全卡参与和本卡AIV→AIC pipe依赖需共同闭合；SHMEM barrier不可随意代替指定跨核读写依赖，CrossCore flag也不自然完成UDMA。底层库barrier是否覆盖所有核及内部完成要按实际版本实现。

通信以GetBlockIdx()==target分工，0..R-1应都有对应活跃AIV；计算tiling usedCoreNum过小时可能根本无人向某target发送，而barrier仍可能到达，造成静默缺数据。rank增加不能只改CLI，需launch映射与库参与规则。self slot虽无Put，所有必须参与barrier/flag的无通信核仍不能随意return。

tileCnt=0时scale NBI没有后续data quiet，不能借正常流程论证完成；本样例M<512可触发零通信轮，并且外层axisM被算成0。多轮调用必须排空上次数据/scale消费者再覆写；同stream本地顺序和远端所有rank进度不是同一条件。
