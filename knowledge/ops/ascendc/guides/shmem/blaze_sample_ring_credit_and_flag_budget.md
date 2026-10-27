---
type: Guide
title: Blaze-SHMEM 环形槽、释放信用与 Flag 预算
description: Blaze-SHMEM 环形槽、释放信用与 Flag 预算的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/mc2_architecture.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/comm_shmem.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/utils/constant.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM 环形槽、释放信用与 Flag 预算

ringId=i&(B-1)要求B为正2幂；B3产生0,0,2,2等错误复用，不能只改空间预算。AIV在i>=B时先等本地AIC释放flag(i-B)，再全卡BarrierAll汇合所有消费者释放，之后才能Put新i轮到该slot。若先barrier后等待，某rank仍可能在别rank未消费时推进写入，失去全球释放条件。

AIV每轮Put完成后Set(mode2,MTE3,i)，AIC Wait(mode2,MTE2,i)后计算；AIC只对i<T-B发Set(mode2,FIX,i)，AIV正好对后B轮以外的i-B等待。这给T个ready与max(T-B,0)个release配对，不能在同一层重复Wait。最后B个slot不再复用，无需为了对称额外发布无人消费的release；但kernel结束还要真实drain设备访问。

flagId是轮次i，不是ringId，增tileCnt会增加flag命名空间需求，即使B固定4也不解决编号上限。来源本地常量FLAG_ID_MAX16不等于已证明当前toolkit跨核flag范围，应检查实际API/架构和计数协议；简单i%16会造成未消费epoch碰撞，必须建立周转信用后才可复用编号。

AIC在首轮ready后立即能算，没有代码要求凑满4轮；因此“深buffer让AIC一定等4轮才开始”不是事实。B1限制本样例相邻轮复用，但更广设计仍可在同轮细粒度做重叠。B4、h512、数百KB高效区和理论50%阈值都是来源经验/流程门槛，收益应以同条件时序和端到端耗时判断。
