---
type: Guide
title: Blaze-SHMEM Rank 累加、Self 路径与 Bias
description: Blaze-SHMEM Rank 累加、Self 路径与 Bias的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/qbmm_mx_kernel.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/review-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM Rank 累加、Self 路径与 Bias

ProcessSingleBatch对每输出MN tile遍历rank0..R-1。远端A取rank*headM+mPos；scaleA取rank*originalM+mPos，因为scale全量驻留。self A/scale取local GM的rank*originalM+mPos，外层已加入当前mOffset；B/scaleB取rank*Ka/rank*scaleKLen的K段，C只加当前输出mOffset再切mPos,nPos。

remoteRankCnt每输出tile从0开始，传给toolkit Blaze BlockMmad，SetupParams splitKNum=R表达首贡献初始化、后续累加、最后Fixpipe的合同意图。跳self、重复self、提前return、splitKNum与遍历数不符都会破坏覆盖。相同A各rank、常量scale可能掩盖错误，用独立source/target/K编码验证。对全K一次GEMM与分rank顺序累加，数学相等不代表浮点舍入次序逐位相同。

isBias在SetupParams固定false，BlockMmadParams其他字段非全初始化但偏置禁用时不应消费无效地址；加bias需同时传地址、isBias及按nPos切片，并证明跨rank只加一次。QuantMatmulMxKernelSwat::Init只在enabled置true，没有disabled置false，同对象以后切换bias状态会遗留；当前样例一直false不触发。needUpdateTail_也跨Run保留，当前shape不变，泛化不同tiling需重置相应状态。

源码Run每轮Init BlockMmad与scheduler后只ProcessSingleBatch(batch0,true)，不能因ProblemShape带batch字段就认定支持任意batch。gmBias为空时构造/slice句柄不应被实际读，需以所用TE/Block实现确认。这里调用的是toolkit带Blaze::Gemm命名空间版本，并非随附本地Block模板；真实累加语义仍需对应toolkit源码/编译验证。
