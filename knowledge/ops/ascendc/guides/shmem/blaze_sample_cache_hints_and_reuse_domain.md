---
type: Guide
title: Blaze-SHMEM Cache Hint 与复用范围
description: Blaze-SHMEM Cache Hint 与复用范围的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/qbmm_mx_kernel.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/mc2_architecture.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/utils/constant.h
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM Cache Hint 与复用范围

qbmm SetL2Cache只在curBaseM>=当前headM时执行，不能把整算子M与headM混用；否则保留默认cache。B为NZ时请求disable，转置B检查Ka对mask对齐，非转置检查N和baseN。scale仅batch1处理，转置按scaleK行字节与scaleKL1行字节，非转置按N*2/baseN*2判断。

kCacheLineAlignMask由AType决定，FP8为0x7f、FP4为0xff，即128/256门槛，源码注释都写128B不能覆盖FP4分支。B/scale类型不同组合时应证明共享mask合理。GM起点、rank偏移与全布局stride也需对齐，单个维数取模不自动证明所有地址对齐。

gmC参数传入但该函数没有修改C cache hint；不得据名字推断输出streaming。Disable是提示选择，不能保证完全绕开物理cache或必增带宽；若N块复用B或scale较小，禁cache可能反而损失复用。总task、MTE耗时及稳定轮次共同评估，不把cube ratio低唯一归因L1浅。
