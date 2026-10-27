---
type: Guide
title: Blaze-SHMEM MX 分 rank 尺度填充与物理布局
description: Blaze-SHMEM MX 分 rank 尺度填充与物理布局的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/scripts/gen_data.py
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_comm_udma.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/qbmm_mx_kernel.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/common_utils.h
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM MX 分 rank 尺度填充与物理布局

当前A/B都是1字节FP8，scale为e8m0，每64个K元素预留2字节，逻辑group32。每source A[R*m,Ka]需R*m*2*ceil(Ka/64)个scale字节，B需与kernel按rank拼接的R*scaleKLen坐标一致。Host A buffer按m*2*ceil(globalK/64)分配，kernel按R*m*scaleKLen读取，ceil不可一般交换乘R。

例Ka96/R2通过ceil(Ka/64)为偶数检查，但kernel scaleB行长8，Host按globalK192行长6；A scale文件也比Host容量大，ReadFile失败被忽略后继续。Ka65/R2同样通过该检查，gen_data又把6列scale reshape成globalK//32=4列直接不匹配。正确选择是显式约束匹配布局的Ka，或让分rank padded scale与全K scale的编码、文件、容量及layout一致；不能只补大分配而保持错stride。

TransB=true时逻辑B[K,N]由物理[N,K] C-order存，scaleB物理[N,scaleColumns]，不能先transpose再被DN layout解释。A、scaleA按target M段切；每rank scale组边界若不在全K组边界上，不能直接沿用全局scale广播。golden从真实FP8/e8m0编码反量化后算，而不是从编码前随机数算。

外层local A的GM_ADDR偏移mOffset*Ka缺sizeof(AType)，当前FP8恰好成立；改BF16应乘2，FP4还涉及packed逻辑与字节。comm bytesPerMRow=Ka*sizeof(XType)也不是FP4逻辑宽度公式。非MX删scale需同步策略、layout、tiling、通信、buffer和golden，stepK是数据K推进参数，不能因文档“去scale”就全部删掉。
