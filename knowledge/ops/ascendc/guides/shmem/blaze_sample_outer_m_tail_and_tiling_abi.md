---
type: Guide
title: Blaze-SHMEM 外层 M 尾部与 Tiling ABI
description: Blaze-SHMEM 外层 M 尾部与 Tiling ABI的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/all_to_all_matmul_tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/quant_matmul_tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/quant_matmul_tiling_base.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/quant_matmul_mx_tiling_swat.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM 外层 M 尾部与 Tiling ABI

Host固定headMSize512、tailMSize0、tileCnt=m/512向下取整；Device axisM=tileCnt*headMSize且GetCurMSize总返回headMSize。M513仅处理512行，M100或511没有计算轮；更严重的是source t*m分片stride也变成截断后的axisM，M不整除时不只是最后少一行，还会错取target输入段。Host正值/K整除检查没有拒绝该域。

内层Blaze能处理一个head tile的MN尾块，不代表外层通信已支持原M尾。若加tail需传原m、每轮实际行数、足够大的slot stride、相应计算tiling与scale/source/C偏移，producer/consumer轮数一致。也可在此固定无tail方案中明确拒绝m%512!=0，不能以模板名“动态shape”略过。

QuantMatmulTilingData包含13个shape/base/scale/tail uint32字段再usedCoreNum uint32、三个uint8(stepK,nBufferNum,dbL0c)，alignas8/pack8；本次普通g++17按源码header验证sizeof=64，外层两个uint32通信参数加该结构总72B。这是Host ABI探针，不是Device编译或SDK兼容验证。按值传入口须与Host完全一致，新增u64、重排、不同packing均需核对偏移和大小。

SetupParams kL1=stepK*baseK、scaleKL1、nBufferNum，splitKNum=R、isBias=false。usedCoreNum同时决定计算调度和通信目标覆盖；stepK uint8及所有u64→u32压缩要验证范围。存在字段或helper支持不证明外层实际消费；原始shape、计算tile、通信轮次是三个层级。
