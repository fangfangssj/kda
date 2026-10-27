---
type: Guide
title: Blaze 本地 MX 模板的 SplitK 哨兵与原子输出
description: Blaze 本地 MX 模板的 SplitK 哨兵与原子输出的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/quant_matmul_mx_block_mmad_swat.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/utils/constant.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/tile_mmad_mx.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/qbmm_mx_kernel.h
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze 本地 MX 模板的 SplitK 哨兵与原子输出

本地BlockMmad operator由caller传tensorL0C与splitKidx；源码最后K段且idx为-1/-2才unitFlag FINAL_ACCUMULATION=3，否则2，并仅-1/-2执行Fixpipe。cmatrixInitVal首K块条件为idx0，或非atomic且idx-1，或atomic且idx-2。它与toolkit remoteRankCnt=0..R-1的接口不同，不能把注释“其他数字块数”当完整状态机。

非atomic idx-1会在首K块重置L0C，因此若caller期望前面idx0/正数的partial延续到最后-1，会被清掉；需选择符合预期的终结标记及初始化协议。atomic idx-1不重置，可延续已初始化L0C；atomic idx-2重置，适合独立partial后原子写。具体caller必须明确首/中/末、是否独立partial，以及L0C在Fixpipe完成前的生命周期，本文件本身不包含完整调用证明。

原子输出围绕Fixpipe DisableDmaAtomic→SetAtomicAdd<CType>→Copy→DisableDmaAtomic；GM目标须有正确初值，所有参与者类型/范围支持，最终reader等完成。L0C本地累加和GM atomic两层不同，不能以enableAtomicAdd替代跨rank归约协议。bias没有这个operator参数，不能把toolkit带bias接口直接代入。

Mmad m/k/n截uint16，K按64补齐，curM/N与L0C有效layout容量必须匹配；unitFlag只是硬件/库协议值，不是任何代码都可抄的通用常量。本文保存来源控制表达和可能失效组合，没有对本地备用模板进行NPU实例化。
