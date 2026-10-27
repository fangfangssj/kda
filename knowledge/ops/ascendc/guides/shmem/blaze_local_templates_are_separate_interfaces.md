---
type: Guide
title: Blaze 随附本地模板与 Toolkit 接口的区别
description: Blaze 随附本地模板与 Toolkit 接口的区别的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/block_mmad.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/block_scheduler_policy.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/block_scheduler_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/policy/dispatch_policy.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/kernel_utils/integral_constant.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/kernel_utils/tuple_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/kernel_utils/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/quant_matmul_mx_block_mmad_swat.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/quant_matmul_mx_block_scheduler_swat.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/qbmm_mx_kernel.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze 随附本地模板与 Toolkit 接口的区别

工程入口包含blaze/gemm/block/block_mmad_qbmm_mx.h、blaze/gemm/block/block_scheduler_qbmm.h和blaze/gemm/policy/dispatch_policy.h，采用Blaze::Gemm::Block及MatmulWithScaleMx。随附include/block/quant_matmul_mx_block_mmad_swat.h在全局Block命名空间，以QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE,2>特化；类型、模板参数、Init和operator签名均不同。文件在工程树中不表示主路径正在使用，不能把这份实现的splitK哨兵解释直接套到remoteRankCnt接口。

本地BlockMmad基模板用依赖型always_false让不支持策略在实例化时失败；TypeUnpack/LayoutUnpack将(A数据,scale)二元tuple拆分，非tuple Scale=void，仅数据类型声明存在不足以实例化所有操作。policy中的fullLoadMode/stages是编译期选择，并不自行分配buffer或形成流水；SchedulerSelector只有声明及具体特化，缺匹配路径会编译失败。

kernel_utils Int与_0表示编译期整数，递归Get<First,Second,...>逐层索引tuple并保留forward/decltype(auto)，别与TE Shape的Get及普通MatmulShape成员接口混用。源代码同时出现AscendC::Shape/Te::Shape等版本接口，include前置依赖/namespace/模板主声明需从实际编译链确认。

kernel helper硬编码L0A/B64KiB、L0C256KiB、L1512KiB、BT4096，与Host platform runtime查询不同；跨平台复用应以实际可用资源校核。其CeilDiv(b0)=a、CeilAlign(b0)=a，而Host Align通过CeilDiv*b在b0返回0；同名helper不是一致合同，正数形状、非零除数和加法/乘法溢出需提前控制。
