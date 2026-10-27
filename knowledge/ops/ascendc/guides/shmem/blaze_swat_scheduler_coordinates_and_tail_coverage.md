---
type: Guide
title: Blaze 本地 SWAT 调度坐标与尾块覆盖
description: Blaze 本地 SWAT 调度坐标与尾块覆盖的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/quant_matmul_mx_block_scheduler_swat.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/block_scheduler_policy.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/block_scheduler_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/kernel_utils/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/utils/constant.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/quant_matmul_mx_tiling_swat.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/qbmm_mx_kernel.h
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze 本地 SWAT 调度坐标与尾块覆盖

本地BlockSchedulerQuantMatmulMxSwat以blockIdx/GetTaskRation得到计算身份，blockNum为launch数；这些API比率需匹配核模式。mCnt=ceil(M/baseM)、nCnt=ceil(N/baseN)，4行窗口内交错M，奇数窗口反转N方向，最后窗口行数单独计算。它优化访问顺序，不等于改变输出布局；每逻辑tile只分配一次且正好覆盖有效矩阵。

BlockCoord的M/N槽存GM起点，K/B槽存逻辑m/n tile index；CalSingleCoreShapeByCoord读取K/B，若错读M/N就把字节/元素坐标当tile序号。GetTileIdx先roundIdx++，GetBlockShape用roundIdx==round判断刚发出的是末轮；移动++会改变tail判断。

末轮拆分只在(endBlockIdx+1)*mTailTile*nTailTile<=blockNum时启用；core号分出逻辑原tile和内部split。FP4 E2M1在转置A的M/非转置B的N split粒度取偶数，坐标与shape两处必须同步，E1M2未列同样分支不能假定全FP4覆盖。split偏移越过实际边则返回0尺寸，caller需只跳过该空tile并确认无后续工作被漏。

合并尾块的m/nBaseNormCnt、TailMain、TailLast决定真实大小与位置，坐标减去每已合并块的base-TailMain差。检查最后一块正长度、全部tile矩形不重叠且并集等于M×N；测试M/N非整除、块数小于核数、末窗口、多个拆尾及N蛇形反转。M/base/blockNum不能0，整数helperb0返回a不是合法调度。随附本地scheduler与实际入口toolkit scheduler为独立实现，不能互相借用测试通过结论。
