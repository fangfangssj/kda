---
type: Guide
title: Blaze-SHMEM 路线约束与集合通信语义
description: Blaze-SHMEM 路线约束与集合通信语义的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/mc2_architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/comm_shmem.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/review-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/scripts/gen_data.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/scripts/verify_result.py
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM 路线约束与集合通信语义

此样例选AIV下发SHMEM UDMA、AIC用Blaze MX BlockMmad，同kernel按AIV/AIC分支分工，入口core_ratio(1,1)与schedmode(1)。Skill禁止Hccl::/AscendC::Matmul属于路线选择，不宜变成“MC2只能SHMEM”“所有高阶Matmul不能直调”通则；是否直调取决于特定接口、context、资源和launch ABI。另一个固定来源MoE样例显式HcclAllocComResourceByTiling后直传context，也说明不能把缺框架自动注入等同任何显式资源方案都不可能。

样例R个source各持A[R*m,Ka]，B在每卡完整复制逻辑[R*Ka,N]；target t收source s的A[t*m:(t+1)*m,:]，其C_t为这些分片沿K贡献之和。文档“每卡只持B的自己的K段”与实际读全B不符；C_t一般各rank不同，verifier打印“All ranks same”不改变其独立golden。M/N/K切轴与通信方向/前后顺序分开，N切不必自动通信翻倍，M切512也不是通用最优。

PutToAllRanks对不同target取不同源M段，所以原样不等于AllGather的同段广播；AllGather需改源坐标和输出布局。AllToAll后本地Reduce可实现ReduceScatter型分段结果，但不自动成为所有rank持完整结果的AllReduce；完整AllReduce还需正确输入复制/分块及后续收集。归约也须包含self恰一次、初始化、dtype与量化尺度一致。文档的扩展伪码只表达思路，未给完整实现。

不依据来源中URMA英文缩写解释推断协议可靠性；平台启用、UDMA nullptr scratch和deprecated barrier说法绑定其声称的v1.5.0来源，本次未下载底层库复核，均不升级为已验证跨版本事实。流程门禁、角色、评分、三轮修复和固定文件交付数量不入知识。
