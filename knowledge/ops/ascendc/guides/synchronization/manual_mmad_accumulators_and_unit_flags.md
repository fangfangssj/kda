---
type: Guide
title: 手动 Mmad 累加器与 UnitFlag 闭环
description: 补齐 L0 输入释放、结果发布及条件屏障示例的失效边界。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P87_manual_mmad_pipeline_timing_control.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P90_conditional_pipebarrier.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P87_manual_mmad_pipeline_timing_control.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P90_conditional_pipebarrier.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P87.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P90.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute/mmad_compute_key_features/unit_flag.md
  role: constraint
created_at: '2026-09-11T02:13:08Z'
updated_at: '2026-09-11T02:13:08Z'
platforms: []
---

# 手动 Mmad 累加器与 UnitFlag 闭环

手动Mmad需要独立管理L1→L0输入发布、Mmad最后读取输入后允许覆盖，以及L0C结果供Fixpipe消费后允许复用。MTE1_M与M_MTE1描述相应输入边，不能由M_MTE1推导L0C已写出。首轮Wait前应有正确可用状态，末轮排空并释放事件；来源仅展示循环内部，没有完整初始化和结果搬出。

TPosition的真实地址空间由API决定，源TBuf<A1/B1/C1>注释成L0A/L0B/L0C不能改变它们的物理位置；必须使用真实L0对应位置与合法类型。每槽32/64KiB、两份等为源例预算，不是所有shape固定需求。高阶Matmul可与手动路径在明确资源和调用边界共存，不必为了grep消除所有库调用。

L0A/B可随k%2轮转，但同一结果的K部分和需连续落入正确累加链。来源同时用l0c[k%2]而未合并两份，会把奇偶K贡献分开；若是两个独立输出，则还需不同任务身份。各结果首贡献初始化、后续累加，最后才发出完整结果；不能只把全局最后k作为所有L0C链的最终发布点。

已有UnitFlag机制以512B L0C内存块同步：Mmad unitFlag=2等待可写后保持写状态，最后贡献设3发布可读；Fixpipe配套设3读取并恢复可写。计算与搬出范围、方向配置以及两端同时开启须匹配。它是细粒度同步，不是把Mmad与Fixpipe合成一条数学指令，也不替代L0A/B的输入生命周期。

来源按 `isL0CAccum && floor(M/16)*floor(N/16)<10` 决定PIPE_M，包含两处危险简化：矩阵执行“足够久”本身不构成跨流水依赖证明；PIPE_M也不能普遍代替M→MTE1门控。条件还会在非累加时跳过屏障，与正文“非累加保留”冲突。必须先以真实接口/调度合同证明已有同步足够，再判断某个屏障是否冗余。

阈值10、尝试8–12/5只是来源特定算子经验，源“A3(910B)”也不应作为平台映射依据。对规则f<threshold而言，降低阈值会让更多f跳过屏障，不能作为小矩阵冒险的通用修复。M/N非16倍数时整除还会低估物理分形数。保留原有必需依赖，再以实际平台行为判断允许移除的范围。

验证K循环0/1/2/3及奇偶尾、多结果、累加/非累加、各UnitFlag区域与Fixpipe读写范围、首轮信号与末轮排空。性能阈值通过不代替同步合同，本批只做固定来源的静态分析。
