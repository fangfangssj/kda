---
type: Guide
title: 双 Matmul 共享与卷积两级缓存模式
description: 保存共享注册和模板开关，修正 Q 在第二次矩阵乘中复用的错误解释。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_01_double_buffer.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_02_adaptive_tiling.md
  role: supporting
created_at: '2026-09-11T06:10:16Z'
updated_at: '2026-09-11T06:10:16Z'
platforms: []
---

# 双 Matmul 共享与卷积两级缓存模式

Attention同一Pipeline注册BMM1/BMM2、各用自己的tiling，可以配合IBShare配置协调L1输入资源。BMM1=QKᵀ，BMM2=PV；Q不是BMM2操作数，因此“BMM2需重新加载Q、IBShare省掉该加载”的解释错误。Q可跨不同S2块的BMM1复用，P/K/V的槽位与共享则取决于真实配置和生命周期。

GetIBShareNormConfig(true)和REGIST_MATMUL_OBJ只是配置入口，不能仅由共同注册推断自动K/V预取或UB Softmax与Cube已重叠；还需看迭代/完成/槽归还、scratch和两个对象总资源约束。在线Softmax需正确重标定累计输出，不因共享L1自动成立。

Conv2D模板包含FmapTiling、WeightTiling、L1PingPong、L0PingPong、WeightUbTrans、FmapCopyMode、InnerBatch等，支持预先编译不同资源/搬运配置。L1/L0各份数需以真实实现判断，不能仅从L0PingPong名称断言只作用于Weight。Weight经UB转换后到L1增加UB区、转换与事件，并非删除所有格式转换成本。

M-Mode按Ho*Wo的线性M分块，HW-Mode独立计算Ho/Wo与输入起点。模板if constexpr消除该分支的运行时选择，不保证Host分发或地址计算零开销，也不自动按输入选择最优模式。M0对齐物理容量与实际有效M须分别传入；N/M/Ho/Wo/batch的尾标志互不替代。

验证同配置的BMM对象共享与独立资源、短长S2、各卷积空间尾块、转换路径及合法平台组合。固定倍数收益或“编译期确定最优”均需实际目标负载证据，材料不足时保留为设计选择。
