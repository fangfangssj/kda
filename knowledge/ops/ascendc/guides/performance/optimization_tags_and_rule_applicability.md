---
type: Guide
title: 优化标签与规则适用性的边界
description: 保存执行单元、症状和上下文的区分，防止相似度代替硬约束。
tags:
- task.optimization
- topic.performance.analysis
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/standards/tag_taxonony.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/standards/TAG_ADDITION_GUIDE.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/architecture/ARCHITECTURE.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/architecture/API_AND_MIGRATION.md
  role: supporting
created_at: '2026-09-11T04:58:58Z'
updated_at: '2026-09-11T04:58:58Z'
platforms: []
---

# 优化标签与规则适用性的边界

来源规则标签分Domain、Symptom、Context：U为Cube/Vector/Mix/DMA/CPU执行侧，O为MatMul/Conv/Pooling/Norm/Loss/Optim/Compare/Condition/Broadcast/Index/Resize/Mask/Sort/TensorCreation/TensorMove/Foreach/Activation/Reduce/Elementwise/Attention/Reorder/Fused/DataCopy/General算子族，T为FP16/BF16/FP32/FP64/INT8/INT32/INT64/BOOL逻辑dtype。这些检索名字不保证相应硬件支持，例如出现T.FP64不证明该Cube接口可执行FP64。

症状包含低Compute/Cube/Vector利用率、ScalarBound、MemoryBound、TransferDominated、DmaOverhead、MteBusy、CacheMiss、LowHbmUtil、StridePenalty、LocalCopyRedundant、PipeStall、HighScalarRatio、IcacheMiss。它们分别描述现象或假设，不应把MTE活动高等同整体搬运主导，也不应从低计算利用率唯一推出访存受限。

上下文包括小/大K、小MN/batch/tile、末轴归约、ragged序列、NCHW/NHWC/FRACTAL布局、架构、256B对齐以及L0/L1/UB容量和共享L2。阈值和对齐要求须绑定具体API/资源对象；架构标签910B/B2/C/D之间无自动兼容继承。U.Cube/Vector/Mix的互斥只适用于明确定义的单个执行实体，算子不同shape可以多路分发，不宜全算子永远一个U标签。

加权Jaccard可用非负权重交集之和除并集之和排序，required_tags覆盖率反映已观察的前提数量；两者都不是成功概率。空并集/空required集合需约定，未知与已不满足不同。硬件、dtype、容量、同步及精度等必要条件不能被其他高分抵消，conflicting rules也需组合验证。来源0.7、0.55+coverage0.8或0.3路由门限为工具策略，不能给分数0.625的案例自动证明适用。

新症状标签要能指向可测量定义与反例；组合已有标签常能表达小tile访存等交互，无需把每个组合当硬件新性质。HighVecUtil>95%不自动证明过度并行争抢，source MteBusy>60%且compute<80%也是启发式，仍需时间线/消融。新增标签改变评分和候选排序时需检验相关检索结果，别名重命名要保持引用身份；本源taxonomy不是知识仓注册tags，未把这些字符串直接写入知识卡frontmatter。
