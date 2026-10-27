---
type: Guide
title: 稀疏与稠密分发需要有效集合等价
description: 区分块数量预算和实际 token 覆盖，保留分发后的掩码与资源合同。
tags:
- task.design
- topic.attention
- topic.compute.indexing
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P62_sparse_dense_dual_path_dispatch_dra.md
  role: primary
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# 稀疏与稠密分发需要有效集合等价

Sparse Attention 可以在有效稀疏选择覆盖完整可达区域时走稠密路径，避免无收益的索引和Gather。来源判断 `curMaxS2 <= actualSelectedBlockCount*selectedBlockSize` 只比较长度与容量；它不是覆盖证明。重复块、索引孔洞、非零起点、页边界、最后块有效长度，以及不同query的causal范围，都可能使乘积足够但实际集合仍缺项。

合法分发应证明稀疏路径与稠密路径处理相同的逻辑token集合及相同权重语义；若原语义允许重复索引贡献，去重后的集合相同仍不充分。sink、mask、位置编码、head映射及有效序列长度必须一致。空集合需要定义输出，不应以0<=0机械调用不支持空输入的路径。

Host条件、Kernel分支、输入起点、workspace大小与同步事件需匹配各路径。稠密路径免除稀疏Gather，不意味着可复用尚未发布的数据或取消其他阶段依赖。用全覆盖、有孔洞、重复项、跨页和causal尾部对照两条路径的输出；再评价分发判断及资源增加的总成本。该来源虽标为draft/不建议直接应用，其覆盖判定机制仍可提炼；示意条件不被认证为完备实现。
