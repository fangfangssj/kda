---
type: Guide
title: Attention 模式组合保持逻辑 Token 身份
description: 从导航提炼变长、分组、分页、稀疏及可见性的组合条件。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/attention-patterns/AttentionPatternIndex.md
  role: primary
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# Attention 模式组合保持逻辑 Token 身份

(T,H,D)变长布局首先确定每序列边界及逻辑位置，GQA/MQA再按接口映射query head与KV head；QK和PV分别使用Dqk/Dv，MLA还需真实latent/位置分量合同，Dqk≠Dv本身不充分证明是MLA。

多个KV来源如sink_k/sink_v与常规cache需保留来源身份和位置；稀疏indices选择逻辑token，再按block/page table映射物理cache行。可见性在softmax前按逻辑query/key坐标定义，不能用物理page号判causal。sink若是额外logit而非KV来源不能套source dispatch，重复索引与无效page需按合同处理。

这些维度可以组合，source给出的顺序是解释依赖的方便方式，不是唯一合法执行顺序。已证明不可见块可以提前不加载，不能把“mask只改score”解释为禁止跳过内存访问；但任何提前裁剪需保持相同可见集合、数值空行及同步完成。

索引中“七个问题全部否定即标准Attention”不是可靠判据，问题有互斥选项且Hq是否相等不是异常特性；标准MHA也必须明确布局、head数、scale及mask。原文件渐进读取、不得发明模式和模板跳转是Skill流程，不进入知识库指令。
