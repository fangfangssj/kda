---
type: Guide
title: 因果窗口、GQA 与整块跳过的证明
description: 补全闭区间边界、query 坐标和空行语义，限定 Attention 性能案例。
tags:
- task.optimization
- topic.compute.reduction
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/algorithm_insights/attention_family.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/causal_block_skip.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/proven_solutions/INDEX.md
  role: supporting
created_at: '2026-09-11T03:58:39Z'
updated_at: '2026-09-11T03:58:39Z'
platforms: []
---

# 因果窗口、GQA 与整块跳过的证明

MHA令Q/KV头数相等，MQA令KV头数为1，GQA在整除且组内共享KV的约定下令G=Hq/Hkv、kv_head=floor(q_head/G)。直接索引可避免repeat_interleave实体扩展，但头分组、实际布局与序列坐标必须对应。BSND、BNSD和BSH的轴/stride不同；query的绝对token位置包括本序列的块起点、偏移或缓存位置，不能用head偏移替代，也不能把块内行号单独当所有块的全局token。

对标准j≤i的因果关系，Q闭区间[q0,q1]、K闭区间[k0,k1]：k0>q1时整块无效；k1≤q0时整块有效；其余逐元素mask。对角线有效，所以跳过条件不能改成≥。尾块q1需min(实际末行,块末行)，有效KV块数还要限制在真实KV范围内。来源proven_solutions称无效块在“below diagonal”错误，应为该坐标约定的上三角。

滑窗i-W<j≤i还排除过旧key：当k1≤q0-W时可证明整个块过旧。来源用严格<会少跳过等号边界但通常不漏合法值；区间若改成半开应重新推导。preTokens/nextTokens等接口参数的包含性、非等长Q/K位置对齐、稀疏模式与padding要按具体合同，不能仅套标准三角。

若n个等大Q/K块且完整等长序列，有效块数n(n+1)/2，占全部n²的(n+1)/(2n)，可跳过比例(n-1)/(2n)趋近1/2。n是块数而非无条件token数。不同tile、负载分配、尾块与流水固定开销会改变真实时间收益。来源S=512/2048/8192/32768报告约22/37/43/45.5%降低及1.003x→1.83x，是其910B3案例，未在本批复现，也不能把降低45.5%写成仅加速45.5%。

只在被跳块对最终输出无贡献且无其他必须副作用时省掉加载、QK、softmax更新与PV；Cube/Vector的任务数、信号和workspace槽必须一起调整，避免一侧仍等待不存在的结果。exp(-inf)=0不意味着全为-inf的softmax必然输出0，整行无有效key需目标合同，见[在线状态与归并](online_softmax_state_and_partition_merge.md)。

在线Attention同时重缩放max、指数和与加权输出；存已归一化O时应乘旧l*exp(m_old-m_new)/l_new，未归一化a则采用对应更新，不能混用。状态能常驻时无需每个KV块写GM，是否溢出到workspace由容量与调度决定。QK和PV都含矩阵乘，来源把softmax+PV全部指定给Vector不是通用Cube/Vector映射。

Decode常为每请求单query但仍有batch/head并行与Split-KV选项；不能由Q长1断言完全无需Q侧调度或总是带宽瓶颈。Prefill处理一段query与完整有效KV，KV-cache和分页管理减少重算/碎片的作用不同。来源FlashAttention-2/3及PagedAttention简介仅为外部候选方向，GPU的warp专用化、FP8支持和A100 2x成绩不迁为Ascend已验证能力。
