---
type: Guide
title: MLA Split 规划与任务一一映射
description: 保存贪心切分与双解码，补齐页容量及蛇形一致性。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-mla-paged.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-mla-handcraft-recipe.md
  role: supporting
created_at: '2026-09-11T05:34:07Z'
updated_at: '2026-09-11T05:34:07Z'
platforms: []
---

# MLA Split 规划与任务一一映射

来源split启用候选maxKV≥blockDim*2*128、batch≤0.8*blockDim、maxQ==1；blockDim20时阈值5120，历史测试4096→6144跳档不证明精确边界6144。MAX_SPLIT8与组件可能允许KV_SPLIT_MAX64是两层上限，不能混用；planner还需每段至少512token的启发式。

贪心从各batch1段开始，总分配<blockDim时选择alignedKV/split最大且split<8、alignedKV/(split+1)≥512者递增，无候选即停。实际page段长ceil(ceil(KV/128)/assignedSplit)*128，实际段数ceil(KV/段长)可能少于assignedSplit；globalMax若取分配数可作容量上界，但任务前缀必须用实际段数，merge只读真实写出分区。

连续前缀路径process定位batch，再curN=process-prefixBatch且qHeadSplitIdx0；蛇形路径按headSplit×maxSplit解码。若用realProcess变换batch，却用原process解head/split，可组合出不对应同一task的坐标，尤其波次尾不整齐时需证明完整双射。所有轴应由同一逻辑task身份解码，或给出严格等价映射；不能只以每核process递增证明无重漏。

KV空batch、curN≥该batch实际split均不应生产数据，但同一pair两侧必须作相同跳过，后续全AIV barrier还需全员到达。末段长度KV-curN*perCore在合法范围算，nLoop=ceil(curKV/128)，Q pack行数qLen*actualHeadSplit并round16，尾head/尾KV同时约束。

源sum(KV长度)≤numBlocks*128不足以证明独立分页容量：两个129token序列需要4个独占page，258≤3*128仍会错误接受3页。应核验每个有效table项范围、page内valid和共享规则；若允许共享page，sum长度检查也不是必要条件。mask前缀与实际压缩/广播布局分开。

QHead候选列表128/64/32/16/8/1按ceil(log2(qLen))选择并cap，qLen0不得求log。H128按每token重组的spec不能沿用通用batch条目数/地址表。排序或task预取只在实际数据身份不变时有cache收益，不能把多batch称“免费性能”。
