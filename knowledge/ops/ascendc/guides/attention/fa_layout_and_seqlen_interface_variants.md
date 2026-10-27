---
type: Guide
title: FA 布局与 SeqLen 接口变体
description: 区分BNSD、TND、分页格式和两类长度数组。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/flash-attention.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-varlen-handcraft-recipe.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/a2-a3-flash-attention-stage-design.md
  role: supporting
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# FA 布局与 SeqLen 接口变体

来源一种公开接口为连续FP16 BNSD Q[B,H,Sq,D]、K/V[B,H,Sk,D]，内部Q/O为BSND[B,S,H,D]、KV为[numBlocks,128,H,D]。Q和O需相应转置，KV逐batch分块并补齐；原始K文件B*H*Sk*D*2B与设备块分配numBlocks*128*H*D*2B分开。接口选择不排除BF16、GQA、TND或设备布局转换的其他实现。

BNSD可在MHA且映射一致时把B*H重解释成batch、内部head1，Q/O无转置；KV也需具有相同每head连续面及正确分页身份。GQA/MQA的KV头少于Q头，不能简单把同一BH数套给KV造成重复索引或越界。连续页表只改变寻址方式，不自动把非连续布局变连续。

TND Q/O[sum Sq,H,D]可按prefixQ*H*D直接偏移；KV可选择连续TND直读，或每batch单独pad到128后转块。两条路线的Host长度合同不同：来源FAInfer适配用B项原始长度，Host/kernel自行累加；varlen s1s2 recipe用B项累加终点，b0取首项、其余差分。它们不能互换，也不同于长度B+1且首项0的常见cu_seqlens形式。

实际合同需检查数组长度、单调性、非负差分、最后终点与真实token总数、Q和KV各自前缀。TND直读无需为了这个recipe自动转块；paged路线也不能只mask尾部却允许DMA读入下一batch。只有布局/长度/输入内容不变时可复用Host tiling、页表和H2D副本，指针相同不证明内容未改。

BNSD固定stride的batch间距与紧密TND实际长度前缀不同，若BNSD中保留padding却按实际长度累加会错位。布局转换属于完整方案的成本，应在端到端口径中计入；不是所有转换都必须在Host，也不是只做kernel计时就能说明整体更快。
