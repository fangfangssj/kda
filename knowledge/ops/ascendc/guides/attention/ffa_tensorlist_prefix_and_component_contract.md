---
type: Guide
title: FFA TensorList 前缀与组件合同
description: 区分拼接KV和双KV对角模型，保存FD LSE与量化入口语义。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/ffa-handcraft-recipe.md
  role: primary
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# FFA TensorList 前缀与组件合同

FIA来源接口族包含query、key/value TensorList、共享key/valuePrefix与实际prefix长度、query/keyRope及rope antiquant scale、learnableSink、deq/quant scales与offset、key/value antiquant scales/offset、Q/KV长度和padding、blocktable、attentionOut与可选softmaxLse。称37参不替代真实签名，参数可选性、dtype/shape、pse/mask与scale顺序需逐项对齐。

sharedPrefix与每batch local KV沿序列轴拼接时，逻辑长度是sharedLen+localLen_b，两段进入同一softmax归一化。TensorList各成员可能不同存储池，必须有能表达pool/base与页号的寻址，不能只令blocktable物理号连续就自动访问多个不连续tensor。无需Host全量拼接是可实现方案，但不是任意已有kernel都天然支持。

来源另称“双KV对角FFA”的S=Q*K1^T加特定对角Q*K2贡献，与prefix/local的序列拼接是不同算子合同，不能用同名FFA互相替代。sink-in-epilogue也需区分虚拟logit与额外KV；rope旁路维度与scale顺序同样不能只套一个加号。

组件路线包含FAIQK/FAIPV及Tail、resetBlockStart(kvStart,pagedBlockSize)、ND或K nZ/V zN布局、MAX_KV_STACK_LEN约512、PRE_LAUNCH延迟PV、CrossCore mode0x2和GM轮转。nZ/zN不是更改layout名字即可转换物理数据；INT4 view-only转换也需原始packed字节、stride与元素数量已经符合。

IS_FD让每核处理指定s2分段，仍需正确的partial max/sum/o归并；空KV任务由epilogueInitOut定义输出，LSE按[totalQTokens,qHeads]真实布局。mask压缩视图、fullMask、sink参数须传到所有相关分支，不是只有主循环一个调用点。

registry tilingKey可编码dtype、LSE、TND、paged、KVNZ、mask、split-fuse等组合，direct入口可以用其他经验证dispatch，不必照搬key。循环从kvStart非0开始时，延迟PV条件应基于已生产步数，不能只写绝对kvSIdx>=PRE_LAUNCH就读取未生产槽。末块按当前分段而非全局kvLoop决定，slot保留到消费者结束。

smax/ssum与基线位级一致不是所有实现必需，应按明确数值合同验证；source几何2.19倍只对应其历史双KV对角实现，不能转借给sharedPrefix组件路径。
