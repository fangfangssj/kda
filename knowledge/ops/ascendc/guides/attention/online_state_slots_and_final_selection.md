---
type: Guide
title: Online Softmax 状态槽的接续与最终选择
description: 明确 max、sum、缩放因子和输出累加器的任务身份，避免双槽递推丢失或重复计数。
tags:
- task.design
- topic.attention
- topic.memory
- topic.compute.reduction
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P38_softmax_state_resident.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P38.yaml
  role: supporting
created_at: '2026-09-11T01:49:44Z'
updated_at: '2026-09-11T01:49:44Z'
platforms: []
---

# Online Softmax 状态槽的接续与最终选择

online softmax 的状态沿同一 query/head 的 KV 分块接续。分配两个 max/sum/exp 物理槽，只能使不同版本有独立存储；第二块仍必须读取第一块更新后的状态，不能让两个槽独立从初始值轮流计算。来源模板取 `s2%PRELOAD_NUM`，却没有展示 inSumTensor/inMaxTensor 如何指向真正的前一状态，因此不足以证明跨块递推正确。

对于串行递推版本，第 k 块的新状态已经包含此前全部有效块，最终使用最后一个有效版本；把最后两个槽直接相加会重复累计。只有明确采用互不重叠 KV 子集的独立 partial，才可使用[带 max 重缩放的归并](online_softmax_state_and_partition_merge.md)，并一并处理未归一化输出累加器。softmaxExp 若保存前后 max 的缩放因子，还要保留到对应输出 rescale 的消费者结束。

max/sum 的数学状态通常按 query 行保存，exp 及具体 API 临时区可能采用复制、广播或其他布局，三者并不必然等大。状态容量不能仅由 KV 循环次数 s2Loops 推出；较长 KV 增加迭代数，不必线性增加每行状态。Mmad 的输入输出存储层次也不能由这份 UB 状态模板直接推定合法。

如果确实三份状态每槽各 S 字节、深度 p，则基础占用3*S*p；一般应逐缓冲求和，另计输出累加器、输入、scratch 和对齐。`slot*(S/sizeof(T))` 转元素偏移要求 S 可整除且每个槽满足地址对齐，使用的 T 必须是实际视图类型。L0A/L0B/L0C 是不同存储资源，不能一并当作 UB 字节扣除。

初始化需满足实际 Softmax 接口的首轮模式或显式旧状态。数学空状态的 m=-Inf、l=0 仍需处理全空/全mask的 -Inf-(-Inf)；-FLT_MAX/-1e30 不是与 -Inf 对所有输入等价的替代。exp 是否需初值以及初值为0/1，由其真正含义和首次读写顺序决定，不能统一设0。

在初始化阶段建立长期缓冲有利于维持分配和状态生命期；不能把循环内反复 InitBuffer 简化为固定的函数调用开销，实际还需遵守资源管理接口是否允许重初始化及如何释放。单槽状态可以与其他独立流的搬运并行；双槽也不保证更快。p>2 或运行时选择是否可用，应由资源/API合同决定，而非来源的建议数字。

状态 spill 到 GM 时，须保持 query/head、状态分量和版本的完整字节布局，在 reload 前完成发布，在下一次覆写前等消费者结束。不能只复制 max 而丢失 sum/输出状态。最终输出与所有延迟消费者完成后才可释放槽。

验证0/1/2/3个KV块、多次轮转、最后一块被mask、切换query/head、不同value维度以及全mask行。用不分块计算对照，并检查每个中间版本覆盖的 token 集合。
