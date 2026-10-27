---
type: Guide
title: 索引位宽、越界策略与动态输出形状
description: 补充 Gather/Scatter 索引窄化、空维度、字节偏移和多输出梯度形状的边界。
tags:
- task.design
- topic.compute.indexing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/A7_index_boundary.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/A7_index_boundary.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/A7.yaml
  role: supporting
created_at: '2026-09-11T01:26:08Z'
updated_at: '2026-09-11T01:26:08Z'
platforms: []
---

# 索引位宽、越界策略与动态输出形状

索引有效范围、索引 dtype 和访问的物理地址是三个约束。对 size>0 的零基索引，合法区间为 [0,size-1]；size=0 时没有合法索引，先算无符号 size-1 再 clamp 会下溢。最终地址还要检查 stride 乘法、字节换算、存储偏移与实际分配范围。

INT64 索引窄化为 INT32 可把同数量索引的存储/传输字节从 8n 减到 4n；这不等于总访存或耗时减半。必须在窄化之前证明每个值能表示，保留负索引/哨兵语义；字节偏移和 shape 乘积可能仍需更宽类型。若调用方 ABI 要求 INT64，不能仅把注册改为 INT32，或在 Kernel 截断高位掩盖溢出。

Gather 的越界报错、补零、边缘 clamp、负索引归一化，以及 Scatter 的跳过、报错或其他策略均由具体算子决定。Clamp 会改变逻辑值，Scatter clamp 还可能把多个无效索引变成合法位置的多写者。重复写入的覆盖/累加规则和同步需另定义，不能以“避免崩溃”为由改变语义。参见[稀疏访问与暂存](sparse_indices_and_contiguous_staging.md)。

来源的 `Maxs(idx,idx,0,1)` 把 C++ 标量当作向量 Tensor 参数；应根据实际标量/向量接口改写，而不是复制该签名。向量化边界检查同样要保留有效 lane mask 与真实索引值，不能以输出被 mask 推断非法地址不会被读取。

输出形状应从算子关系推导。源文给出的多输出梯度例子中，grad_value 对应 value 的形状，grad_sampling_loc 对应 sampling_loc，grad_attn_weight 对应 attn_weight；不能把第一个输入 shape 复制给全部输出。这只是所述接口的对应关系，未知维、可选输入或其他算子的形状规则仍需单独确定。

验证输入大小 0/1、-1、0、size-1、size，INT32 两端和越界值、重复 Scatter、非连续 stride，以及静态/动态形状的每个输出。索引有 INT32 类型或存在一条 clamp 指令，不证明上述约束都已成立。
