---
type: Guide
title: 递推状态版本与梯度 lane 累计
description: 明确反向更新旧状态、跨 batch 的 lane 归约以及最终输出的初始化与所有权。
tags:
- task.design
- topic.memory
- topic.optimizer
- topic.compute.reduction
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P35_tbuf_resident_accumulator.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P37_grad_accumulator_cross_batch.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P35.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P37.yaml
  role: supporting
created_at: '2026-09-11T01:49:44Z'
updated_at: '2026-09-11T01:49:44Z'
platforms: []
---

# 递推状态版本与梯度 lane 累计

来源三状态递推的具体更新为 `y2_new=x*w2+y1_old; y1_new=x*w1+y0_old; y0_new=x*w0`。从 y2 向 y0 更新，是为了在覆写前保留旧 y1/y0。它与先更新 y1、再把 y1_new 用于 y2 的前向依赖链不同，不能因为都有三个 y buffer 就互换循环顺序。新独立序列/实例还需按目标合同重置或加载初始历史状态。

TBuf 可为这些跨迭代状态持有 UB，但不会替代所需的同步。各句柄必须对应正确区域和实际初始化，源示例 y2BufLocal 没有在所示函数内取得。取 Get 不重新分配或清零；常驻区要在整个仍被引用的期间计入预算。TBuf 与 TQue 可以管理不同数据流，不必因为用了常驻状态就把其他队列全部删除。

梯度来源另用 `baseS*calNum*3` 个 FP32 元素保存三组 lane 累加，基础大小为 `12*baseS*calNum` 字节，再计对齐与其他状态。在 B×loopS 循环前清零，所有被最终归约读取的 lane 都要初始化；若目标是在既有梯度上累加，则必须把初始梯度与中间零初值的责任区分清楚。

跨 batch 重复工作可先累计到相同 lane，最后再沿 lane 归约并写出，以减少重复 GM 交通。只有语义上归约的轴才能合并，不能把独立输出通道一起求和；tail lane 也要屏蔽，不能把上一块遗留值再次计入。归约 scratch 和最终输出区需与常驻 accumulator 一起预算。

单核只保存自己实际拥有的贡献，多个核的结果仍需合法且完整的归并或原子更新协议。延迟写回会改变可见时刻及浮点累计次序；FP32 可改善某些宽度问题，但不保证不会溢出、丢失小增量或逐位确定。空 batch/loop 是否产生零梯度或保留输入状态由算子定义决定。

验证至少两个时间步以区分 old/new 版本、不同 batch、不同输出通道、非整 baseS 尾部及大动态范围。测量的对象包括最终归约与发布，不能只计循环体；选择常驻长度时同时考虑 UB 峰值与主 tile。
