---
type: Optimization
title: 反量化输出的 Bank Padding 与 L1 搬运步长
description: 将 bank 候选变换落实为完整物理地址关系，纠正固定吞吐与禁止 Tensor API 的外推。
tags:
- task.optimization
- topic.memory
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/prologue-ub-layout-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/pitfalls-and-diagnosis.md
  role: supporting
created_at: '2026-09-11T03:07:47Z'
updated_at: '2026-09-11T03:07:47Z'
platforms: []
---

# 反量化输出的 Bank Padding 与 L1 搬运步长

来源采用16个bank、32B粒度的特定模型描述跨DataBlock store：若连续写出的块地址以stride推进，stride为16倍数可映到同bank；给该物理步长加1可能分散访问。是否存在这种模型/冲突及真实吞吐，应由目标架构和指令访问集合确认，不能把“8倍吞吐、48%至62%时间下降”视为通用或本批实测。

padding必须改变真正store使用的outerAxisSize及bOut物理容量，bIn可保持其原布局。来源表与后续NZ/ZN文字对外轴命名不一致，不能按名字机械套公式。应先列每个输出元素写到UB的字节地址，再列每个burst从UB到L1的地址，证明有效元素一一对应、padding被跳过、目标分形仍正确。

来源低层copy_ubuf_to_cbuf以nBurst、lenBurst、srcGap、dstGap表达搬运；在其32B块单位模型中，两burst起点分别增加(lenBurst+srcGap)*32与(lenBurst+dstGap)*32。候选NZ为ceil(Nu/16)个burst、lenBurst=Ku、srcGap=1、dstGap=Align(curKL1,16)-Ku；候选ZN为ceil(Ku/16)、lenBurst=Nu、srcGap=1、dstGap=Align(curN,16)-Nu。只有与实际分形store、尾部和容量匹配时才成立，dstGap不可为负，字段范围及sid语义也需核对具体接口。

若当前Te::Copy特化未消费新增stride，单改Layout可能无效；这不证明Tensor API永远无法表达+1。可选支持该stride的特化，或使用已证明的显式搬运参数，并在所有生产/消费者中保持同一物理布局。不能只替换API名称。

优化前后先用跨分形块、不同通道、非零K/N偏移和tail的可区分数据确认完全相同的B_dequant与Matmul输出，再比较目标指令、AIV时间与端到端时间。AIC成为瓶颈时局部store改善未必等比例降低任务耗时。容量和生命周期见[联合预算](../../guides/memory/weight_dequant_ub_l1_and_bias_budget.md)。
