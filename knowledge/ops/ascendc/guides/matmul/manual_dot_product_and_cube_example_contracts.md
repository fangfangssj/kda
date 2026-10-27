---
type: Guide
title: 手工 Dot Product 与 Cube 示例的适用条件
description: 保存K分块累加与激活融合，纠正容量、布局和误差承诺。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/matmul/manual_matmul_example.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/matmul/matmul_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/common_pitfalls.md
  role: supporting
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-09-11T05:05:12Z'
platforms: []
---

# 手工 Dot Product 与 Cube 示例的适用条件

Vector候选针对input[B,H]与weight[E,H]计算input@weight^T：每个H分块加载B_t行input与E_t行weight，逐(b,e)做Mul与ReduceSum，再累加到B_t*E_t结果；K/H的partial必须在最终激活前全部求和。初值0、有效hCount、b/e尾、GM完整行距H、Local物理行距及输出行距E必须一致。源码循环实际Htile→batch→expert，不是文字所称batch→expert→hidden。

temp存乘积时至少容纳hidden_tile有效元素及归约工作量，来源只分配B_t*E_t个float可能不足，如1*1输出却需要128个乘积。result/exp须容纳输出块，input/weight加起来还占(B_t+E_t)*Htile，所有区按UB而非“fit L1 cache”核算；该手工Vector例使用UB队列。src与reduce workspace同址及Vector后GetValue依实际重载/同步，不因示例这样写就自动合法。

行块搬运必须保留实际GM/UB stride，Htile非整数据块时用独立padding行距，B/E/H任一尾不能固定满块。输出循环DataCopy block_experts以及Local→Local输出copy也有对齐约束；各核需正确b/e基址，省略的外层任务划分不等于自动处理全矩阵。融合激活不必迫使选择手工Vector，Cube输出后也可用合法Vector阶段。

softplus数学log(1+exp(x))直接FP32链会在大正值overflow，小负值的1+exp(x)还可能舍入成1丢掉小结果。稳定候选max(x,0)+log1p(exp(-abs(x)))及框架beta/threshold规则需按目标合同和真实API实现；近似可在证明误差时使用。来源max绝对误差1e-5、均值1e-6、相对1%未给输入域与测量，不能作实现保证。

另一份完整输出实际是TileLang示例，不是Ascend C成品；可提取其矩阵任务坐标和存储合同。其256x256输出tile、K_L1128只是假设，A_L1/B_L1各131072B(FP32)，C_L0为262144B，须匹配目标真实L1/L0C并加入转换/双缓冲；不能因Python可声明就认为Cube支持这组dtype/容量。

M//256、N//256的grid只覆盖整块，M或N小于256会得到0任务；K//128与输出T.ceildiv不同，ceil本身不等于尾K已经补0。加载二维tile的完整行距不能仅从起始指针推断，A为K、B/C为N的逻辑leading dimension要由实际T.copy/backend保留。不同输出tile独占意味着无需跨核归约，但launch调度、核内搬运计算同步及K0初始化仍不可少。
