---
type: Guide
title: 权重反量化 Prologue 的数值链与交接
description: 将低比特权重的逐通道反量化、物理布局和 AIV 到 AIC 的数据交接作为同一输入合同验证。
tags:
- task.design
- topic.compute.quantization
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/prologue-vf-dequant-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/weight-dequant-prologue-fusion-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/weight-dequant-prologue-fusion-development.md
  role: supporting
created_at: '2026-09-10T14:30:34Z'
updated_at: '2026-09-10T14:30:34Z'
platforms: []
---

# 权重反量化 Prologue 的数值链与交接

当 Matmul 的输入合同要求先把低比特权重转换为浮点权重时，可以由 Vector Prologue 生成 B_dequant，再交给 Cube 计算。它与矩阵乘之后对输出做 scale/dequant 的位置不同；不能只因两条路径都含“反量化”就认定等价。

来源的逐通道公式为 `B_dequant[k,n]=(Cast(B[k,n])+offset[n])*scale[n]`。这是一种已指定的输入语义，不是所有量化格式的定义。offset 的正负、是否存在、广播轴、每步 dtype 与舍入均须按实际合同确定。省略 Add 与加入数值零，在特殊值或带符号零行为上也应服从目标合同。

位宽转换同时影响数值与寄存器 lane 排布。多步 Cast 的中间类型、Load/Store 分布、mask 所控制的元素及重排方式需要匹配；仅最终 dtype 相同不能证明转换链等价。来源中的具体 FP8 拆合、Trait 配置和“已验证”标识未在本批设备复核，不作为通用 API 支持表。

输入权重、转换输出、scale 和可选 offset 可能同时存活，缓冲预算须按各自 dtype、物理 pitch、padding 和缓冲份数计算。每个输出 slot 从 Vector 写完到搬运消费者读完之间不能覆盖；Scale 跨 K 迭代复用时，其 N 坐标和有效范围也必须保持一致。

如果 AIV 将 B_dequant 写入 L1 供 AIC 使用，双方必须认同同一个字节区间、布局、K/N tile 与完成事件；还需防止原 AIC 的 B 加载路径再次覆盖该区域。跨核交接的完成和复用条件必须按目标平台及实际调用链证明，本地 HardEvent 不能仅凭逻辑数据流替代跨核协议。

可分别比较反量化结果、使用已知反量化权重的 Matmul 结果和完整融合结果。前两项通过仍不证明融合时的资源复用与交接正确。Golden 应模拟约定的逐操作舍入并使用同一实际输入；语言或后端名称本身不决定数值正确性。覆盖不同通道 scale、transpose、K/N 尾块、dtype 边界及多轮 slot 复用。
