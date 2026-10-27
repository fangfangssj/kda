---
type: Guide
title: Blaze GLU 双分支的布局与元数据配对
description: 保持 act/gate 的列配对、padding 和量化元数据一致，并在完整 K 累加后执行非线性融合。
tags:
- task.design
- topic.data_movement.layout_transform
- topic.compute.quantization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/kernel-design/glu-development.md
  role: primary
created_at: '2026-09-10T14:12:04Z'
updated_at: '2026-09-10T14:12:04Z'
platforms: []
---

# Blaze GLU 双分支的布局与元数据配对

对 `act=A×B_act`、`gate=A×B_gate`、`Y=activation(act)×gate` 的 GLU 融合，两个分支在输出列 q 上有一一对应关系。只有输入合同明确把两个 Q 列分支拼成 N=2Q 时，才能用 Q=N/2；独立权重、连续拼接与交错存储需要不同地址映射。

权重、bias 和沿输出 N/Q 轴变化的量化 scale 应一起按 act/gate 配对。沿 M 或 K 变化的元数据仍按各自坐标索引，不能因为 N 轴拆成两支就一律复制或偏移全部 scale。

## 逻辑列与物理存储

若内部存储为 `[act有效列, act填充, gate有效列, gate填充]`，gate 起点由 act 的物理跨度决定，不能使用逻辑 Q。内部 packed 宽度、外部 GM 行距、Scheduler 当前有效宽度和 Copy 特化所需基形需要分别计算。

小逻辑 Q 不代表小物理资源开销：两支对齐后的 L0C/UB 暂存、元数据、slot 和搬运粒度都计入预算。MX 路径还需对齐数据与 ScaleB 的分支坐标，以及逻辑 K 与物理 K 的 padding；填充数据和 scale 的编码要以实际格式为准。

## 计算与搬运边界

非线性 activation 必须消费已经完成全部 K 累加的 act。对各 K 分段先激活再求和通常不等价，因此 SplitK/StreamK 的部分结果不能直接送入最终 GLU Epilogue。

L0C 到 UB 的布局与传输粒度应根据实际累加结果类型和 Copy 特化确认，不能只按 A/B 的输入 dtype 推算；即使两个类型字节数相同，也不足以证明同一特化支持它们。经 GM 绕转得到正确结果，可以帮助隔离 Cube 计算与 L0C→UB 链路，但不能证明直接链路已经正确。

验证应让 act/gate 权重、bias 和列 scale 具有可区分值，覆盖小 Q、非整块 Q、多 N tile、K 分段及两支物理 padding。分别比较 act、gate 与融合结果，避免只观察最终误差而把布局问题归因于 activation 近似。

本卡保留双分支合同与验证方法，不把来源中的具体模板名、固定 tile、局部近零近似或修正 kernel 当作已验证的通用实现。
