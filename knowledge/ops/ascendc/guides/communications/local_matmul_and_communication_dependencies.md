---
type: Guide
title: MC2 本地 Matmul 与通信依赖的重叠
description: 将不依赖远端输入的本地计算从通信块循环中独立调度，并保持贡献归属、输出位置及独立 Tiling 合同。
tags:
- task.optimization
- topic.performance.optimization.optimization_overview
- topic.data_movement
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/comm-compute/local_matmul.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/mc2/local_matmul_design.md
  role: supporting
created_at: '2026-09-10T14:24:50Z'
updated_at: '2026-09-10T14:24:50Z'
platforms: []
---

# MC2 本地 Matmul 与通信依赖的重叠

## 适用条件

在部分 MC2 算子中，本 rank 的输入贡献无需经过远端通信。当它与远端贡献的依赖可以分开时，可以用这段本地 Matmul 填充通信开始或结束附近的计算空隙。这里的“本地”是数据依赖属性；不能仅按 rank 编号把仍需归约或重排的数据视为最终结果。

## 调度与结果归属

对通信产生输入的路径，可先启动本地 Matmul，同时通信侧准备远端数据，之后计算侧逐块等待并处理远端贡献。对计算产生待发送结果的路径，可优先计算远端需要的部分，再在通信进行时处理本地部分。

独立出来的 self 贡献必须从原通信块循环中排除或采用明确的组合规则，避免重复计算、重复累加或遗漏。独立结果与远端结果写入不同区域还是归约到同一输出，由算子语义决定；后者需要明确累加初值、完成顺序与数值精度。

本地部分可以使用独立的 Tiling。其 shape 应来自本地计算实际覆盖域；来源所述“全量 M”属于对应实现的参数含义，不是所有集合通信布局的通用公式。

## 收益边界与验证

前置本地计算若短于首次通信，后续仍可能等待；它也可能因小 shape、资源竞争或额外启动成本而没有收益。前置、后置与保留在原循环均是候选策略，需要在同样的输入、rank 数与测量口径下比较端到端时间。

验证包括 self 和远端贡献恰好各出现一次、输出坐标一致、独立 Tiling 尾块正确，以及 trace 中确实存在本地计算与通信重叠。不能通过删除必要的通信等待来获得表面加速。
