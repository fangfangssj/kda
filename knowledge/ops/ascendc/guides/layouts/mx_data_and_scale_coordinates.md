---
type: Guide
title: MX 矩阵加载的数据与 Scale 坐标
description: 将逻辑元素、打包字节、NZ 分形和 Scale 载体分别建模，避免跨 dtype 复制 LoadData 偏移公式。
tags:
- task.design
- topic.data_movement.layout_transform
- topic.compute.quantization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-loaddata.md
  role: primary
created_at: '2026-09-10T12:53:40Z'
updated_at: '2026-09-10T12:53:40Z'
platforms: []
---

# MX 矩阵加载的数据与 Scale 坐标

LoadData 系列存在不同加载方向、布局及 MX 数据/scale 变体。字段的单位由具体重载决定，不能因为名字同为 stride 或 step，就统一解释为元素数或 32 字节。

对按 32 字节组成一个 C0 块的特定 NZ 表示，C0 中可容纳的逻辑元素数随位宽变化：16 bit 对应 16 个，8 bit 对应 32 个，4 bit 对应 64 个。这是特定表示的容量推导，不意味着每一种 Cube 输入布局或 LoadData 参数都采用同一规则。4 bit 打包尤其不能直接把 C++ 载体的 `sizeof` 当作单个逻辑元素的存储大小。

MX 的 scale 有自己的量化块粒度、打包方式和物理布局。数据起点、scale 起点及二者各自的 K/N/head 步进应分别推导，再验证它们仍描述同一逻辑 tile。沿 N 分段加载时，复用首段 scale 坐标会使后续段使用错误放缩；沿 K 切块时，边界也需满足所选量化块与加载接口约束。

多矩阵拼入 L1 时，head 间距应由完整物理布局确定，可能包含补齐和交错。仅凭 `head*mEff*C0` 这一示例公式无法证明任意 K 或任意拼接布局的地址正确。数据载体与 scale 载体的 head 间距也不必相同。

可用每个 head、N 段、K 段采用不同数据和 scale 的输入检查坐标对应关系；仅单 head 或偏移为零的首段通过，不能验证后续段。测试同时核对合法地址范围和数值，因为错位地址仍可能落在已分配区域内。
