---
type: Guide
title: CATLASS 的 Tile、Block 与 Kernel 组件契约
description: 说明矩阵计算、搬运、后处理与调度组件的职责，以及集成 DeviceGemm 或 Kernel 时需要核对的参数和生命周期。
tags:
- task.design
- topic.architecture
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/architecture/02-block-layer.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/architecture/01-tile-layer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/architecture/03-kernel-layer.md
  role: supporting
created_at: '2026-09-10T12:37:22Z'
updated_at: '2026-09-10T12:37:22Z'
platforms:
- a2
---

# CATLASS 的 Tile、Block 与 Kernel 组件契约

CATLASS 将矩阵计算组织成不同层级的组件。TileMmad 承担矩阵乘累加原语，TileCopy 承担 GM、L1、L0 与输出之间的相应搬运；BlockMmad 组合这些原语实现分块主循环；Kernel 进一步结合后处理与工作块调度。

| 组件 | 决定的行为 | 集成时的重要条件 |
|---|---|---|
| BlockMmad | 主循环、预加载和缓冲阶段 | DispatchPolicy 的实际特化、L1/L0 tile、输入与累计类型 |
| BlockEpilogue | 激活、偏置、反量化等后处理 | 输入源数量、计算长度、UB/workspace 与输出类型 |
| BlockScheduler | 不同计算块的访问顺序 | 覆盖范围、M/N 遍历和尾块 |
| Kernel | 组合调用与阶段衔接 | 是否实际调用所选 Epilogue、Params 布局、同步和 workspace |

改变一个组件可能同时改变其他层的契约。例如增加 Vector Epilogue 后，需要选择具有相应调用阶段的 Kernel，并重新核算中间结果交接和存储。仅将 BlockEpilogue 的模板实参从 void 改成一个类型，不能证明后处理会被执行。

Tile 层通常由 Block 的默认参数推导。需要特殊搬运时，可以替换对应 TileCopy 组件，但仍要核对地址、layout、对齐、尾部和同步语义。来源中小 M 的间隔搬运方案是具体实现候选，不构成所有小 M 输入的通用最优规则。

## Host 与 Device 边界

DeviceGemm 是 Host 侧适配器，可处理可实现性检查、workspace 和 stream 启动；算子 Device body 使用对应 Kernel 及其 Params。二者的 Arguments/Params 不保证同构，不能把 Host 资源管理代码直接搬入 op_kernel。

workspace 的解释取决于直调 launcher 或框架入口约定。应由实际调用方决定是否包含框架预留区，而不是把某个直调样例的指针透传规则推广到全部 ACLNN 工程。

## 适用与验证

本卡蒸馏固定 reference 中 AtlasA2 组件说明，未复核 CATLASS 上游实现版本，也未编译示意模板，因此保持 draft。策略名称、模板槽位、computeLength 单位和精确 Params 必须从目标版本声明确认。

验证组件组合时，应覆盖纯矩阵乘与真实后处理、各 dtype/layout、非整 tile 和空任务；同时检查结果、workspace 范围与每个阶段是否执行。Swizzle 的选择需结合矩阵形状和实测，不能将示例的 Offset=3 当成通用规则。
