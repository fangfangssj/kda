---
type: Guide
title: Matmul 高阶 API 的执行模式与 Host 契约
description: 将纯 Cube、消息驱动 MIX 和双主模式分别匹配 Tiling、参与核、结果取得方式与数值模式。
tags:
- task.implementation
- topic.tiling
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-matmul.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-host-runtime.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_pure_cube.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/aic_aiv_independent_execution.md
  role: supporting
created_at: '2026-09-10T12:53:40Z'
updated_at: '2026-09-10T12:53:40Z'
platforms:
- a2
- a3
---

# Matmul 高阶 API 的执行模式与 Host 契约

Matmul 的实际执行模式决定由哪类核发起工作、哪些核必须运行，以及如何取得结果。纯 Cube 直接在 AIC 执行，默认消息驱动 MIX 由 AIV 发起消息，双主模式则让 AIC/AIV 独立运行代码。因此，“AIV 直接返回”只适用于对应的纯 Cube 路径，不能作为所有 Matmul 的统一入口守卫。

Host 的参与核数和 Kernel 的 block 映射必须采用同一执行模型。设备的 AI Core、Cube Core 和 Vector Core 资源数不是通用同义字段；启动数还受工作量、调度模式和同步参与者约束。不能硬编码某一芯片的 20/40 核示例，也不能把混合算子的 block 数简单等同于某一硬件计数。

## Tiling 与结果接口

类型、ND/NZ 布局、转置、原矩阵跨度、单核 shape 与尾块范围共同决定地址和计算区域。Tiling 对象的存储地址空间还必须匹配初始化接口所需的指针：对 LocalTensor 的物理地址作普通指针强转，不会改变存储空间。

`IterateAll` 与逐次 `Iterate`/`GetTensorC` 的选择要匹配实际模板。已有 [双主模式约束](../operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/aic_aiv_independent_execution.md)明确该模式仅支持特定模板和输出方式；不能根据一个纯 Cube 示例为所有模式推荐相同取结果路径。

## 数值与验证

Host 配置的计算类型应与 Kernel 中的 HF32 等数值模式一致。HF32 将 FP32 输入按较低有效精度参与矩阵计算，可能改变误差；吞吐收益依赖硬件与 shape，不能固定承诺 4 倍。比较结果时也需确认参考框架采用的数值模式。

验证应包含不同执行模式、转置、dtype、尾块和多次迭代，确认每个输出均被实际写入。输出全零不足以单独证明某个 `MatmulConfig` 开关缺失；应结合路由、Tiling 与输出路径判断。
