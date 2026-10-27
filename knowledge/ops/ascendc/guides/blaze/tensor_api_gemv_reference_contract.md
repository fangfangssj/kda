---
type: Guide
title: 基础APIGEMV：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。GEMV的核心功能体现为：当矩阵A的维度M取值为1时，接口可以手动启用GEMV功能，该操作退化为1×K维度的行向量与K×N维度矩阵之间的乘法运算。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute_tensor_api/mmad_compute_key_features/gemv.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/tensor-api-reference.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute/mmad_compute_key_features/gemv.md
  role: constraint
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- Tensor API GEMV
- MmadTrait disableGemv
---


# GEMV（矩阵向量乘）

GEMV 在这里指 A 的 M=1、形状为 1×K 的行向量与 K×N 矩阵相乘的优化路径。它可以减少不必要的搬运和计算，但其他任意一维为 1 不足以证明匹配该接口的优化条件。

启用方式由具体接口和版本决定，不能统一写成自动识别开启。原资料涉及 half/float 等类型，实际类型与 shape 组合仍须满足对应 Mmad 支持矩阵。

## Te::MmadTrait 的默认设置

固定 reference 所述 DAV-3510/CANN 9.0.0-beta.2 接口使用 `bool MmadTrait::disableGemv=true`，即默认关闭 GEMV 优化。要考虑该快照的 GEMV 路线，需显式选择允许它的配置，并确认实际 dtype、布局、尺寸及实现支持；M=1 本身不证明已开启。

Tensor API 的编译期 MmadTrait 配置与运行时 MmadParams 分开。该快照的 MmadParams 只有 m、n、k、unitFlag、cmatrixInitVal 等列明字段，不能凭“自动识别”的概述补造运行时 GEMV 开关；也不能把其他版本模板或 Tiling 的启用机制直接套入。

基础 [Mmad GEMV](../../apis/simd_api/basic_api/cube_compute_isasi/mmad_compute_key_features/gemv.md) 说明另一接口形式的自动启用及 950 关闭选项；选择接口时应对齐版本与参数所在类型。验证先确认真实 M=1 的数学形状和合法 dtype，再对比允许/关闭优化时的结果及实际执行路径，不能只凭算子名或维度猜测性能收益。该 reference 没有固定上游 master 提交，因此不声明所有版本都使用相同默认值。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [GEMV](../../apis/simd_api/tensor_api/matrix_compute/mmad_compute_key_features/gemv.md)
