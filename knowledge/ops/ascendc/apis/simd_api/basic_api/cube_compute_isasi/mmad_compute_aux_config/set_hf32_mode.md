---
type: API
title: SetHF32Mode
description: 用于设置Mmad计算是否开启HF32模式，开启该模式后L0A Buffer/L0B Buffer中的FP32数据将在参与Mmad计算之前被舍入为HF32。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute_aux_config/SetHF32Mode.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_mm_intf.h
---

# SetHF32Mode

## 功能说明

用于设置Mmad计算是否开启HF32模式，开启该模式后L0A Buffer/L0B Buffer中的FP32数据将在参与Mmad计算之前被舍入为HF32。

## 函数原型

```cpp
__aicore__ inline void SetHF32Mode(HF32Mode mode)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| mode | 输入 | Mmad HF32模式控制入参，HF32Mode枚举类型。支持如下两种取值：<br>&nbsp;&nbsp;&bull; ENABLE：L0A Buffer/L0B Buffer中的FP32数据将在矩阵乘法之前被舍入为HF32。<br>&nbsp;&nbsp;&bull; DISABLE：将执行常规的FP32矩阵乘法。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
AscendC::SetHF32Mode(HF32Mode::ENABLE); // 控制mmad计算时是否使用HF32精度进行计算。
```

- [MhcPreSinkhorn A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md)
- [Conv2D 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/conv/conv_2d_950.md)

## 配套 API 与常见组合

- [SetHF32TransMode](set_hf32_trans_mode.md)——SetHF32TransMode 卡片明示"需先使用 SetHF32Mode 开启 HF32 取整模式"
