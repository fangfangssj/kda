---
type: API
title: SetHF32TransMode
description: 设置HF32模式取整的具体方式，需要先使用SetHF32Mode开启HF32取整模式。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute_aux_config/SetHF32TransMode.md
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

# SetHF32TransMode

## 功能说明

设置HF32模式取整的具体方式，需要先使用[SetHF32Mode](set_hf32_mode.md)开启HF32取整模式。

## 函数原型

```cpp
__aicore__ inline void SetHF32TransMode(HF32TransMode mode)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| mode | 输入 | Mmad HF32取整模式控制入参，HF32TransMode类型。支持如下两种取值：<br>&nbsp;&nbsp;&bull; NEAREST_AWAY：FP32将以向最接近的值舍入，平局时远离零的方式舍入为HF32。<br>&nbsp;&nbsp;&bull; NEAREST_EVEN：FP32将以向最接近的值舍入，平局时向偶数舍入的方式舍入为HF32。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

- 设置HF32模式取整的具体方式，需要先使用[SetHF32Mode](set_hf32_mode.md)开启HF32取整模式。
- 本接口兼容枚举值`NEAREST_ZERO`，但不推荐使用，实际行为等同于`NEAREST_AWAY`。

```cpp
AscendC::SetHF32TransMode(HF32TransMode::NEAREST_AWAY);
```

- [MhcPreSinkhorn A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md)
- [Conv2D 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/conv/conv_2d_950.md)
