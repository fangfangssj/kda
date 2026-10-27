---
type: API
title: GetSaturationStrategy
description: 头文件路径为："basic_api/kernel_operator_common_intf.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/special_register_access/GetSaturationStrategy.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_common_intf.h
---

# GetSaturationStrategy

## 功能说明

头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。

获取精度转换接口当前使用的饱和模式配置策略。饱和模式用于处理精度转换过程中结果溢出或出现INF/NAN的场景，详细说明请参考[SetSaturationFlag功能说明](set_saturation_flag_isasi.md#功能说明)。该接口用于查询通过[SetSaturationStrategy](set_saturation_strategy.md)选择的配置来源，主要与Memory矢量计算的[Cast](../memory_vector_compute/type_conversion/cast.md)和Reg矢量计算的[Cast](../reg_vector_compute/type_conversion/cast.md)配合使用：

- 使用接口自身的饱和配置时，Reg矢量计算Cast通过`CastTrait::satMode`配置单接口的饱和或非饱和模式；Memory矢量计算Cast不提供单接口饱和模式参数，其单接口配置为饱和模式。
- 使用全局饱和配置时，两类Cast均使用[SetSaturationFlag](set_saturation_flag_isasi.md)设置的全局饱和模式。

## 函数原型

```cpp
__aicore__ inline OverrideStrategy GetSaturationStrategy()
```

## 参数说明

无

## 返回值说明

返回值类型为`OverrideStrategy`，表示当前使用的饱和模式配置策略。

**表1**  返回值说明

| 返回值 | 描述 |
| --- | --- |
| `OverrideStrategy::USE_API` | 使用精度转换接口自身的饱和配置。 |
| `OverrideStrategy::USE_GLOBAL` | 使用[SetSaturationFlag](set_saturation_flag_isasi.md)设置的全局饱和配置，系统默认采用该策略。 |

## 约束说明

无

## 调用示例

如下示例中查询当前是否使用`SetSaturationFlag`设置的全局饱和模式。

```cpp
bool useGlobal = AscendC::GetSaturationStrategy() == AscendC::OverrideStrategy::USE_GLOBAL;
```
