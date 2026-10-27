---
type: API
title: GetSaturationFlag
description: 获取计算或精度转换时饱和模式的开启状态。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.system_variable
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/special_register_access/GetSaturationFlag_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_common_intf.h
---

# GetSaturationFlag(ISASI)

## 功能说明

头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。

获取`mode`指定的全局饱和模式的开启状态，该状态通过[SetSaturationFlag](set_saturation_flag_isasi.md)设置。饱和模式用于处理计算或精度转换过程中结果溢出或出现INF/NAN的场景，详细说明请参考[SetSaturationFlag功能说明](set_saturation_flag_isasi.md#功能说明)。

## 函数原型

```cpp
template <SaturationMode mode>
__aicore__ inline bool GetSaturationFlag()
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| mode | 饱和模式，取值如下：<br>&bull;`SaturationMode::FLOAT`：浮点数计算和浮点数精度转换的饱和模式，系统默认开启。<br>&bull;`SaturationMode::FLOAT8`：浮点数精度转换时，控制FP8类型输出为NAN时是否饱和为0，系统默认开启。该模式仅在精度转换接口处于饱和模式时生效，具体请参考[约束说明](#约束说明)。<br>&bull;`SaturationMode::INT`：整数计算的饱和模式，系统默认关闭。<br>&bull;`SaturationMode::CAST`：浮点数转整数或整数转整数时的精度转换饱和模式，系统默认开启。<br>各模式影响的数据类型请参考[SetSaturationFlag的数据类型](set_saturation_flag_isasi.md#数据类型)。模板参数无默认值，调用本接口时必须显式指定。 |

## 返回值说明

返回`true`表示`mode`指定的全局饱和模式已开启，返回`false`表示该模式已关闭。

## 约束说明


- Ascend 950PR&950DT系列产品支持`SaturationMode::FLOAT`、`SaturationMode::FLOAT8`、`SaturationMode::INT`和`SaturationMode::CAST`。
- 查询`SaturationMode::FLOAT8`时，返回值仅表示该模式的开启状态。该模式仅在精度转换接口处于饱和模式时生效，可通过全局`SaturationMode::FLOAT`或单接口`satMode`进入饱和模式。


- Atlas A3系列产品支持`SaturationMode::FLOAT`、`SaturationMode::INT`和`SaturationMode::CAST`，不支持`SaturationMode::FLOAT8`。


- Atlas A2系列产品支持`SaturationMode::FLOAT`、`SaturationMode::INT`和`SaturationMode::CAST`，不支持`SaturationMode::FLOAT8`。


## 调用示例

如下示例中查询整数计算的饱和模式是否开启。

```cpp
bool enableSat = AscendC::GetSaturationFlag<AscendC::SaturationMode::INT>();
```
