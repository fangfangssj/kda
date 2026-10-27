---
type: API
title: CastFloatToBfloat16
description: 将float类型标量数据转换为bfloat16_t类型标量数据。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.scalar
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/scalar_compute/Cast_float_to_bfloat16.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_scalar_intf.h
---

# Cast（float转bfloat16\_t）<a name="ZH-CN_TOPIC_0000001623365980"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_scalar_intf.h"`。

float类型标量数据转换成bfloat16\_t类型标量数据。

在使用该接口之前，用户需要先了解[内置数据类型](../data_structures/builtin_data_types.md)和[精度转换](../data_structures/precision_conversion.md)。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline bfloat16_t Cast(const float& fVal)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

| 参数名称 | 输入/输出 | 描述 |
| ------ | ------ | ------ |
| fVal | 输入 | float类型标量数据。 |

## 返回值说明<a name="section640mcpsimp"></a>

转换后的bfloat16\_t类型标量数据。

## 约束说明<a name="section633mcpsimp"></a>

- float到bfloat16\_t转换采用[CAST\_RINT舍入](../data_structures/precision_conversion.md#二进制精度舍入规则)。
- 输入参数为立即数时，需要使用显式类型转换后输入，例如：

    ```cpp
    // 错误写法，AscendC暂不支持整数类型转bfloat16_t，编译报错
    bfloat16_t x = AscendC::Cast(1);
    // 正确写法
    bfloat16_t x = AscendC::Cast((float)1);
    ```

## 调用示例<a name="section19372434133520"></a>

```cpp
float m = 3.14159f;
bfloat16_t n = AscendC::Cast(m);  // n = 3.140625  float到bfloat16_t进行了舍入
```
