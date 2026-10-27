---
type: API
title: CastMultiTypeToFloat
description: 将bfloat16_t、fp8、fp4等多种类型标量数据转换为float类型。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.scalar
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/scalar_compute/Cast_multi_type_to_float.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_scalar_intf.h
---

# Cast（多类型转float）<a name="ZH-CN_TOPIC_0000001623525812"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_scalar_intf.h"`。

该接口将输入数据转换为float类型。

在使用该接口之前，用户需要先了解[内置数据类型](../data_structures/builtin_data_types.md)和[精度转换](../data_structures/precision_conversion.md)。

## 函数原型<a name="section620mcpsimp"></a>


以下函数原型仅支持：


- Atlas A2系列产品

 

- Atlas A3系列产品


bfloat16\_t类型转换为float类型：

```cpp
__aicore__ inline float Cast(const bfloat16_t& bVal)
```



以下函数原型仅支持Ascend 950PR&950DT系列产品：

支持多种数据类型转换为float类型：

```cpp
template <typename T, typename U = float,
        typename = Std::enable_if_t<
        (Std::is_same<T, bfloat16_t>::value || Std::is_same<T, hifloat8_t>::value ||
            Std::is_same<T, fp8_e5m2_t>::value || Std::is_same<T, fp8_e4m3fn_t>::value ||
            Std::is_same<T, fp4x2_e1m2_t>::value || Std::is_same<T, fp4x2_e2m1_t>::value), 
        void>>
__aicore__ constexpr inline U Cast(T bVal)
```


## 参数说明<a name="section622mcpsimp"></a>


**表1**  模板参数说明

| 参数名 | 描述 |
| ------ | ------ |
| T | 操作数数据类型。 |
| U | 返回值数据类型。 |



**表2**  接口参数说明

| 参数名称 | 输入/输出 | 含义               |
| -------- | --------- | ------------------ |
| bVal     | 输入      | 待转换的标量数据。 |

## 数据类型<a name="zh-cn_topic_0000002562161257_section4219135304818"></a>

- 支持的操作数数据类型如下：

  
  - Ascend 950PR&950DT系列产品，支持fp4x2_e2m1_t、fp4x2_e1m2_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、bfloat16_t。
  
  
  - Atlas A3系列产品，支持bfloat16_t。
  
  
  - Atlas A2系列产品，支持bfloat16_t。
  


- fp4x2_e1m2_t和fp4x2_e2m1_t类型为了满足1byte数据大小，构造时由两个四位标量数据拼接生成。转换时，被转换的是fp4x2_e1m2_t和fp4x2_e2m1_t标量数据中低位4bit的值，即[0:3]位。


- 支持的返回值数据类型为float。

## 返回值说明<a name="section640mcpsimp"></a>

转换后的float类型标量数据。

## 约束说明<a name="section633mcpsimp"></a>

输入参数为立即数时，需要使用显式类型转换，例如：

```cpp
// 错误写法，AscendC暂不支持整数类型转float，编译报错
float x = AscendC::Cast(1);
// 正确写法
float x = AscendC::Cast((bfloat16_t)1);
```

## 调用示例<a name="section19372434133520"></a>

```cpp
float fVal = 3.14159;
bfloat16_t bVal = AscendC::Cast(fVal); // 返回3.140625
fVal = AscendC::Cast(bVal);            // 返回3.140625
```
