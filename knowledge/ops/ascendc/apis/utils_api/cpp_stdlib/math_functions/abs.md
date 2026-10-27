---
type: API
title: abs（数学函数）
description: 获取输入数据的绝对值。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/math_functions/abs.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
---

# abs

## 功能说明

获取输入数据的绝对值。

## 函数原型

```
template <typename T>
__aicore__ inline T abs(const T src)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 含义 |
| --- | --- |
| T | 输入数据src的数据类型。当前支持的数据类型为int8_t、int16_t、int32_t、float、int64_t。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 含义 |
| --- | --- | --- |
| src | 输入 | 源操作数。 |

## 约束说明

无

## 返回值说明

输入数据的绝对值。

## 调用示例

```
int64_t src = -2;

int64_t result = AscendC::Std::abs(src);
// result: 2
```
