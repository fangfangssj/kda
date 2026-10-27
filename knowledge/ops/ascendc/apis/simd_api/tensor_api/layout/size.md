---
type: API
title: size
description: 用于返回layout的有效元素个数。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/size.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# size

## 功能概述

用于返回layout的有效元素个数。

## 函数原型

```cpp
template <size_t... Is, typename Layout>
__aicore__ inline constexpr auto size(const Layout& layout);
```

### 参数说明

**表1**  模板参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| Is... | size_t | 索引序列，用于编译时递归选择shape和stride的子结构。 |
| Layout | 输入 | layout对象的类型。 |

**表2**  参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| layout | 输入 | 用于描述张量的布局。 |

### 返回值

返回layout的有效元素个数。

### 约束说明

Is...必须为有效范围内的索引。

### 示例代码

   ```cpp
   #include "tensor_api/tensor.h"

   using namespace asc::te;

  auto layout = make_layout(make_shape(10, 20), make_stride(1, 100));
  auto size_result = size(layout); // size_result = 200
  ```
