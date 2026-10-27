---
type: API
title: shape
description: shape用于定义张量的逻辑形状，用于描述各维度的大小。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/shape.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# shape

shape用于定义张量的逻辑形状，用于描述各维度的大小。

头文件路径为：`tensor_api/tensor.h`。

## 原型定义

```cpp
template <typename... Shapes>
using shape = Std::tuple<Shapes...>;
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|----------|------|
| Shapes... | 输入 | 可变参数模板，表示各维度的形状值，也可表示嵌套的shape子结构。 |

## API映射关系

shape通过[make_shape](make_shape.md)函数创建，通过[get_shape](get_shape.md)函数和layout的成员[shape](layout.md)获取。
