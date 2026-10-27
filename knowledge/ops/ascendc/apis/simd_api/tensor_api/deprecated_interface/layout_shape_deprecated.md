---
type: API
title: layout::Shape（废弃）
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/deprecated_interface/layout_shape_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# layout::Shape（废弃）

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

**该接口已废弃，请使用[layout::shape](../layout/layout.md#shape)替代。**

获取Layout中的Shape对象或指定子维度对应的Shape子结构。

## 函数原型

```cpp
template <size_t... I>
__aicore__ inline constexpr decltype(auto) Shape();

template <size_t... I>
__aicore__ inline constexpr decltype(auto) Shape() const;
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| I... | 输入 | 模板参数，表示需要访问的子维度索引序列。不指定时处理完整Layout。 |

## 返回值说明

返回Shape对象或其子结构的引用；const重载返回常量引用。

## 流水类型

无

## 约束说明

`I...`必须是Shape中的有效索引。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto& old_shape = layout.Shape();           // 废弃写法
auto& new_shape = layout.shape();           // 替代写法
```
