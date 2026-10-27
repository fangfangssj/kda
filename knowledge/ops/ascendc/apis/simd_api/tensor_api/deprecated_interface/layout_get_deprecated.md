---
type: API
title: layout::Get（废弃）
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/deprecated_interface/layout_get_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# layout::Get（废弃）

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

**该接口已废弃，请使用[layout::get](../layout/layout.md#get)替代。**

按照模板索引从Layout中提取Shape、Stride中的元素或子结构。

## 函数原型

```cpp
template <size_t... I>
__aicore__ inline constexpr decltype(auto) Get();

template <size_t... I>
__aicore__ inline constexpr decltype(auto) Get() const;
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| I... | 输入 | 模板参数。首个索引0表示Shape，1表示Stride；后续索引用于继续访问嵌套子结构。 |

## 返回值说明

返回指定位置的元素或子结构引用；const重载返回常量引用。

## 流水类型

无

## 约束说明

`I...`必须是Layout及其嵌套元组中的有效索引。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto& old_value = layout.Get<0, 1>();       // 废弃写法
auto& new_value = layout.get<0, 1>();       // 替代写法
```
